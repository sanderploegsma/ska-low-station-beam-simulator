package common

import (
	"runtime"
	"sync"
)

// HeapAccumulator buffers per-channel samples until HeapLen samples are
// available per channel, then emits one ChannelHeap per channel.
//
// Storage is PER-CHANNEL (bufV[ch]/bufH[ch], each its own growable,
// contiguous []complex128), not one shared flat buffer — deliberately
// matching Streamer.GenerateNextTick's channel-major (numChannels,
// nSamples) output layout (see that interface's doc comment). This
// replaced an earlier flat, sample-major design that had to TRANSPOSE
// every tick's chunk into per-channel order via a scalar, cache-hostile
// loop (768 small allocations/tick at 384 channels, plus the strided
// copy itself) — see BenchmarkHeapAccumulator_OneTickPerPop and
// BenchmarkProducerTick, which caught this as the dominant per-tick
// cost, exceeding the ENTIRE per-tick budget at 384 channels on its own.
// With generation already producing channel-major output, Add/pop reduce
// to per-channel bulk memmoves (append/copy), never a per-element loop —
// the same total bytes moved, but via the runtime's optimized memmove
// instead of a scalar loop, and with far fewer, far larger allocations.
//
// Add and PopReadyHeaps both split their per-channel work across
// numWorkers goroutines (see forEachChannelRange): profiling a real
// end-to-end run on the target hardware found ScanRunner's single
// producer goroutine (this type plus DirectSynthesisStreamer.
// GenerateNextTick) running at ~99% duty cycle on ONE core for the
// whole scan, unable to keep pace even after removing every allocation
// this method used to make -- unlike sending (already parallelized
// across SenderPool's goroutines), the producer had never been split
// across cores at all. Each channel's work here is already fully
// independent (disjoint bufV[ch]/bufH[ch] slices), so this is a direct
// channel-range split, the same approach GenerateNextTick uses for its
// own dominant cost.
type HeapAccumulator struct {
	numChannels          int
	obsTime              float64
	sampleRatePerChannel float64
	channelIDMap         []int
	numWorkers           int

	bufV, bufH      [][]complex128 // per-channel, len == numChannels; bufV[ch] grows as ticks are added
	rowsV, rowsH    int            // samples buffered per channel so far (same for every channel: Add always delivers every channel's share together)
	samplesConsumed int64          // total per-channel samples already popped, for timestamping
}

// NewHeapAccumulator creates an accumulator for numChannels channels.
// channelIDMap maps a column index in the arriving chunks to the
// external channel_id to label it with — pass nil for identity (what
// synth.DirectSynthesisStreamer always produces, since its output is
// already in external ascending-channel-id order).
func NewHeapAccumulator(numChannels int, obsTime, sampleRatePerChannel float64, channelIDMap []int) *HeapAccumulator {
	if channelIDMap == nil {
		channelIDMap = make([]int, numChannels)
		for i := range channelIDMap {
			channelIDMap[i] = i
		}
	}
	return &HeapAccumulator{
		numChannels:          numChannels,
		obsTime:              obsTime,
		sampleRatePerChannel: sampleRatePerChannel,
		channelIDMap:         channelIDMap,
		numWorkers:           defaultParallelism(numChannels),
		bufV:                 make([][]complex128, numChannels),
		bufH:                 make([][]complex128, numChannels),
	}
}

// defaultParallelism caps worker/goroutine counts consistently with
// internal/synth's identical helper (kept separate, not shared, since
// common must not depend on synth): runtime.GOMAXPROCS(0) (which —
// unlike runtime.NumCPU() — respects a Kubernetes pod's CPU
// request/limit) capped to n and to 16.
func defaultParallelism(n int) int {
	w := runtime.GOMAXPROCS(0)
	if w > n {
		w = n
	}
	if w > 16 {
		w = 16
	}
	if w < 1 {
		w = 1
	}
	return w
}

// forEachChannelRange splits [0, numChannels) into numWorkers contiguous
// ranges and runs fn on each in its own goroutine, waiting for all to
// finish before returning. numWorkers<=1 (or too few channels to split)
// runs fn once, synchronously, with no goroutine spawned at all.
func forEachChannelRange(numWorkers, numChannels int, fn func(chStart, chEnd int)) {
	if numWorkers <= 1 || numChannels <= 1 {
		fn(0, numChannels)
		return
	}
	chunkSize := (numChannels + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	for chStart := 0; chStart < numChannels; chStart += chunkSize {
		chEnd := chStart + chunkSize
		if chEnd > numChannels {
			chEnd = numChannels
		}
		wg.Add(1)
		go func(chStart, chEnd int) {
			defer wg.Done()
			fn(chStart, chEnd)
		}(chStart, chEnd)
	}
	wg.Wait()
}

// Add appends one tick's chunk -- flat, CHANNEL-MAJOR (numChannels,
// nSamples), index = channel*nSamples+sample, matching
// Streamer.GenerateNextTick's output layout -- for the given
// polarisation ("V" or "H"). Each channel's nSamples segment is a
// contiguous run in chunk, so distributing it into bufV/bufH is
// numChannels bulk appends, never a per-element copy -- and, since each
// channel's append is independent of every other's, split across
// a.numWorkers goroutines by channel range.
func (a *HeapAccumulator) Add(pol string, chunk []complex128) {
	if len(chunk) == 0 {
		return
	}
	nSamples := len(chunk) / a.numChannels
	bufs := a.bufV
	if pol == "H" {
		bufs = a.bufH
	}
	forEachChannelRange(a.numWorkers, a.numChannels, func(chStart, chEnd int) {
		for ch := chStart; ch < chEnd; ch++ {
			bufs[ch] = append(bufs[ch], chunk[ch*nSamples:(ch+1)*nSamples]...)
		}
	})
	switch pol {
	case "V":
		a.rowsV += nSamples
	case "H":
		a.rowsH += nSamples
	}
}

// PopReadyHeaps pops every fully-buffered heap (HeapLen samples available
// per channel on both pols) and returns one ChannelHeap per channel, per
// popped block.
func (a *HeapAccumulator) PopReadyHeaps() []*ChannelHeap {
	var heaps []*ChannelHeap
	for a.rowsV >= HeapLen && a.rowsH >= HeapLen {
		heapStartTime := a.obsTime + float64(a.samplesConsumed)/a.sampleRatePerChannel
		a.samplesConsumed += HeapLen

		// One flat buffer per pol per pop iteration (2 allocations, not
		// 2*numChannels) -- each ChannelHeap's VSamples/HSamples is a
		// sub-slice into it, filled via a bulk copy per channel (not a
		// per-element loop: each channel's HeapLen samples are already
		// contiguous in bufV[ch]/bufH[ch]).
		flatV := make([]complex128, a.numChannels*HeapLen)
		flatH := make([]complex128, a.numChannels*HeapLen)
		iterHeaps := make([]*ChannelHeap, a.numChannels)

		forEachChannelRange(a.numWorkers, a.numChannels, func(chStart, chEnd int) {
			for ch := chStart; ch < chEnd; ch++ {
				copy(flatV[ch*HeapLen:(ch+1)*HeapLen], a.bufV[ch])
				copy(flatH[ch*HeapLen:(ch+1)*HeapLen], a.bufH[ch])

				iterHeaps[ch] = &ChannelHeap{
					ChannelID:     a.channelIDMap[ch],
					VSamples:      flatV[ch*HeapLen : (ch+1)*HeapLen],
					HSamples:      flatH[ch*HeapLen : (ch+1)*HeapLen],
					HeapStartTime: heapStartTime,
				}

				// Shift each channel's leftover tail down IN PLACE
				// (reusing its backing array's capacity) rather than
				// reallocating -- in the normal case (one HeapLen-sample
				// tick per pop) the leftover is empty and this is a
				// no-op.
				remV := copy(a.bufV[ch], a.bufV[ch][HeapLen:])
				a.bufV[ch] = a.bufV[ch][:remV]
				remH := copy(a.bufH[ch], a.bufH[ch][HeapLen:])
				a.bufH[ch] = a.bufH[ch][:remH]
			}
		})

		heaps = append(heaps, iterHeaps...)
		a.rowsV -= HeapLen
		a.rowsH -= HeapLen
	}
	return heaps
}
