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
//
// PopReadyHeaps hands off each channel's bufV[ch]/bufH[ch] backing array
// directly as the outgoing ChannelHeap's VSamples/HSamples -- it does
// NOT copy into a separate flat buffer first. An earlier version did:
// re-profiling a real end-to-end run AFTER the goroutine-parallelization
// above (this section) found runtime.memmove alone at 56.9% of ALL CPU
// time, with this method's flatV/flatH copy (plus the make()/zero-fill
// that came with it) responsible for over 14 points of that on its own
// -- parallelizing had spread the cost across cores without eliminating
// the redundant work, which is exactly why CPU usage rose sharply for
// only a marginal throughput gain rather than closing the pacing gap.
// Since TickNSamples() == HeapLen by construction (see that method's doc
// comment on DirectSynthesisStreamer), bufV[ch]/bufH[ch] holds EXACTLY
// HeapLen samples at the moment PopReadyHeaps runs in the real pipeline
// -- so handing off the array Add already built (one copy, in Add's own
// append) replaces a second full copy with a zero-copy reslice.
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

// defaultParallelism picks a worker/goroutine count consistently with
// internal/synth's identical helper (kept separate, not shared, since
// common must not depend on synth): runtime.GOMAXPROCS(0) (which —
// unlike runtime.NumCPU() — respects a Kubernetes pod's CPU
// request/limit), capped only to n.
//
// This used to also cap at a flat 16, matching fillNoiseBank's identical
// cap for its one-time noise-bank *construction* cost -- wrong to share
// here, caught by real-hardware profiling: this method's work runs on
// EVERY tick under the fixed per-tick budget, not once at startup, and
// the Python CLAUDE.md's own EPYC benchmarking already established that
// this class of per-channel-independent work keeps scaling well past 16
// threads once allocation overhead is out of the way (see its "Target
// server results" section: throughput kept improving monotonically up
// to 96 threads). A real profile on 2-socket EPYC target hardware showed
// average concurrency pinned at ~15.18 -- suspiciously exactly this cap
// -- while the machine had far more cores sitting idle and pacing was
// still falling behind. Removed; GOMAXPROCS is now trusted on its own,
// same as it already is for NumWorkers callers who set it explicitly.
func defaultParallelism(n int) int {
	w := runtime.GOMAXPROCS(0)
	if w > n {
		w = n
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

		iterHeaps := make([]*ChannelHeap, a.numChannels)

		// HAND OFF each channel's existing backing array as VSamples/
		// HSamples directly -- NO copy into a separate flat buffer.
		// Profiling a real end-to-end run (384 channels, target
		// hardware) after the goroutine-parallelization fix above found
		// this exact copy dominating: 56.9% of ALL CPU time was
		// runtime.memmove, and this method's flatV/flatH copy alone
		// (plus the make()/zero-fill that came with it) was over 14% on
		// its own -- parallelizing spread that cost across cores but
		// never eliminated the redundant work, which is why CPU usage
		// went up markedly for only a slight throughput gain rather than
		// closing the pacing gap. In the normal case (TickNSamples() ==
		// HeapLen, true by construction -- see that method's doc
		// comment) each bufV[ch]/bufH[ch] holds EXACTLY HeapLen samples
		// here, so takeHeapSlice below just reslices the array Add
		// already built (one copy, in Add's append) instead of copying
		// it a second time.
		takeHeapSlice := func(buf []complex128) (heapSlice, remainder []complex128) {
			heapSlice = buf[:HeapLen:HeapLen] // 3-index: cap HeapLen so a future append by any holder can't alias into the leftover below
			leftoverLen := len(buf) - HeapLen
			if leftoverLen == 0 {
				return heapSlice, nil
			}
			// Only reached if a caller ever delivers a tick whose
			// nSamples doesn't evenly divide HeapLen -- not the case
			// for DirectSynthesisStreamer today, but keep it correct
			// rather than assuming. The old backing array is now owned
			// by heapSlice, so the leftover must be copied OUT into a
			// fresh array, not shifted in place.
			remainder = make([]complex128, leftoverLen)
			copy(remainder, buf[HeapLen:])
			return heapSlice, remainder
		}

		forEachChannelRange(a.numWorkers, a.numChannels, func(chStart, chEnd int) {
			for ch := chStart; ch < chEnd; ch++ {
				vHeap, vRem := takeHeapSlice(a.bufV[ch])
				hHeap, hRem := takeHeapSlice(a.bufH[ch])
				a.bufV[ch] = vRem
				a.bufH[ch] = hRem

				iterHeaps[ch] = &ChannelHeap{
					ChannelID:     a.channelIDMap[ch],
					VSamples:      vHeap,
					HSamples:      hHeap,
					HeapStartTime: heapStartTime,
				}
			}
		})

		heaps = append(heaps, iterHeaps...)
		a.rowsV -= HeapLen
		a.rowsH -= HeapLen
	}
	return heaps
}
