package common

import (
	"runtime"
	"sync"
)

// HeapAccumulator buffers per-channel samples until HeapLen samples are
// available per channel, then emits one ChannelHeap per channel.
//
// Storage is PER-CHANNEL (bufV[ch]/bufH[ch], each its own growable,
// contiguous []complex64), matching Streamer.GenerateNextTick's
// channel-major (numChannels, nSamples) output layout (see that
// interface's doc comment) -- Add/PopReadyHeaps reduce to per-channel
// bulk memmoves (append/copy), never a per-element loop.
//
// Add and PopReadyHeaps both split their per-channel work across
// numWorkers goroutines (see forEachChannelRange): each channel's work
// is fully independent (disjoint bufV[ch]/bufH[ch] slices), so this is
// a direct channel-range split, the same approach GenerateNextTick uses
// for its own dominant cost.
//
// PopReadyHeaps hands off each channel's bufV[ch]/bufH[ch] backing array
// directly as the outgoing ChannelHeap's VSamples/HSamples -- it does
// NOT copy into a separate flat buffer first. Since TickNSamples() ==
// HeapLen by construction (see that method's doc comment on
// DirectSynthesisStreamer), bufV[ch]/bufH[ch] holds EXACTLY HeapLen
// samples at the moment PopReadyHeaps runs in the real pipeline -- so
// handing off the array Add already built (one copy, in Add's own
// append) replaces a second full copy with a zero-copy reslice.
type HeapAccumulator struct {
	numChannels          int
	obsTime              float64
	sampleRatePerChannel float64
	channelIDMap         []int
	numWorkers           int

	bufV, bufH      [][]complex64 // per-channel, len == numChannels; bufV[ch] grows as ticks are added
	rowsV, rowsH    int           // samples buffered per channel so far (same for every channel: Add always delivers every channel's share together)
	samplesConsumed int64         // total per-channel samples already popped, for timestamping
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
		bufV:                 make([][]complex64, numChannels),
		bufH:                 make([][]complex64, numChannels),
	}
}

// defaultParallelism picks a worker/goroutine count consistently with
// internal/synth's identical helper (kept separate, not shared, since
// common must not depend on synth): runtime.GOMAXPROCS(0) (which —
// unlike runtime.NumCPU() — respects a Kubernetes pod's CPU
// request/limit), capped only to n.
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

// sampleBufferPool holds reusable, HeapLen-capacity []complex64 buffers
// -- shared package-wide, not per-HeapAccumulator, since a buffer is
// fungible once its previous contents have been fully consumed (see
// ReleaseSampleBuffers): there is nothing accumulator-, channel-, or
// pol-specific baked into the memory itself.
var sampleBufferPool = sync.Pool{
	New: func() any {
		return make([]complex64, HeapLen)
	},
}

// getSampleBuffer draws a HeapLen-length buffer from sampleBufferPool
// (allocating one, via New above, if the pool is empty). Content is
// UNDEFINED -- possibly stale from a previous tick's samples, never
// zeroed -- which is safe here specifically because PrepareWrite's
// caller (GenerateNextTick's noise fill, or its no-noise-configured
// zero-fill branch) always WRITES every cell before anything ever reads
// it; nothing in this codebase relies on a fresh buffer starting at
// zero.
func getSampleBuffer() []complex64 {
	return sampleBufferPool.Get().([]complex64)
}

// quantizedBufferPool holds reusable, HeapLen*2-capacity []byte buffers
// (real,imag int8 pairs, interleaved) for ChannelHeap.VQuantized/
// HQuantized -- the pre-quantized counterpart to sampleBufferPool, same
// reuse discipline (GetQuantizedBuffer/ReleaseSampleBuffers), just a
// quarter the size per sample (2 bytes vs. complex64's 8).
var quantizedBufferPool = sync.Pool{
	New: func() any {
		return make([]byte, HeapLen*2)
	},
}

// GetQuantizedBuffer draws a HeapLen*2-length buffer from
// quantizedBufferPool (allocating one if the pool is empty). Content is
// UNDEFINED, same as getSampleBuffer -- safe only because every caller
// (synth.DirectSynthesisStreamer.GenerateQuantizedHeaps) always writes
// every byte before anything reads it.
func GetQuantizedBuffer() []byte {
	return quantizedBufferPool.Get().([]byte)
}

// ReleaseSampleBuffers returns heap.VSamples/HSamples (and/or
// VQuantized/HQuantized, whichever this heap actually used -- see
// ChannelHeap's doc comment) to their respective pools for reuse by a
// future PrepareWrite/GetQuantizedBuffer call, once heap has been fully
// consumed (spead.BatchSendLoop calls this right after SPEAD-encoding a
// heap, successfully or not -- either way nothing reads heap's sample
// data again). Only buffers with the pool's exact capacity are pooled
// (exactly what PrepareWrite's fast path, PopReadyHeaps' zero-copy
// handoff, and GetQuantizedBuffer always produce) -- anything else (the
// rare nSamples > HeapLen case PrepareWrite falls back to plain make()
// for) is simply left for the garbage collector, no correctness impact
// either way, just a missed reuse. Safe to call with heap == nil or with
// nil/short sample slices.
func ReleaseSampleBuffers(heap *ChannelHeap) {
	if heap == nil {
		return
	}
	if cap(heap.VSamples) == HeapLen {
		sampleBufferPool.Put(heap.VSamples[:HeapLen])
	}
	if cap(heap.HSamples) == HeapLen {
		sampleBufferPool.Put(heap.HSamples[:HeapLen])
	}
	if cap(heap.VQuantized) == HeapLen*2 {
		quantizedBufferPool.Put(heap.VQuantized[:HeapLen*2])
	}
	if cap(heap.HQuantized) == HeapLen*2 {
		quantizedBufferPool.Put(heap.HQuantized[:HeapLen*2])
	}
}

// WarmBufferPools pre-populates sampleBufferPool and quantizedBufferPool
// with numChannels*2 buffers each (covering V+H for every channel,
// regardless of which pool a given channel's path actually draws from --
// an oversized Put on the "wrong" pool is harmless, just a few unused
// entries that age out on the next GC like anything else in a
// sync.Pool) before a scan's first tick ever runs.
//
// Exists to close a real gap: sync.Pool starts completely empty --
// process-fresh for a one-shot CLI run, and just as importantly, EVERY
// entry a pool held is dropped on each GC cycle even in a long-running
// process (a Tango device server pod handling many scans over its
// lifetime) -- so if a real gap between scans lets even one GC cycle
// land in between, the NEXT scan starts with empty pools too, not just
// the very first scan a process ever runs. Without this, the first
// several hundred ticks of every scan would pay make()'s cost on every
// Get() until enough buffers have cycled through ReleaseSampleBuffers to
// fill the pool "for free".
func WarmBufferPools(numChannels int) {
	n := numChannels * 2
	for i := 0; i < n; i++ {
		sampleBufferPool.Put(make([]complex64, HeapLen))
		quantizedBufferPool.Put(make([]byte, HeapLen*2))
	}
}

// PrepareWrite grows each channel's buffer for pol by nSamples and
// returns, per channel, the newly-added nSamples-length slice as a
// direct write target -- the caller (ScanRunner, via
// synth.DirectSynthesisStreamer.GenerateNextTick) writes generated
// samples straight into these instead of building a separate chunk that
// then has to be copied in via Add.
//
// Generation writes directly into these slices (bank tile -> here, one
// copy), rather than building a separate chunk that Add would then have
// to copy in a second time.
//
// Draws from sampleBufferPool instead of make()-ing a fresh buffer in
// the common case (growing from empty, size <= HeapLen): pooled buffers
// carry stale content by design (see getSampleBuffer), which is safe
// here since GenerateNextTick's noise fill overwrites every cell of a
// freshly-grown buffer anyway, and it lets this skip make()'s
// unconditional zero-fill.
//
// Each channel's growth is independent (disjoint bufV[ch]/bufH[ch]
// slices), so -- like Add and PopReadyHeaps -- this is split across
// a.numWorkers goroutines by channel range.
func (a *HeapAccumulator) PrepareWrite(pol string, nSamples int) [][]complex64 {
	if nSamples <= 0 {
		return nil
	}
	bufs := a.bufV
	if pol == "H" {
		bufs = a.bufH
	}
	targets := make([][]complex64, a.numChannels)
	forEachChannelRange(a.numWorkers, a.numChannels, func(chStart, chEnd int) {
		for ch := chStart; ch < chEnd; ch++ {
			old := bufs[ch]
			oldLen := len(old)
			newLen := oldLen + nSamples
			switch {
			case cap(old) >= newLen:
				bufs[ch] = old[:newLen]
			case newLen <= HeapLen:
				pooled := getSampleBuffer()
				copy(pooled, old) // no-op in the normal case: old is empty right after the previous PopReadyHeaps handoff
				bufs[ch] = pooled[:newLen]
			default:
				grown := make([]complex64, newLen)
				copy(grown, old)
				bufs[ch] = grown
			}
			targets[ch] = bufs[ch][oldLen:newLen]
		}
	})
	switch pol {
	case "V":
		a.rowsV += nSamples
	case "H":
		a.rowsH += nSamples
	}
	return targets
}

// Add appends one tick's chunk -- flat, CHANNEL-MAJOR (numChannels,
// nSamples), index = channel*nSamples+sample -- for the given
// polarisation ("V" or "H"). A thin convenience wrapper around
// PrepareWrite for callers that already have a fully-built chunk (tests,
// BenchmarkHeapAccumulator_OneTickPerPop); ScanRunner's real per-tick
// path calls PrepareWrite directly instead, so generation can write into
// the target slices without ever building chunk in the first place (see
// PrepareWrite's doc comment).
func (a *HeapAccumulator) Add(pol string, chunk []complex64) {
	if len(chunk) == 0 {
		return
	}
	nSamples := len(chunk) / a.numChannels
	targets := a.PrepareWrite(pol, nSamples)
	forEachChannelRange(a.numWorkers, a.numChannels, func(chStart, chEnd int) {
		for ch := chStart; ch < chEnd; ch++ {
			copy(targets[ch], chunk[ch*nSamples:(ch+1)*nSamples])
		}
	})
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
		// HSamples directly -- NO copy into a separate flat buffer. In
		// the normal case (TickNSamples() == HeapLen, true by
		// construction -- see that method's doc comment) each
		// bufV[ch]/bufH[ch] holds EXACTLY HeapLen samples here, so
		// takeHeapSlice below just reslices the array Add already
		// built (one copy, in Add's append) instead of copying it a
		// second time.
		takeHeapSlice := func(buf []complex64) (heapSlice, remainder []complex64) {
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
			remainder = make([]complex64, leftoverLen)
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
