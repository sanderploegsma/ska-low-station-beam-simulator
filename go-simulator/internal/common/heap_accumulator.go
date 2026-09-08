package common

import (
	"runtime"
	"sync"
)

// HeapAccumulator buffers per-channel samples until HeapLen samples are
// available per channel, then emits one ChannelHeap per channel.
//
// Storage is PER-CHANNEL (bufV[ch]/bufH[ch], each its own growable,
// contiguous []complex64), not one shared flat buffer — deliberately
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

	bufV, bufH      [][]complex64 // per-channel, len == numChannels; bufV[ch] grows as ticks are added
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
		bufV:                 make([][]complex64, numChannels),
		bufH:                 make([][]complex64, numChannels),
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
// lifetime) -- so if a real gap between scans (multi-second-plus,
// control-software overhead, per the parent Python CLAUDE.md's
// documented deployment cadence) lets even one GC cycle land in between,
// the NEXT scan starts with empty pools too, not just the very first
// scan a process ever runs. Without this, the first several hundred
// ticks of every scan pay make()'s cost on every Get() until enough
// buffers have cycled through ReleaseSampleBuffers to fill the pool
// "for free" -- a plausible contributor to the early-scan pacing drift
// this project's real-hardware captures keep showing (see
// go-simulator/README.md's profiling log), independent of and
// additional to the CPU-clock-ramp/GC causes already investigated and
// ruled out or mitigated there.
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
// This exists because profiling a real end-to-end run (384 channels,
// EPYC target hardware) found the OLD flow -- generate into a scratch
// buffer, then Add() copies it into bufV/bufH -- moving every tick's
// samples TWICE: once out of the noise tile bank into the scratch
// buffer, once more out of the scratch buffer into here. Neither more
// threads (see defaultParallelism's doc comment: raising the worker cap
// left this exact copy volume unchanged, and pacing didn't improve) nor
// the earlier PopReadyHeaps fix (which removed a THIRD copy, out of here
// into a flat per-pop buffer) touched this one. Generating directly into
// PrepareWrite's returned slices removed it -- one copy (bank tile ->
// here) instead of two.
//
// Removing that copy surfaced the NEXT cost: re-profiling found
// runtime.memclrNoHeapPointers alone at 20.9% of all CPU time --
// make()'s unconditional zero-fill on every channel's freshly-grown
// buffer, every tick, even though GenerateNextTick's noise fill was
// about to overwrite every cell of it anyway (runtime.makeslicecopy,
// the make()+copy() fusion the old grow path compiled to, accounted for
// essentially all of this method's own cost: 140.16s of its 140.77s).
// Fixed by drawing from sampleBufferPool instead of make()-ing a fresh
// buffer in the common case (growing from empty, size <= HeapLen) --
// pooled buffers carry stale content by design (see getSampleBuffer),
// so skipping the zero-fill is safe.
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
