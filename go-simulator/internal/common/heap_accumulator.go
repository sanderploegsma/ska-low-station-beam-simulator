package common

// HeapAccumulator buffers per-channel samples (a flat, row-major
// (nSamples, numChannels) slice per tick — index = sample*numChannels+ch,
// matching the layout Streamer.GenerateNextTick returns) until HeapLen
// rows are available per channel, then emits one ChannelHeap per
// channel.
type HeapAccumulator struct {
	numChannels          int
	obsTime              float64
	sampleRatePerChannel float64
	channelIDMap         []int

	bufV, bufH      []complex128 // flat, row-major (rows, numChannels)
	rowsV, rowsH    int
	samplesConsumed int64 // total per-channel samples already popped, for timestamping
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
	}
}

// Add appends one tick's chunk (flat, row-major (nSamples, numChannels))
// for the given polarisation ("V" or "H").
func (a *HeapAccumulator) Add(pol string, chunk []complex128) {
	if len(chunk) == 0 {
		return
	}
	rows := len(chunk) / a.numChannels
	switch pol {
	case "V":
		a.bufV = append(a.bufV, chunk...)
		a.rowsV += rows
	case "H":
		a.bufH = append(a.bufH, chunk...)
		a.rowsH += rows
	}
}

// PopReadyHeaps pops every fully-buffered heap (HeapLen rows available on
// both pols) and returns one ChannelHeap per channel, per popped block.
//
// Two allocation/access-pattern fixes here, found via
// BenchmarkHeapAccumulator_OneTickPerPop after this project's own
// producer started measurably falling behind its per-tick budget at 96
// channels and badly so at 384 (mirroring the Python codebase's bug #13:
// allocation churn, not compute, turned out to be the dominant per-tick
// cost). Both fixes are about EVERY-TICK cost, since
// TickNSamples()/Add() always deliver exactly one HeapLen-row chunk per
// pol per tick in production (see
// synth.DirectSynthesisStreamer.TickNSamples's doc comment) — this loop
// runs its body once per tick, every tick, for the lifetime of a scan:
//
//  1. The old code allocated TWO fresh HeapLen-length slices PER
//     CHANNEL, per tick (768 small allocations/tick at 384 channels).
//     Fixed: one flat (numChannels*HeapLen) destination buffer per pol
//     per pop iteration (2 allocations total, not 2*numChannels), each
//     ChannelHeap's VSamples/HSamples now a sub-slice into it. The
//     per-channel loop order (read vBlock/hBlock strided, write
//     SEQUENTIALLY into that channel's own contiguous slice of
//     flatV/flatH) is kept as-is, deliberately, not "fixed" to read
//     sequentially/write strided — that swap was tried and MEASURED
//     WORSE (2.5-2.6x, not better): flatV/flatH are as large as
//     vBlock/hBlock themselves, so scattering writes across one
//     (instead of reading from one) just moves the cache pressure onto
//     the write side, which glibc/Go's write-allocate caches punish
//     just as hard; the original read-strided/write-sequential order
//     benefits from vBlock/hBlock being small enough to stay resident
//     across the whole channel loop, which a naive "fix" based on
//     reasoning about which side is `the source` rather than measuring
//     would have missed.
//  2. The old code reset bufV/bufH to a fresh nil-backed allocation
//     every tick (`append(nil, bufV[HeapLen*numChannels:]...)`),
//     discarding capacity and forcing Add's next append to reallocate
//     the whole HeapLen*numChannels block again. Fixed: shift the
//     leftover tail down IN PLACE (copy + reslice) so the same backing
//     array's capacity is reused across ticks — in the (normal) case
//     where a tick's Add exactly empties the buffer, the leftover is
//     zero-length and this is a no-op.
func (a *HeapAccumulator) PopReadyHeaps() []*ChannelHeap {
	var heaps []*ChannelHeap
	for a.rowsV >= HeapLen && a.rowsH >= HeapLen {
		vBlock := a.bufV[:HeapLen*a.numChannels]
		hBlock := a.bufH[:HeapLen*a.numChannels]

		heapStartTime := a.obsTime + float64(a.samplesConsumed)/a.sampleRatePerChannel
		a.samplesConsumed += HeapLen

		flatV := make([]complex128, a.numChannels*HeapLen) // channel-major: index = ch*HeapLen+i
		flatH := make([]complex128, a.numChannels*HeapLen)
		for ch := 0; ch < a.numChannels; ch++ {
			vSamples := flatV[ch*HeapLen : (ch+1)*HeapLen]
			hSamples := flatH[ch*HeapLen : (ch+1)*HeapLen]
			for i := 0; i < HeapLen; i++ {
				vSamples[i] = vBlock[i*a.numChannels+ch]
				hSamples[i] = hBlock[i*a.numChannels+ch]
			}
			heaps = append(heaps, &ChannelHeap{
				ChannelID:     a.channelIDMap[ch],
				VSamples:      vSamples,
				HSamples:      hSamples,
				HeapStartTime: heapStartTime,
			})
		}

		remainingV := copy(a.bufV, a.bufV[HeapLen*a.numChannels:])
		a.bufV = a.bufV[:remainingV]
		remainingH := copy(a.bufH, a.bufH[HeapLen*a.numChannels:])
		a.bufH = a.bufH[:remainingH]
		a.rowsV -= HeapLen
		a.rowsH -= HeapLen
	}
	return heaps
}
