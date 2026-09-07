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
func (a *HeapAccumulator) PopReadyHeaps() []*ChannelHeap {
	var heaps []*ChannelHeap
	for a.rowsV >= HeapLen && a.rowsH >= HeapLen {
		vBlock := a.bufV[:HeapLen*a.numChannels]
		hBlock := a.bufH[:HeapLen*a.numChannels]

		heapStartTime := a.obsTime + float64(a.samplesConsumed)/a.sampleRatePerChannel
		a.samplesConsumed += HeapLen

		for ch := 0; ch < a.numChannels; ch++ {
			vSamples := make([]complex128, HeapLen)
			hSamples := make([]complex128, HeapLen)
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

		a.bufV = append([]complex128(nil), a.bufV[HeapLen*a.numChannels:]...)
		a.bufH = append([]complex128(nil), a.bufH[HeapLen*a.numChannels:]...)
		a.rowsV -= HeapLen
		a.rowsH -= HeapLen
	}
	return heaps
}
