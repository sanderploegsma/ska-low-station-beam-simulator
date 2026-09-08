package common

import (
	"strconv"
	"testing"
)

// BenchmarkHeapAccumulator_OneTickPerPop mirrors ScanRunner's real
// per-tick usage: exactly one HeapLen-row Add per pol, then one
// PopReadyHeaps call, repeated -- the case that runs on every tick of a
// real scan (see synth.DirectSynthesisStreamer.TickNSamples's doc
// comment: "HeapLen by construction", so this is the only shape
// PopReadyHeaps ever actually sees in production, not just one shape
// among many). Releases each popped heap immediately, standing in for
// spead.encodeHeapInto -- without it, PrepareWrite's buffer pool never
// gets refilled and this would silently measure the pre-pool cost
// instead (see BenchmarkProducerTick's doc comment for the same note).
func BenchmarkHeapAccumulator_OneTickPerPop(b *testing.B) {
	for _, numChannels := range []int{96, 384} {
		b.Run("channels="+strconv.Itoa(numChannels), func(b *testing.B) {
			acc := NewHeapAccumulator(numChannels, 0, 1.0, nil)
			vChunk := makeChunk(numChannels, HeapLen, func(r, c int) complex128 { return complex(float64(r), float64(c)) })
			hChunk := makeChunk(numChannels, HeapLen, func(r, c int) complex128 { return complex(float64(c), float64(r)) })
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				acc.Add("V", vChunk)
				acc.Add("H", hChunk)
				heaps := acc.PopReadyHeaps()
				if len(heaps) != numChannels {
					b.Fatalf("expected %d heaps, got %d", numChannels, len(heaps))
				}
				for _, heap := range heaps {
					ReleaseSampleBuffers(heap)
				}
			}
		})
	}
}

// makeChunk builds a flat, CHANNEL-MAJOR (numChannels, rows) chunk
// (index = ch*rows+row), matching Streamer.GenerateNextTick's real
// output layout -- see common.Streamer's doc comment.
func makeChunk(numChannels, rows int, valueFor func(row, ch int) complex128) []complex128 {
	chunk := make([]complex128, rows*numChannels)
	for c := 0; c < numChannels; c++ {
		for r := 0; r < rows; r++ {
			chunk[c*rows+r] = valueFor(r, c)
		}
	}
	return chunk
}

func TestHeapAccumulator_NoHeapUntilFull(t *testing.T) {
	numChannels := 3
	acc := NewHeapAccumulator(numChannels, 0, 1.0, nil)
	chunk := makeChunk(numChannels, HeapLen-1, func(r, c int) complex128 { return complex(float64(r), float64(c)) })
	acc.Add("V", chunk)
	acc.Add("H", chunk)
	if heaps := acc.PopReadyHeaps(); len(heaps) != 0 {
		t.Fatalf("expected no heaps before HeapLen rows are buffered, got %d", len(heaps))
	}
}

func TestHeapAccumulator_EmitsOneHeapPerChannel(t *testing.T) {
	numChannels := 4
	acc := NewHeapAccumulator(numChannels, 100.0, 2.0, nil)

	vChunk := makeChunk(numChannels, HeapLen, func(r, c int) complex128 { return complex(float64(r), float64(c)) })
	hChunk := makeChunk(numChannels, HeapLen, func(r, c int) complex128 { return complex(float64(c), float64(r)) })
	acc.Add("V", vChunk)
	acc.Add("H", hChunk)

	heaps := acc.PopReadyHeaps()
	if len(heaps) != numChannels {
		t.Fatalf("expected %d heaps (one per channel), got %d", numChannels, len(heaps))
	}

	seenChannels := map[int]bool{}
	for _, h := range heaps {
		seenChannels[h.ChannelID] = true
		if len(h.VSamples) != HeapLen || len(h.HSamples) != HeapLen {
			t.Fatalf("channel %d: sample slice length wrong: v=%d h=%d", h.ChannelID, len(h.VSamples), len(h.HSamples))
		}
		// obs_time + samples_consumed(0)/sample_rate(2.0) = 100.0
		if h.HeapStartTime != 100.0 {
			t.Fatalf("channel %d: HeapStartTime = %v, want 100.0", h.ChannelID, h.HeapStartTime)
		}
		// Spot-check a couple of samples landed in the right column.
		if real(h.VSamples[5]) != 5 || imag(h.VSamples[5]) != float64(h.ChannelID) {
			t.Fatalf("channel %d: VSamples[5] = %v, want (5+%di)", h.ChannelID, h.VSamples[5], h.ChannelID)
		}
	}
	for ch := 0; ch < numChannels; ch++ {
		if !seenChannels[ch] {
			t.Fatalf("channel %d missing from emitted heaps", ch)
		}
	}
}

func TestHeapAccumulator_SecondHeapAdvancesStartTime(t *testing.T) {
	numChannels := 1
	sampleRate := 4.0
	acc := NewHeapAccumulator(numChannels, 0.0, sampleRate, nil)

	full := makeChunk(numChannels, HeapLen, func(r, c int) complex128 { return 0 })
	acc.Add("V", full)
	acc.Add("H", full)
	acc.Add("V", full)
	acc.Add("H", full)

	heaps := acc.PopReadyHeaps()
	if len(heaps) != 2 {
		t.Fatalf("expected 2 heaps from 2*HeapLen buffered rows, got %d", len(heaps))
	}
	if heaps[0].HeapStartTime != 0.0 {
		t.Fatalf("first heap HeapStartTime = %v, want 0.0", heaps[0].HeapStartTime)
	}
	wantSecond := float64(HeapLen) / sampleRate
	if heaps[1].HeapStartTime != wantSecond {
		t.Fatalf("second heap HeapStartTime = %v, want %v", heaps[1].HeapStartTime, wantSecond)
	}
}

func TestHeapAccumulator_PrepareWriteDirectFill(t *testing.T) {
	// Exercises the real ScanRunner path -- PrepareWrite then write
	// directly into the returned per-channel targets -- rather than
	// building a chunk and going through Add.
	numChannels := 3
	acc := NewHeapAccumulator(numChannels, 100.0, 2.0, nil)

	vTargets := acc.PrepareWrite("V", HeapLen)
	hTargets := acc.PrepareWrite("H", HeapLen)
	if len(vTargets) != numChannels || len(hTargets) != numChannels {
		t.Fatalf("PrepareWrite returned %d/%d targets, want %d", len(vTargets), len(hTargets), numChannels)
	}
	for ch := 0; ch < numChannels; ch++ {
		if len(vTargets[ch]) != HeapLen || len(hTargets[ch]) != HeapLen {
			t.Fatalf("channel %d: target length v=%d h=%d, want %d", ch, len(vTargets[ch]), len(hTargets[ch]), HeapLen)
		}
		for i := 0; i < HeapLen; i++ {
			vTargets[ch][i] = complex(float64(i), float64(ch))
			hTargets[ch][i] = complex(float64(ch), float64(i))
		}
	}

	heaps := acc.PopReadyHeaps()
	if len(heaps) != numChannels {
		t.Fatalf("expected %d heaps, got %d", numChannels, len(heaps))
	}
	for _, h := range heaps {
		if real(h.VSamples[5]) != 5 || imag(h.VSamples[5]) != float64(h.ChannelID) {
			t.Fatalf("channel %d: VSamples[5] = %v, want (5+%di) -- direct-write target wasn't reflected in the popped heap", h.ChannelID, h.VSamples[5], h.ChannelID)
		}
		if real(h.HSamples[5]) != float64(h.ChannelID) || imag(h.HSamples[5]) != 5 {
			t.Fatalf("channel %d: HSamples[5] = %v, want (%d+5i)", h.ChannelID, h.HSamples[5], h.ChannelID)
		}
	}
}

func TestHeapAccumulator_ReleaseAndReusePreservesCorrectness(t *testing.T) {
	numChannels := 2
	acc := NewHeapAccumulator(numChannels, 0, 1.0, nil)

	vTargets := acc.PrepareWrite("V", HeapLen)
	hTargets := acc.PrepareWrite("H", HeapLen)
	for ch := 0; ch < numChannels; ch++ {
		for i := 0; i < HeapLen; i++ {
			vTargets[ch][i] = complex(1.0, float64(ch))
			hTargets[ch][i] = complex(2.0, float64(ch))
		}
	}
	heaps := acc.PopReadyHeaps()
	if len(heaps) != numChannels {
		t.Fatalf("expected %d heaps, got %d", numChannels, len(heaps))
	}
	for _, h := range heaps {
		ReleaseSampleBuffers(h)
	}

	// sync.Pool doesn't guarantee these released buffers come back on
	// the next PrepareWrite (they might not, depending on GC timing) --
	// but IF they do, they must come back fully overwritten, with no
	// stale content from the first tick surviving.
	vTargets = acc.PrepareWrite("V", HeapLen)
	hTargets = acc.PrepareWrite("H", HeapLen)
	for ch := 0; ch < numChannels; ch++ {
		for i := 0; i < HeapLen; i++ {
			vTargets[ch][i] = complex(3.0, float64(ch))
			hTargets[ch][i] = complex(4.0, float64(ch))
		}
	}
	heaps = acc.PopReadyHeaps()
	for _, h := range heaps {
		for i, v := range h.VSamples {
			if v != complex(3.0, float64(h.ChannelID)) {
				t.Fatalf("channel %d: VSamples[%d] = %v, want (3+%di) -- stale released data leaked through", h.ChannelID, i, v, h.ChannelID)
			}
		}
		for i, v := range h.HSamples {
			if v != complex(4.0, float64(h.ChannelID)) {
				t.Fatalf("channel %d: HSamples[%d] = %v, want (4+%di)", h.ChannelID, i, v, h.ChannelID)
			}
		}
	}
}

func TestReleaseSampleBuffers_NilSafe(t *testing.T) {
	ReleaseSampleBuffers(nil)
	ReleaseSampleBuffers(&ChannelHeap{})
}

func TestHeapAccumulator_CustomChannelIDMap(t *testing.T) {
	numChannels := 2
	channelIDMap := []int{64, 72} // e.g. a station's first_channel_id offset
	acc := NewHeapAccumulator(numChannels, 0, 1.0, channelIDMap)

	full := makeChunk(numChannels, HeapLen, func(r, c int) complex128 { return 0 })
	acc.Add("V", full)
	acc.Add("H", full)

	heaps := acc.PopReadyHeaps()
	gotIDs := map[int]bool{}
	for _, h := range heaps {
		gotIDs[h.ChannelID] = true
	}
	for _, want := range channelIDMap {
		if !gotIDs[want] {
			t.Fatalf("expected a heap with ChannelID=%d (from channelIDMap), got IDs %v", want, gotIDs)
		}
	}
}
