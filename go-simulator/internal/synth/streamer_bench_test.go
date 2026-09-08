package synth

import (
	"strconv"
	"testing"

	"github.com/skao/station-beam-simulator-go/internal/common"
)

// BenchmarkProducerTick reproduces one ScanRunner tick end-to-end
// (HeapAccumulator.PrepareWrite -> GenerateNextTick -> PopReadyHeaps ->
// common.ReleaseSampleBuffers) -- the exact per-tick work whose
// real-time budget is common.BlockDurationS (~2.21ms), and whose
// overrun is what noise-stream's "producer falling behind pacing" log
// line is reporting. Noise-only (matches cmd/noise-stream, which sets
// no ToneSources), n_tiles left at DefaultNTiles.
//
// Releasing each popped heap's buffers immediately (standing in for
// spead.encodeHeapInto, which does the same once a real SenderPool
// finishes encoding a heap) matters for this benchmark's own validity,
// not just realism: without it, PrepareWrite's buffer pool never gets
// refilled, so every call falls back to a fresh make() regardless of
// the pool existing at all -- silently measuring the pre-pool cost
// again under a name that no longer describes it.
func BenchmarkProducerTick(b *testing.B) {
	for _, numChannels := range []int{96, 384} {
		b.Run("channels="+strconv.Itoa(numChannels), func(b *testing.B) {
			streamer, err := NewDirectSynthesisStreamer(StreamerConfig{
				Station:     testStation(),
				ObsTimeRef:  1_700_000_000.0,
				NumChannels: numChannels,
				Noise:       &NoiseConfig{Std: 0.05, Seed: 1},
			})
			if err != nil {
				b.Fatalf("NewDirectSynthesisStreamer: %v", err)
			}
			nSamples := streamer.TickNSamples()
			acc := common.NewHeapAccumulator(numChannels, 1_700_000_000.0, streamer.channelOutputRate, streamer.ChannelIDMap())

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				t := 1_700_000_000.0 + float64(i)*common.BlockDurationS
				dst := map[string][][]complex128{
					"V": acc.PrepareWrite("V", nSamples),
					"H": acc.PrepareWrite("H", nSamples),
				}
				streamer.GenerateNextTick(t, nSamples, dst)
				heaps := acc.PopReadyHeaps()
				if len(heaps) != numChannels {
					b.Fatalf("expected %d heaps, got %d", numChannels, len(heaps))
				}
				for _, heap := range heaps {
					common.ReleaseSampleBuffers(heap)
				}
			}
		})
	}
}
