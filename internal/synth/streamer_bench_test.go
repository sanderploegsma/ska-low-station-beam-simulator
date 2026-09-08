package synth

import (
	"strconv"
	"testing"

	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/common"
)

// BenchmarkProducerTick reproduces one ScanRunner tick end-to-end --
// HeapAccumulator.PrepareWrite -> GenerateNextTick -> PopReadyHeaps for
// whatever channels need the complex path (ComplexPathChannelIDMap),
// PLUS GenerateQuantizedHeaps for every other (noise-only) channel --
// mirroring common.ScanRunner.run's actual per-tick sequence, including
// common.ReleaseSampleBuffers for both kinds of heap. This is the exact
// per-tick work whose real-time budget is common.BlockDurationS
// (~2.21ms), and whose overrun is what noise-stream's "producer falling
// behind pacing" log line is reporting. Noise-only (matches
// cmd/noise-stream, which sets no ToneSources by default) -- so in
// practice EVERY channel here goes through GenerateQuantizedHeaps, and
// PrepareWrite/GenerateNextTick/PopReadyHeaps see zero channels, exactly
// as ComplexPathChannelIDMap intends. n_tiles left at DefaultNTiles.
//
// Releasing each heap's buffers immediately (standing in for
// spead.encodeHeapInto, which does the same once a real SenderPool
// finishes encoding a heap) matters for this benchmark's own validity,
// not just realism: without it, the buffer pools (both the complex64
// one AND the pre-quantized one) never get refilled, so every call
// falls back to a fresh make() regardless of the pool existing at all --
// silently measuring the pre-pool cost again under a name that no
// longer describes it.
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
			complexChannelIDMap := streamer.ComplexPathChannelIDMap()
			acc := common.NewHeapAccumulator(len(complexChannelIDMap), 1_700_000_000.0, streamer.channelOutputRate, complexChannelIDMap)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				t := 1_700_000_000.0 + float64(i)*common.BlockDurationS
				dst := map[string][][]complex64{
					"V": acc.PrepareWrite("V", nSamples),
					"H": acc.PrepareWrite("H", nSamples),
				}
				streamer.GenerateNextTick(t, nSamples, dst)
				heaps := acc.PopReadyHeaps()
				heaps = append(heaps, streamer.GenerateQuantizedHeaps(t)...)
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
