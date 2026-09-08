package synth

import (
	"bytes"
	"math"
	"testing"

	"github.com/skao/station-beam-simulator-go/internal/common"
	"github.com/skao/station-beam-simulator-go/internal/spead"
)

func testStation() *common.StationConfig {
	return &common.StationConfig{StationID: 1, SubstationID: 0, SubarrayID: 1, BeamID: 1, ScanID: 1}
}

// newDst builds a fresh V/H destination map for GenerateNextTick -- one
// nSamples-long slice per channel, per pol, matching what
// HeapAccumulator.PrepareWrite hands the real ScanRunner every tick.
func newDst(numChannels, nSamples int) map[string][][]complex64 {
	dst := make(map[string][][]complex64, 2)
	for _, pol := range [...]string{"V", "H"} {
		chBufs := make([][]complex64, numChannels)
		for ch := range chBufs {
			chBufs[ch] = make([]complex64, nSamples)
		}
		dst[pol] = chBufs
	}
	return dst
}

// flatten concatenates dst[pol] (per-channel slices) into one flat,
// channel-major slice -- a convenience for tests written against the
// old flat-buffer return value.
func flatten(chBufs [][]complex64) []complex64 {
	if len(chBufs) == 0 {
		return nil
	}
	nSamples := len(chBufs[0])
	out := make([]complex64, 0, len(chBufs)*nSamples)
	for _, chBuf := range chBufs {
		out = append(out, chBuf...)
	}
	return out
}

func TestNewDirectSynthesisStreamer_RejectsInvalidNumChannels(t *testing.T) {
	// 7: below MinNumChannels and not a multiple of 8. 100: a multiple of
	// neither. 392: a multiple of 8 but over MaxNumChannels (384). Note
	// NumChannels=0 is deliberately NOT tested here -- it's the "use the
	// default (MaxNumChannels)" sentinel, covered by
	// TestNewDirectSynthesisStreamer_ZeroNumChannelsDefaultsToMax.
	for _, nc := range []int{7, 100, 392} {
		nc := nc
		_, err := NewDirectSynthesisStreamer(StreamerConfig{
			Station:     testStation(),
			ObsTimeRef:  0,
			NumChannels: nc,
		})
		if err == nil {
			t.Errorf("NumChannels=%d: expected a validation error, got none", nc)
		}
	}
}

func TestNewDirectSynthesisStreamer_ZeroNumChannelsDefaultsToMax(t *testing.T) {
	s, err := NewDirectSynthesisStreamer(StreamerConfig{Station: testStation(), ObsTimeRef: 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s.NumChannels() != common.MaxNumChannels {
		t.Fatalf("NumChannels() = %d, want default %d", s.NumChannels(), common.MaxNumChannels)
	}
}

func TestNewDirectSynthesisStreamer_RejectsToneSourceWithoutDelayFeed(t *testing.T) {
	_, err := NewDirectSynthesisStreamer(StreamerConfig{
		Station:    testStation(),
		ObsTimeRef: 0,
		ToneSources: []ToneSourceConfig{
			{DelayFeed: nil, FreqHz: 60e6, Amplitude: 1.0},
		},
	})
	if err == nil {
		t.Fatal("expected an error for a tone source with no delay_feed -- there is no default/fallback delay")
	}
}

func TestDirectSynthesisStreamer_ToneLandsInConfiguredChannel(t *testing.T) {
	numChannels := 96
	channelIdx := 20
	freqHz := common.BaseFreqHz + float64(channelIdx)*common.ChannelWidthHz + 500.0

	feed := common.NewDelayFeed("test-tone")
	feed.Update(&common.DelayPolynomial{
		StationID:         1,
		StartValiditySec:  0,
		ValidityPeriodSec: 1e9,
		XYPolCoeffsNs:     []float64{0.0},
		YPolOffsetNs:      0.0,
	})

	s, err := NewDirectSynthesisStreamer(StreamerConfig{
		Station:     testStation(),
		ObsTimeRef:  1000.0,
		NumChannels: numChannels,
		ToneSources: []ToneSourceConfig{
			{DelayFeed: feed, FreqHz: freqHz, Amplitude: 1.0},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The tone's channel must be the ONLY one using the complex
	// (dst-write) path -- every other channel is noise-only and never
	// reaches GenerateNextTick at all any more (see
	// ComplexPathChannelIDMap's doc comment).
	complexChannels := s.ComplexPathChannelIDMap()
	if len(complexChannels) != 1 || complexChannels[0] != channelIdx {
		t.Fatalf("ComplexPathChannelIDMap() = %v, want exactly [%d]", complexChannels, channelIdx)
	}

	n := s.TickNSamples()
	dst := newDst(len(complexChannels), n)
	s.GenerateNextTick(1000.0, n, dst)

	for _, pol := range []string{"V", "H"} {
		out := dst[pol]
		if len(out) != 1 {
			t.Fatalf("pol %s: len(out) = %d, want 1 (exactly the tone's channel)", pol, len(out))
		}
		mag := 0.0
		for i := 0; i < n; i++ {
			v := out[0][i]
			re, im := float64(real(v)), float64(imag(v))
			mag += re*re + im*im
		}
		if mag < 1e-6 {
			t.Fatalf("pol %s: expected tone energy in the tone's dst slot, got %v", pol, mag)
		}
	}
}

func TestDirectSynthesisStreamer_ToneOutsideRangeIsSkippedNotFatal(t *testing.T) {
	feed := common.NewDelayFeed("out-of-range")
	feed.Update(&common.DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1e9, XYPolCoeffsNs: []float64{0.0}})

	s, err := NewDirectSynthesisStreamer(StreamerConfig{
		Station:     testStation(),
		ObsTimeRef:  0,
		NumChannels: 8,
		ToneSources: []ToneSourceConfig{
			// Frequency far outside the configured 8-channel band.
			{DelayFeed: feed, FreqHz: common.BaseFreqHz + 1000*common.ChannelWidthHz, Amplitude: 1.0},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// An out-of-range tone maps to no valid channel, so it never enters
	// ComplexPathChannelIDMap() -- every channel here is noise-only.
	if got := s.ComplexPathChannelIDMap(); len(got) != 0 {
		t.Fatalf("ComplexPathChannelIDMap() = %v, want empty (the only tone is out of range)", got)
	}

	n := s.TickNSamples()
	dst := newDst(0, n)
	s.GenerateNextTick(0, n, dst) // must not panic against an empty dst

	// No noise configured either -- every channel's pre-quantized heap
	// must be all-zero.
	heaps := s.GenerateQuantizedHeaps(0)
	if len(heaps) != 8 {
		t.Fatalf("GenerateQuantizedHeaps returned %d heaps, want 8 (every channel, none of them tone-affected)", len(heaps))
	}
	for _, h := range heaps {
		for _, b := range h.VQuantized {
			if b != 0 {
				t.Fatalf("channel %d: expected all-zero VQuantized for an out-of-range tone with no noise configured, got byte %d", h.ChannelID, b)
			}
		}
	}
}

func TestDirectSynthesisStreamer_NoiseIsDeterministicAcrossRepeatedCalls(t *testing.T) {
	// No tone configured -- every channel is noise-only, so this exercises
	// GenerateQuantizedHeaps (see ComplexPathChannelIDMap's doc comment),
	// not GenerateNextTick.
	s, err := NewDirectSynthesisStreamer(StreamerConfig{
		Station:     testStation(),
		ObsTimeRef:  0,
		NumChannels: 8,
		Noise:       &NoiseConfig{Std: 0.1, Seed: 42},
		NTiles:      4,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	heapsA := s.GenerateQuantizedHeaps(5.0)
	heapsB := s.GenerateQuantizedHeaps(5.0)
	if len(heapsA) != len(heapsB) {
		t.Fatalf("heap count mismatch: %d vs %d", len(heapsA), len(heapsB))
	}
	for i := range heapsA {
		if !bytes.Equal(heapsA[i].VQuantized, heapsB[i].VQuantized) {
			t.Fatalf("channel %d: repeated call at the same t gave different noise", heapsA[i].ChannelID)
		}
	}
}

func TestDirectSynthesisStreamer_NoiseIndependentAcrossStationSeeds(t *testing.T) {
	buildV := func(seed int64) []byte {
		s, err := NewDirectSynthesisStreamer(StreamerConfig{
			Station:     testStation(),
			ObsTimeRef:  0,
			NumChannels: 8,
			Noise:       &NoiseConfig{Std: 1.0, Seed: seed},
			NTiles:      4,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		var flat []byte
		for _, h := range s.GenerateQuantizedHeaps(0) {
			flat = append(flat, h.VQuantized...)
		}
		return flat
	}

	stationA := buildV(1)
	stationB := buildV(2)

	identical := 0
	for i := range stationA {
		if stationA[i] == stationB[i] {
			identical++
		}
	}
	if identical == len(stationA) {
		t.Fatal("two stations with different noise seeds produced byte-identical noise -- cross-station independence is broken")
	}
}

func TestDirectSynthesisStreamer_VAndHNoiseDiffer(t *testing.T) {
	// An earlier bug (see docs/history.md) reused one noise source for
	// both pols, giving numerically identical "noise" for V and H. Guard
	// against regressing that here.
	s, err := NewDirectSynthesisStreamer(StreamerConfig{
		Station:     testStation(),
		ObsTimeRef:  0,
		NumChannels: 8,
		Noise:       &NoiseConfig{Std: 1.0, Seed: 42},
		NTiles:      4,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var v, h []byte
	for _, hp := range s.GenerateQuantizedHeaps(0) {
		v = append(v, hp.VQuantized...)
		h = append(h, hp.HQuantized...)
	}
	identical := 0
	for i := range v {
		if v[i] == h[i] {
			identical++
		}
	}
	if identical == len(v) {
		t.Fatal("V and H noise are byte-identical -- must use independent per-pol seeds")
	}
}

func TestDirectSynthesisStreamer_NoiseNeverDelayCorrected(t *testing.T) {
	// Receiver noise must be identical regardless of a source's delay
	// polynomial -- it never enters a delay pipeline. Checked on both
	// paths: the tone's own channel (complex, GenerateNextTick) and
	// every other, noise-only channel (pre-quantized,
	// GenerateQuantizedHeaps -- which doesn't even look at
	// toneCfgs/delay at all, a stronger structural guarantee than the
	// complex path's "noise generation never reads a DelayFeed").
	feedA := common.NewDelayFeed("a")
	feedA.Update(&common.DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1e9, XYPolCoeffsNs: []float64{0.0}})
	feedB := common.NewDelayFeed("b")
	feedB.Update(&common.DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1e9, XYPolCoeffsNs: []float64{1e6}}) // large delay

	build := func(feed *common.DelayFeed) (complexV []complex64, quantizedV []byte) {
		s, err := NewDirectSynthesisStreamer(StreamerConfig{
			Station:     testStation(),
			ObsTimeRef:  0,
			NumChannels: 8,
			Noise:       &NoiseConfig{Std: 1.0, Seed: 42},
			NTiles:      4,
			// Amplitude 1e-300, not 0: a literal 0.0 is treated by
			// NewDirectSynthesisStreamer as "unset" and defaults to
			// 1.0 (see StreamerConfig's doc comment) -- 1e-300 is
			// negligible enough to vanish entirely when added to an
			// O(1) noise sample at float64 precision, while still
			// exercising the real (non-defaulted) tone code path.
			ToneSources: []ToneSourceConfig{{DelayFeed: feed, FreqHz: common.BaseFreqHz, Amplitude: 1e-300}},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		n := s.TickNSamples()
		dst := newDst(len(s.ComplexPathChannelIDMap()), n)
		s.GenerateNextTick(0, n, dst)
		complexV = flatten(dst["V"])
		for _, h := range s.GenerateQuantizedHeaps(0) {
			quantizedV = append(quantizedV, h.VQuantized...)
		}
		return complexV, quantizedV
	}

	complexA, quantizedA := build(feedA)
	complexB, quantizedB := build(feedB)
	for i := range complexA {
		if complexA[i] != complexB[i] {
			t.Fatalf("complex-path (tone channel) sample %d differs between a zero-delay and a large-delay source (%v vs %v) -- noise must be delay-independent", i, complexA[i], complexB[i])
		}
	}
	if !bytes.Equal(quantizedA, quantizedB) {
		t.Fatal("pre-quantized noise-only channels differ between a zero-delay and a large-delay tone source -- they must never depend on any tone's delay at all")
	}
}

// TestGenerateQuantizedHeaps_MatchesComplexPathQuantizedWithSameFixedScale
// is the core numerical proof behind pre-quantizing the noise bank at
// construction time instead of the complex64-then-quantize-per-heap
// path: for the SAME underlying noise draw (same seed) and the SAME
// fixed scale, the two must produce IDENTICAL wire bytes. The only
// difference this change is allowed to make is WHEN the (already-
// approved, see QuantizeScale's doc comment) fixed-scale quantization
// happens -- construction time instead of every tick -- never WHAT it
// computes.
func TestGenerateQuantizedHeaps_MatchesComplexPathQuantizedWithSameFixedScale(t *testing.T) {
	const seed, std = 7, 0.05
	const numChannels = 4
	const nTiles, tileNSamples = 4, 16
	const scale = 127.0 / (8.0*std + 1e-12) // matches QuantizeScale()'s own formula

	complexBank := fillNoiseBank(seed, std, nTiles, tileNSamples, numChannels)
	quantBank := fillQuantizedNoiseBank(seed, std, scale, nTiles, tileNSamples, numChannels)

	if len(quantBank) != len(complexBank)*2 {
		t.Fatalf("quantized bank length = %d, want %d (2 bytes/sample)", len(quantBank), len(complexBank)*2)
	}
	for i, c := range complexBank {
		wantRe := spead.QuantizeComponent(float64(real(c)) * scale)
		wantIm := spead.QuantizeComponent(float64(imag(c)) * scale)
		gotRe := int8(quantBank[i*2])
		gotIm := int8(quantBank[i*2+1])
		if gotRe != wantRe || gotIm != wantIm {
			t.Fatalf("sample %d: pre-quantized bank = (%d,%d), want (%d,%d) -- same underlying draw (same seed), quantized differently", i, gotRe, gotIm, wantRe, wantIm)
		}
	}
}

func TestDirectSynthesisStreamer_QuantizeScale_NoSourcesIsZero(t *testing.T) {
	s, err := NewDirectSynthesisStreamer(StreamerConfig{Station: testStation(), ObsTimeRef: 0, NumChannels: 8})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := s.QuantizeScale(); got != 0 {
		t.Fatalf("QuantizeScale() with no noise/tone configured = %v, want 0 (moot -- every sample is zero anyway)", got)
	}
}

func TestDirectSynthesisStreamer_QuantizeScale_NoiseOnlyMatchesSigmaMargin(t *testing.T) {
	const std = 0.05
	s, err := NewDirectSynthesisStreamer(StreamerConfig{
		Station: testStation(), ObsTimeRef: 0, NumChannels: 8,
		Noise: &NoiseConfig{Std: std, Seed: 1},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := 127.0 / (quantizeSigmaMargin*std + 1e-12)
	if got := s.QuantizeScale(); math.Abs(got-want) > 1e-6 {
		t.Fatalf("QuantizeScale() = %v, want %v (127 / %v-sigma noise headroom)", got, want, quantizeSigmaMargin)
	}
}

func TestDirectSynthesisStreamer_QuantizeScale_SumsAllToneAmplitudes(t *testing.T) {
	feedA := common.NewDelayFeed("a")
	feedA.Update(&common.DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1e9, XYPolCoeffsNs: []float64{0.0}})
	feedB := common.NewDelayFeed("b")
	feedB.Update(&common.DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1e9, XYPolCoeffsNs: []float64{0.0}})

	s, err := NewDirectSynthesisStreamer(StreamerConfig{
		Station: testStation(), ObsTimeRef: 0, NumChannels: 8,
		ToneSources: []ToneSourceConfig{
			{DelayFeed: feedA, FreqHz: common.BaseFreqHz, Amplitude: 2.0},
			{DelayFeed: feedB, FreqHz: common.BaseFreqHz + 8*common.ChannelWidthHz, Amplitude: 3.0},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Worst case: both tones land in the same channel and add exactly in
	// phase -- the bound must cover 2.0+3.0=5.0, not just the larger of
	// the two, even though these two tones are actually configured in
	// different channels here.
	want := 127.0 / (5.0 + 1e-12)
	if got := s.QuantizeScale(); math.Abs(got-want) > 1e-6 {
		t.Fatalf("QuantizeScale() = %v, want %v (127 / sum-of-amplitudes)", got, want)
	}
}

func TestDelayFeed_ZeroDelayUntilFirstUpdate(t *testing.T) {
	feed := common.NewDelayFeed("unstarted")
	poly := feed.Get(123.0)
	if len(poly.XYPolCoeffsNs) != 1 || poly.XYPolCoeffsNs[0] != 0.0 {
		t.Fatalf("expected zero-delay coefficients before any Update, got %v", poly.XYPolCoeffsNs)
	}
	if math.IsInf(poly.ValidityPeriodSec, 0) == false {
		t.Fatalf("expected an infinite validity period for the zero-delay placeholder, got %v", poly.ValidityPeriodSec)
	}
}

func TestDelayFeed_KeepsApplyingExpiredPolynomial(t *testing.T) {
	feed := common.NewDelayFeed("stale")
	feed.Update(&common.DelayPolynomial{
		StartValiditySec:  0,
		ValidityPeriodSec: 1.0,
		XYPolCoeffsNs:     []float64{42.0},
	})
	// t=10 is well past valid_until=1.0 -- Get must still return the same
	// (now-stale) polynomial rather than substituting zero delay.
	poly := feed.Get(10.0)
	if poly.XYPolCoeffsNs[0] != 42.0 {
		t.Fatalf("expected the stale polynomial's coefficients to still be applied, got %v", poly.XYPolCoeffsNs)
	}
}
