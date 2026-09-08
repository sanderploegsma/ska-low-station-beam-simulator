package synth

import (
	"math"
	"testing"

	"github.com/skao/station-beam-simulator-go/internal/common"
)

func testStation() *common.StationConfig {
	return &common.StationConfig{StationID: 1, SubstationID: 0, SubarrayID: 1, BeamID: 1, ScanID: 1}
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

	n := s.TickNSamples()
	result := s.GenerateNextTick(1000.0, n)

	for _, pol := range []string{"V", "H"} {
		out := result[pol]
		if len(out) != n*numChannels {
			t.Fatalf("pol %s: len(out) = %d, want %d", pol, len(out), n*numChannels)
		}
		// Energy should be concentrated in channelIdx, ~zero elsewhere.
		for ch := 0; ch < numChannels; ch++ {
			mag := 0.0
			chSamples := out[ch*n : (ch+1)*n] // channel-major: this channel's samples are contiguous
			for i := 0; i < n; i++ {
				v := chSamples[i]
				mag += real(v)*real(v) + imag(v)*imag(v)
			}
			if ch == channelIdx {
				if mag < 1e-6 {
					t.Fatalf("pol %s: expected tone energy in channel %d, got %v", pol, ch, mag)
				}
			} else if mag > 1e-9 {
				t.Fatalf("pol %s: unexpected energy %v in channel %d (tone should only occupy channel %d)", pol, mag, ch, channelIdx)
			}
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
	n := s.TickNSamples()
	result := s.GenerateNextTick(0, n)
	for _, v := range result["V"] {
		if real(v) != 0 || imag(v) != 0 {
			t.Fatalf("expected all-zero output for an out-of-range tone, got %v", v)
		}
	}
}

func TestDirectSynthesisStreamer_NoiseIsDeterministicAcrossRepeatedCalls(t *testing.T) {
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
	n := s.TickNSamples()

	a := s.GenerateNextTick(5.0, n)
	// GenerateNextTick reuses its output buffer, so copy before calling
	// again -- otherwise both "results" would alias the same backing
	// array and this test would trivially pass.
	aV := append([]complex128(nil), a["V"]...)

	b := s.GenerateNextTick(5.0, n)
	bV := b["V"]

	for i := range aV {
		if aV[i] != bV[i] {
			t.Fatalf("sample %d: repeated call at the same t gave different noise: %v vs %v", i, aV[i], bV[i])
		}
	}
}

func TestDirectSynthesisStreamer_NoiseIndependentAcrossStationSeeds(t *testing.T) {
	n := 8
	numChannels := 8
	buildBankOutput := func(seed int64) []complex128 {
		s, err := NewDirectSynthesisStreamer(StreamerConfig{
			Station:      testStation(),
			ObsTimeRef:   0,
			NumChannels:  numChannels,
			Noise:        &NoiseConfig{Std: 1.0, Seed: seed},
			NTiles:       4,
			TileNSamples: n,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		out := s.GenerateNextTick(0, s.TickNSamples())["V"]
		return append([]complex128(nil), out...)
	}

	stationA := buildBankOutput(1)
	stationB := buildBankOutput(2)

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
	// bug #1 in the Python CLAUDE.md: reusing one noise source for both
	// pols gave numerically identical "noise" for V and H. Guard against
	// regressing that here.
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
	result := s.GenerateNextTick(0, s.TickNSamples())
	v, h := result["V"], result["H"]
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
	// polynomial -- it never enters a delay pipeline (see the Python
	// CLAUDE.md's Noise section). This is trivially true in this Go port
	// (noise generation never reads any DelayFeed at all) but is worth a
	// regression test in case a future edit accidentally threads delay
	// into the noise path.
	feedA := common.NewDelayFeed("a")
	feedA.Update(&common.DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1e9, XYPolCoeffsNs: []float64{0.0}})
	feedB := common.NewDelayFeed("b")
	feedB.Update(&common.DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1e9, XYPolCoeffsNs: []float64{1e6}}) // large delay

	build := func(feed *common.DelayFeed) []complex128 {
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
		return append([]complex128(nil), s.GenerateNextTick(0, s.TickNSamples())["V"]...)
	}

	outA := build(feedA)
	outB := build(feedB)
	for i := range outA {
		if outA[i] != outB[i] {
			t.Fatalf("sample %d differs between a zero-delay and a large-delay source (%v vs %v) -- noise must be delay-independent", i, outA[i], outB[i])
		}
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
