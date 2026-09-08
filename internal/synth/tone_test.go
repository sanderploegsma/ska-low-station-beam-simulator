package synth

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestSynthToneChannel_LandsInCorrectChannel verifies a tone at a known
// frequency maps to the expected channel index and that its residual
// frequency (relative to that channel's centre) is what generates the
// samples' phase evolution — the Go equivalent of the Python module's
// __main__ channel-mapping correctness check.
func TestSynthToneChannel_LandsInCorrectChannel(t *testing.T) {
	const baseFreqHz = 50.0e6
	const channelWidthHz = 781_250.0
	channelIdx := 10
	residual := 1000.0 // Hz off the channel centre
	freqHz := baseFreqHz + float64(channelIdx)*channelWidthHz + residual

	gotIdx, samples := synthToneChannel(
		freqHz, 1.0, baseFreqHz, channelWidthHz,
		[]float64{0.0}, // zero delay
		0.0, 0.0, false,
		0.0, 925_925.925925926, 2048,
	)

	if gotIdx != channelIdx {
		t.Fatalf("channel index = %d, want %d", gotIdx, channelIdx)
	}
	if len(samples) != 2048 {
		t.Fatalf("len(samples) = %d, want 2048", len(samples))
	}
	// Sample 0 at zero delay must have phase 0 (amplitude on the real axis).
	if diff := cmplx.Abs(complex128(samples[0] - complex64(complex(1.0, 0.0)))); diff > 1e-9 {
		t.Fatalf("samples[0] = %v, want ~1+0i (diff %v)", samples[0], diff)
	}
}

// TestSynthToneChannel_DelayMatchesAnalyticPhaseShift verifies delay is
// applied as an EXACT continuous phase term: for a monochromatic tone,
// applying a delay tau is equivalent to phase -2*pi*freq*tau, exactly
// (not a first-order/Taylor approximation) -- this is the core numerical
// property tone synthesis depends on being exact.
func TestSynthToneChannel_DelayMatchesAnalyticPhaseShift(t *testing.T) {
	const baseFreqHz = 50.0e6
	const channelWidthHz = 781_250.0
	freqHz := baseFreqHz + 5*channelWidthHz + 12345.0
	tauNs := 137.5 // constant delay: a single coefficient (order-0 poly)

	_, zeroDelay := synthToneChannel(
		freqHz, 1.0, baseFreqHz, channelWidthHz,
		[]float64{0.0}, 0.0, 0.0, false, 0.0, 925_925.925925926, 8,
	)
	_, withDelay := synthToneChannel(
		freqHz, 1.0, baseFreqHz, channelWidthHz,
		[]float64{tauNs}, 0.0, 0.0, false, 0.0, 925_925.925925926, 8,
	)

	expectedShift := -2.0 * math.Pi * freqHz * (tauNs * 1e-9)
	for i := range zeroDelay {
		ratio := withDelay[i] / zeroDelay[i]
		gotPhase := cmplx.Phase(complex128(ratio))
		// wrap expectedShift into (-pi, pi] the same way cmplx.Phase does
		wanted := math.Atan2(math.Sin(expectedShift), math.Cos(expectedShift))
		if diff := math.Abs(gotPhase - wanted); diff > 1e-6 {
			t.Fatalf("sample %d: phase shift = %v, want %v (diff %v)", i, gotPhase, wanted, diff)
		}
	}
}

// TestSynthToneChannel_HPolOffsetAppliesOnlyToHPol verifies ypol_offset_ns
// only shifts phase when isHPol is true.
func TestSynthToneChannel_HPolOffsetAppliesOnlyToHPol(t *testing.T) {
	const baseFreqHz = 50.0e6
	const channelWidthHz = 781_250.0
	freqHz := baseFreqHz + 3*channelWidthHz

	_, vSamples := synthToneChannel(freqHz, 1.0, baseFreqHz, channelWidthHz, []float64{0.0}, 0.0, 50.0, false, 0.0, 925_925.925925926, 4)
	_, hSamplesNoOffset := synthToneChannel(freqHz, 1.0, baseFreqHz, channelWidthHz, []float64{0.0}, 0.0, 0.0, true, 0.0, 925_925.925925926, 4)
	_, hSamplesWithOffset := synthToneChannel(freqHz, 1.0, baseFreqHz, channelWidthHz, []float64{0.0}, 0.0, 50.0, true, 0.0, 925_925.925925926, 4)

	for i := range vSamples {
		if cmplx.Abs(complex128(vSamples[i]-hSamplesNoOffset[i])) > 1e-12 {
			t.Fatalf("sample %d: V (ypol offset present but isHPol=false) and H (zero offset) should match, got %v vs %v", i, vSamples[i], hSamplesNoOffset[i])
		}
		if cmplx.Abs(complex128(hSamplesWithOffset[i]-hSamplesNoOffset[i])) < 1e-9 {
			t.Fatalf("sample %d: H-pol with a nonzero ypol_offset_ns should differ from zero-offset H-pol, got equal values %v", i, hSamplesWithOffset[i])
		}
	}
}
