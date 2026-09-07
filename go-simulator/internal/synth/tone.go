// Package synth is the numeric core — direct per-channel synthesis of
// tone and per-pol station (receiver) noise, ported from the Python
// project's direct_synthesis.py. Pulsar/pulsed sources are deliberately
// OUT OF SCOPE for this prototype (see the go-simulator README): the
// Python side already establishes tone/noise are feature-parity-critical
// for replacing CNIC, while pulsar support is a later addition.
package synth

import (
	"math"
	"math/cmplx"
)

// evalDelayPolyNs evaluates a delay polynomial (ascending-order
// coefficients, in nanoseconds) at tRel, which MUST already be relative
// to the polynomial's own start_validity_sec (small magnitude) — same
// precision-safety requirement as common.DelayPolynomial.EvalDelaySeconds.
func evalDelayPolyNs(coeffs []float64, tRel float64) float64 {
	tauNs := 0.0
	power := 1.0
	for _, c := range coeffs {
		tauNs += c * power
		power *= tRel
	}
	return tauNs
}

// synthToneChannel synthesizes nSamples of a tone in whichever channel
// freqHz maps to. Delay is applied as a CONTINUOUS PHASE TERM in the
// exponent — no ring buffer, no coarse/fine integer-sample split. Exact
// for a truly monochromatic tone (this is a closed-form solution, not an
// approximation), and O(1) per tone regardless of channel count.
//
// delayCoeffs, poly_t_rel_start and t_local_rel_start are all
// small-magnitude relative times — see the module-level note above (and
// the Python CLAUDE.md's extensive discussion of why raw epoch-scale time
// collapses float64 precision here).
func synthToneChannel(
	freqHz, amplitude, baseFreqHz, channelWidthHz float64,
	delayCoeffs []float64,
	polyTRelStart, ypolOffsetNs float64,
	isHPol bool,
	tLocalRelStart, sampleRatePerChannel float64,
	nSamples int,
) (channelIdx int, samples []complex128) {
	channelIdx = int(math.Round((freqHz - baseFreqHz) / channelWidthHz))
	channelCenter := baseFreqHz + float64(channelIdx)*channelWidthHz
	residualFreq := freqHz - channelCenter

	samples = make([]complex128, nSamples)
	for i := 0; i < nSamples; i++ {
		tLocal := tLocalRelStart + float64(i)/sampleRatePerChannel
		tPoly := polyTRelStart + float64(i)/sampleRatePerChannel
		tauNs := evalDelayPolyNs(delayCoeffs, tPoly)
		if isHPol {
			tauNs += ypolOffsetNs
		}
		tauS := tauNs * 1e-9
		phase := 2.0*math.Pi*residualFreq*tLocal - 2.0*math.Pi*freqHz*tauS
		samples[i] = cmplx.Rect(amplitude, phase)
	}
	return channelIdx, samples
}
