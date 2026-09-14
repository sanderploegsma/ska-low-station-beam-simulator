package common

import "testing"

// unixNowForTest is an arbitrary Unix-epoch "current time" used to build
// realistic wire-format polynomials in these tests: StartValiditySec is
// TAI2000-relative, so it's derived via UnixToTAI2000Seconds rather than
// reused directly as a Unix timestamp (see EvalDelaySeconds's doc comment).
const unixNowForTest = 1_700_000_000.0

func TestDelayPolynomial_EvalDelaySeconds_VPol(t *testing.T) {
	p := &DelayPolynomial{
		StartValiditySec: UnixToTAI2000Seconds(unixNowForTest),
		XYPolCoeffsNs:    []float64{10.0, 2.0}, // 10 + 2*t_rel (ns)
		YPolOffsetNs:     5.0,
	}
	// t_rel = 2s after StartValiditySec -> tau_ns = 10 + 2*2 = 14
	got := p.EvalDelaySeconds(unixNowForTest+2.0, "V")
	want := 14.0 * 1e-9
	if diff := got - want; diff > 1e-15 || diff < -1e-15 {
		t.Fatalf("EvalDelaySeconds(V) = %v, want %v", got, want)
	}
}

func TestDelayPolynomial_EvalDelaySeconds_HPolAddsOffset(t *testing.T) {
	p := &DelayPolynomial{
		StartValiditySec: UnixToTAI2000Seconds(unixNowForTest),
		XYPolCoeffsNs:    []float64{10.0},
		YPolOffsetNs:     5.0,
	}
	got := p.EvalDelaySeconds(unixNowForTest, "H")
	want := 15.0 * 1e-9 // 10 (poly) + 5 (ypol offset)
	if diff := got - want; diff > 1e-15 || diff < -1e-15 {
		t.Fatalf("EvalDelaySeconds(H) = %v, want %v", got, want)
	}
}

func TestDelayFeed_UpdateThenGetReturnsThatPolynomial(t *testing.T) {
	feed := NewDelayFeed("src")
	poly := &DelayPolynomial{StationID: 3, StartValiditySec: 0, ValidityPeriodSec: 100, XYPolCoeffsNs: []float64{1.0}}
	feed.Update(poly)
	got := feed.Get(50.0)
	if got != poly {
		t.Fatalf("Get returned a different polynomial than the one Updated")
	}
}

func TestDelayFeed_ExpiredWarningDoesNotChangeReturnedPolynomial(t *testing.T) {
	feed := NewDelayFeed("src")
	poly := &DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1.0, XYPolCoeffsNs: []float64{7.0}}
	feed.Update(poly)
	// Call Get multiple times past expiry -- must keep returning the same
	// stale polynomial (not substitute zero delay), matching the "surface
	// it, don't paper over it" behaviour documented on DelayFeed.
	for _, t2 := range []float64{5.0, 6.0, 100.0} {
		got := feed.Get(t2)
		if got != poly {
			t.Fatalf("Get(%v) after expiry returned a different polynomial, want the same stale one", t2)
		}
	}
}

func TestDelayFeed_GetComparesExpiryAgainstTAI2000NotUnixEpoch(t *testing.T) {
	feed := NewDelayFeed("src")
	// A realistic wire-format polynomial: StartValiditySec/ValidityPeriodSec
	// are TAI2000-relative (per ska-low-csp-delaymodel/1.0), so they sit
	// ~9.4e8 below a same-instant Unix timestamp. Get is called with the
	// Unix-epoch t the rest of this codebase uses everywhere else -- if Get
	// compared it to ValidUntil() without converting, this poly would look
	// expired (and warn) from the instant it's installed, ~9.4e8s "early".
	const unixNow = 1_700_000_000.0
	startValidityTAI2000 := UnixToTAI2000Seconds(unixNow)
	poly := &DelayPolynomial{StartValiditySec: startValidityTAI2000, ValidityPeriodSec: 100.0, XYPolCoeffsNs: []float64{1.0}}
	feed.Update(poly)

	feed.Get(unixNow + 50.0) // still well within the 100s validity window
	if feed.hasWarnedStaleValidity {
		t.Fatalf("Get treated a not-yet-expired real (TAI2000-relative) polynomial as expired — expiry check is comparing mismatched epochs")
	}

	feed.Get(unixNow + 150.0) // now genuinely past the 100s validity window
	if !feed.hasWarnedStaleValidity {
		t.Fatalf("Get did not flag a genuinely expired polynomial as stale")
	}
}

func TestDelayFeed_FreshUpdateClearsStaleness(t *testing.T) {
	feed := NewDelayFeed("src")
	feed.Update(&DelayPolynomial{StartValiditySec: 0, ValidityPeriodSec: 1.0, XYPolCoeffsNs: []float64{1.0}})
	feed.Get(10.0) // now stale, warned

	fresh := &DelayPolynomial{StartValiditySec: 10.0, ValidityPeriodSec: 100.0, XYPolCoeffsNs: []float64{2.0}}
	feed.Update(fresh)
	got := feed.Get(20.0)
	if got != fresh {
		t.Fatalf("expected the fresh polynomial after Update, got a different one")
	}
}
