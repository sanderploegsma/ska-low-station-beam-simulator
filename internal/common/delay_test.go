package common

import "testing"

func TestDelayPolynomial_EvalDelaySeconds_VPol(t *testing.T) {
	p := &DelayPolynomial{
		StartValiditySec: 1000.0,
		XYPolCoeffsNs:    []float64{10.0, 2.0}, // 10 + 2*t_rel (ns)
		YPolOffsetNs:     5.0,
	}
	// t_rel = 1002 - 1000 = 2 -> tau_ns = 10 + 2*2 = 14
	got := p.EvalDelaySeconds(1002.0, "V")
	want := 14.0 * 1e-9
	if diff := got - want; diff > 1e-15 || diff < -1e-15 {
		t.Fatalf("EvalDelaySeconds(V) = %v, want %v", got, want)
	}
}

func TestDelayPolynomial_EvalDelaySeconds_HPolAddsOffset(t *testing.T) {
	p := &DelayPolynomial{
		StartValiditySec: 1000.0,
		XYPolCoeffsNs:    []float64{10.0},
		YPolOffsetNs:     5.0,
	}
	got := p.EvalDelaySeconds(1000.0, "H")
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
