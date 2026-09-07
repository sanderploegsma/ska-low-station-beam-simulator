package common

import (
	"log"
	"math"
	"sync"
	"sync/atomic"
)

// DelayPolynomial is a per-source delay model, per the
// ska-low-csp-delaymodel/1.0 schema (ADR-88 in ska-telmodel).
type DelayPolynomial struct {
	StationID         int32
	StartValiditySec  float64
	ValidityPeriodSec float64
	XYPolCoeffsNs     []float64
	YPolOffsetNs      float64
}

// ValidUntil is the absolute epoch time this polynomial stops being
// current.
func (p *DelayPolynomial) ValidUntil() float64 {
	return p.StartValiditySec + p.ValidityPeriodSec
}

// EvalDelaySeconds evaluates this polynomial at absolute epoch time t,
// for polarisation "V" or "H" ("H" adds YPolOffsetNs). IMPORTANT:
// evaluated relative to StartValiditySec, NOT raw absolute epoch time — a
// high-order polynomial loses float64 precision otherwise (the same
// precision-collapse bug documented throughout the Python codebase's
// CLAUDE.md; every kernel in this port takes a small-magnitude relative
// time for the same reason).
func (p *DelayPolynomial) EvalDelaySeconds(t float64, pol string) float64 {
	tRel := t - p.StartValiditySec
	tauXNs := 0.0
	power := 1.0
	for _, c := range p.XYPolCoeffsNs {
		tauXNs += c * power
		power *= tRel
	}
	tauNs := tauXNs
	if pol == "H" {
		tauNs += p.YPolOffsetNs
	}
	return tauNs * 1e-9
}

var zeroDelayCoeffsNs = []float64{0.0}

// DelayFeed answers "what delay polynomial applies at time t" for one
// source. Update is called from whatever goroutine learns of a new
// polynomial (a gRPC PushDelayUpdate handler in this prototype, direct
// calls in tests); Get is called from the generation goroutine. Unlike
// the Python original — where a plain reference swap is safe under the
// GIL — Go has no GIL, so the polynomial pointer is swapped via
// atomic.Pointer and the "warned once" bookkeeping is guarded by a mutex.
//
// Two deliberate behaviours, ported from Python's DelayFeed, not
// oversights:
//   - No polynomial received yet -> zero delay, warned ONCE (not every
//     tick) — a reasonable default for "hasn't started publishing yet"
//     rather than blocking scan start on an external device being up.
//   - Polynomial expired (t >= ValidUntil()) with no replacement arrived
//     -> keep applying it as-is, warned once per staleness episode.
//     Recovering from a stalled upstream publisher is explicitly NOT
//     this simulator's job.
type DelayFeed struct {
	name string
	poly atomic.Pointer[DelayPolynomial]

	mu                     sync.Mutex
	warnedNoPoly           bool
	warnedStaleValidUntil  float64
	hasWarnedStaleValidity bool
}

// NewDelayFeed creates a DelayFeed with no polynomial yet — Get will
// apply zero delay (and warn once) until Update is called.
func NewDelayFeed(name string) *DelayFeed {
	return &DelayFeed{name: name}
}

// Update installs a freshly-received polynomial, clearing any prior
// staleness warning state (a fresh polynomial is not stale).
func (f *DelayFeed) Update(poly *DelayPolynomial) {
	f.poly.Store(poly)
	f.mu.Lock()
	f.hasWarnedStaleValidity = false
	f.mu.Unlock()
}

// Get returns the polynomial currently in effect at time t.
func (f *DelayFeed) Get(t float64) *DelayPolynomial {
	p := f.poly.Load()
	if p == nil {
		f.mu.Lock()
		if !f.warnedNoPoly {
			log.Printf("delay source %q has not received a polynomial yet — applying zero delay until one arrives", f.name)
			f.warnedNoPoly = true
		}
		f.mu.Unlock()
		return &DelayPolynomial{
			StationID:         -1,
			StartValiditySec:  t,
			ValidityPeriodSec: math.Inf(1),
			XYPolCoeffsNs:     zeroDelayCoeffsNs,
			YPolOffsetNs:      0.0,
		}
	}
	if validUntil := p.ValidUntil(); t >= validUntil {
		f.mu.Lock()
		if !f.hasWarnedStaleValidity || f.warnedStaleValidUntil != validUntil {
			log.Printf("delay source %q polynomial expired at t=%.3f (valid_until=%.3f) with no replacement received yet — continuing to apply the expired coefficients", f.name, t, validUntil)
			f.warnedStaleValidUntil = validUntil
			f.hasWarnedStaleValidity = true
		}
		f.mu.Unlock()
	}
	return p
}
