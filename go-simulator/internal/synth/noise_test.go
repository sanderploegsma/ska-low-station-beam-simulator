package synth

import (
	"math"
	"testing"
)

func TestSplitmix64Hash_PureFunctionOfIndex(t *testing.T) {
	// Must be a pure function of (seed, index): calling it twice with the
	// same arguments must give byte-identical output -- this is what lets
	// GenerateNextTick be called twice with the same t and require
	// identical results (no hidden state to advance).
	a := splitmix64Hash(42, 7)
	b := splitmix64Hash(42, 7)
	if a != b {
		t.Fatalf("splitmix64Hash(42, 7) not repeatable: %d != %d", a, b)
	}
}

func TestSplitmix64Hash_DifferentIndicesDiffer(t *testing.T) {
	seen := map[uint64]bool{}
	for i := uint64(0); i < 1000; i++ {
		h := splitmix64Hash(1, i)
		if seen[h] {
			t.Fatalf("collision at index %d: hash %d already seen", i, h)
		}
		seen[h] = true
	}
}

func TestSplitmix64Hash_DifferentSeedsDiffer(t *testing.T) {
	// Two different (station, pol) seeds must not produce the same tile
	// sequence -- this is the property cross-station noise independence
	// relies on (see TestFillNoiseBank_CrossStationIndependence).
	same := 0
	const n = 1000
	for i := uint64(0); i < n; i++ {
		if splitmix64Hash(1, i) == splitmix64Hash(2, i) {
			same++
		}
	}
	if same > n/100 { // allow rare coincidental collisions, not systematic equality
		t.Fatalf("seeds 1 and 2 agree on %d/%d indices -- suspiciously correlated", same, n)
	}
}

func TestFillNoiseBank_Deterministic(t *testing.T) {
	bankA := fillNoiseBank(123, 1.0, 8, 16, 4)
	bankB := fillNoiseBank(123, 1.0, 8, 16, 4)
	if len(bankA) != len(bankB) {
		t.Fatalf("bank length mismatch: %d vs %d", len(bankA), len(bankB))
	}
	for i := range bankA {
		if bankA[i] != bankB[i] {
			t.Fatalf("bank mismatch at index %d: %v vs %v -- fillNoiseBank must be deterministic given the same seed", i, bankA[i], bankB[i])
		}
	}
}

func TestFillNoiseBank_CrossStationIndependence(t *testing.T) {
	// Different seeds (e.g. two different stations' noise seeds) must
	// produce statistically INDEPENDENT banks, never byte-identical ones
	// -- sharing content across stations would silently break any test
	// relying on receiver noise being uncorrelated between stations (see
	// the Python CLAUDE.md's Noise section).
	bankA := fillNoiseBank(1, 1.0, 8, 16, 4)
	bankB := fillNoiseBank(2, 1.0, 8, 16, 4)
	identical := 0
	for i := range bankA {
		if bankA[i] == bankB[i] {
			identical++
		}
	}
	if identical == len(bankA) {
		t.Fatalf("banks from different seeds are byte-identical -- noise is not station-independent")
	}
}

func TestFillNoiseBank_Statistics(t *testing.T) {
	// A large-enough bank's empirical std should land close to the
	// requested std, for both real and imaginary parts independently.
	const std = 2.5
	bank := fillNoiseBank(99, std, 64, 256, 4) // 64*256*4 = 65536 complex samples

	var sumSqReal, sumSqImag float64
	n := float64(len(bank))
	for _, c := range bank {
		re, im := float64(real(c)), float64(imag(c))
		sumSqReal += re * re
		sumSqImag += im * im
	}
	gotStdReal := math.Sqrt(sumSqReal / n)
	gotStdImag := math.Sqrt(sumSqImag / n)

	const tolerance = 0.05 * std // 5% -- generous for a finite-sample check
	if math.Abs(gotStdReal-std) > tolerance {
		t.Fatalf("real part empirical std = %v, want ~%v (tolerance %v)", gotStdReal, std, tolerance)
	}
	if math.Abs(gotStdImag-std) > tolerance {
		t.Fatalf("imag part empirical std = %v, want ~%v (tolerance %v)", gotStdImag, std, tolerance)
	}
}

func TestFillNoiseBank_TileBoundariesDisjoint(t *testing.T) {
	// Each tile should be independently drawn, not a repeated pattern --
	// a regression guard against a worker-partitioning bug that
	// accidentally reused the same slice for multiple tiles.
	const nTiles, tileNSamples, numChannels = 4, 8, 2
	bank := fillNoiseBank(7, 1.0, nTiles, tileNSamples, numChannels)
	tileLen := tileNSamples * numChannels
	for i := 0; i < nTiles; i++ {
		for j := i + 1; j < nTiles; j++ {
			same := true
			for k := 0; k < tileLen; k++ {
				if bank[i*tileLen+k] != bank[j*tileLen+k] {
					same = false
					break
				}
			}
			if same {
				t.Fatalf("tile %d and tile %d are identical -- expected independent draws", i, j)
			}
		}
	}
}
