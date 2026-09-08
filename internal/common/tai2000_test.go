package common

import "testing"

func TestUnixToTAI2000Seconds_IsAffineInInput(t *testing.T) {
	// The conversion is a fixed offset -- a 1-second step in unix_time
	// must be exactly a 1-second step in the result, regardless of the
	// (currently hardcoded) leap-second offset's actual value.
	base := 1_700_000_000.0
	a := UnixToTAI2000Seconds(base)
	b := UnixToTAI2000Seconds(base + 1.0)
	if diff := b - a; diff != 1.0 {
		t.Fatalf("a 1s step in unix_time produced a %vs step in TAI2000 seconds, want exactly 1s", diff)
	}
}

func TestUnixToTAI2000Seconds_PositiveForCurrentEraTimestamps(t *testing.T) {
	// Any unix time well after the TAI2000 epoch (2000-01-01) must yield
	// a positive result.
	got := UnixToTAI2000Seconds(1_700_000_000.0) // ~2023-11-14
	if got <= 0 {
		t.Fatalf("UnixToTAI2000Seconds(2023-ish) = %v, want > 0", got)
	}
}

func TestUnixToTAI2000Seconds_FitsIn40BitHeapCounterForCurrentEra(t *testing.T) {
	// An earlier version of this formula inflated heap_counter by
	// HeapLen (2048x, see docs/history.md), overflowing the ICD's 40-bit
	// field for any current-era timestamp. Guard the whole conversion + block-count
	// pipeline against silently regressing that: heap_counter for
	// "now-ish" must comfortably fit in 40 bits.
	const fortyBitMax = (int64(1) << 40) - 1
	heapCounter := int64(UnixToTAI2000Seconds(1_800_000_000.0) / BlockDurationS)
	if heapCounter < 0 || heapCounter > fortyBitMax {
		t.Fatalf("heap_counter = %d, does not fit in 40 bits (max %d)", heapCounter, fortyBitMax)
	}
}
