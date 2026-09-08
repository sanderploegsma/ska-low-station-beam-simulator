package spead

import "testing"

func TestDefaultNumSendersForChannels_KnownPoints(t *testing.T) {
	cases := []struct {
		numChannels int
		want        int
	}{
		{8, 1},    // MinNumChannels
		{96, 4},   // this project's historical "current target" channel count -- matches the old flat DefaultNumSenders exactly
		{192, 8},  // half band
		{384, 16}, // MaxNumChannels (full band) -- the confirmed real-hardware baseline itself
	}
	for _, c := range cases {
		if got := DefaultNumSendersForChannels(c.numChannels); got != c.want {
			t.Errorf("DefaultNumSendersForChannels(%d) = %d, want %d", c.numChannels, got, c.want)
		}
	}
}

func TestDefaultNumSendersForChannels_MonotonicAndPositive(t *testing.T) {
	prev := 0
	for numChannels := 8; numChannels <= 384; numChannels += 8 {
		got := DefaultNumSendersForChannels(numChannels)
		if got < 1 {
			t.Fatalf("DefaultNumSendersForChannels(%d) = %d, want >= 1", numChannels, got)
		}
		if got < prev {
			t.Fatalf("DefaultNumSendersForChannels(%d) = %d, less than DefaultNumSendersForChannels(%d) = %d -- expected non-decreasing", numChannels, got, numChannels-8, prev)
		}
		if got > fullBandNumSenders {
			t.Fatalf("DefaultNumSendersForChannels(%d) = %d, exceeds fullBandNumSenders (%d)", numChannels, got, fullBandNumSenders)
		}
		prev = got
	}
}
