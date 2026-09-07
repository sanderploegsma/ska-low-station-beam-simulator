package common

import (
	"sync"
	"testing"
	"time"
)

// fakeStreamer implements Streamer with a fixed, cheap-to-generate
// output, so ScanRunner tests exercise pacing/accumulation without
// depending on synth.DirectSynthesisStreamer.
type fakeStreamer struct {
	numChannels int
	tickN       int
	callCount   int
	mu          sync.Mutex
}

func (f *fakeStreamer) ChannelIDMap() []int {
	m := make([]int, f.numChannels)
	for i := range m {
		m[i] = i
	}
	return m
}
func (f *fakeStreamer) NumChannels() int  { return f.numChannels }
func (f *fakeStreamer) TickNSamples() int { return f.tickN }
func (f *fakeStreamer) GenerateNextTick(t float64, n int) map[string][]complex128 {
	f.mu.Lock()
	f.callCount++
	f.mu.Unlock()
	v := make([]complex128, n*f.numChannels)
	h := make([]complex128, n*f.numChannels)
	for i := range v {
		v[i] = complex(t, 0)
		h[i] = complex(t, 1)
	}
	return map[string][]complex128{"V": v, "H": h}
}

// fakeSender records every heap it receives, never drops (large enough
// buffer for these small tests).
type fakeSender struct {
	mu    sync.Mutex
	heaps []*ChannelHeap
}

func (f *fakeSender) Send(heap *ChannelHeap) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.heaps = append(f.heaps, heap)
	return true
}
func (f *fakeSender) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.heaps)
}

func TestScanRunner_EmitsExpectedHeapCount(t *testing.T) {
	numChannels := 2
	streamer := &fakeStreamer{numChannels: numChannels, tickN: HeapLen}
	sender := &fakeSender{}

	scanDuration := 3 * BlockDurationS // 3 ticks
	runner := NewScanRunner(streamer, sender, 0.0, scanDuration)
	runner.Start()
	select {
	case <-runner.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("scan did not finish within 2s")
	}

	wantHeaps := 3 * numChannels // one heap per channel per tick (tickN == HeapLen exactly)
	if got := sender.count(); got != wantHeaps {
		t.Fatalf("got %d heaps, want %d", got, wantHeaps)
	}
}

func TestScanRunner_StopEndsLoopEarly(t *testing.T) {
	streamer := &fakeStreamer{numChannels: 1, tickN: HeapLen}
	sender := &fakeSender{}

	// A long scan duration -- if Stop didn't work, this test would hang
	// for the full (many-second) duration.
	runner := NewScanRunner(streamer, sender, 0.0, 1000*BlockDurationS)
	runner.Start()
	time.Sleep(20 * time.Millisecond)
	runner.Stop(2 * time.Second)

	if runner.IsRunning() {
		t.Fatal("ScanRunner still reports running after Stop returned")
	}
}

func TestScanRunner_IsRunningReflectsLifecycle(t *testing.T) {
	streamer := &fakeStreamer{numChannels: 1, tickN: HeapLen}
	sender := &fakeSender{}
	runner := NewScanRunner(streamer, sender, 0.0, BlockDurationS)

	if runner.IsRunning() {
		t.Fatal("IsRunning() true before Start()")
	}
	runner.Start()
	runner.Stop(2 * time.Second)
	if runner.IsRunning() {
		t.Fatal("IsRunning() true after the scan finished and Stop returned")
	}
}
