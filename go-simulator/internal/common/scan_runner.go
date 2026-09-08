package common

import (
	"log"
	"sync/atomic"
	"time"
)

// Streamer is the small, backend-agnostic surface ScanRunner drives —
// ported from Python's common.Streamer Protocol. This package never
// imports synth.DirectSynthesisStreamer (the sole implementation) so it
// stays testable independent of the generation strategy behind it.
type Streamer interface {
	ChannelIDMap() []int
	NumChannels() int

	// TickNSamples: how many per-channel output samples
	// GenerateNextTick's n should be for one tick — sized so one tick
	// produces close to exactly one heap's worth of per-channel samples.
	TickNSamples() int

	// GenerateNextTick writes this tick's samples directly into dst: for
	// each pol ("V"/"H") present as a key, dst[pol] has NumChannels()
	// entries, each already sized to exactly n samples (allocated by
	// HeapAccumulator.PrepareWrite) — implementations write generated
	// samples straight into dst[pol][ch], never build their own separate
	// output buffer to hand back.
	//
	// This replaced an earlier "return a fresh/reused buffer, caller
	// copies it into the accumulator" design: even after that buffer was
	// made channel-major (see below) and reused across ticks (removing
	// per-tick allocation), the accumulator's own Add still had to copy
	// every sample out of it a second time. Profiling a real end-to-end
	// run (384 channels, EPYC target hardware) found that second copy
	// was the dominant remaining cost even after PARALLELIZING it across
	// every available core — more threads don't help a fixed amount of
	// memory traffic that a single copy doesn't need to move at all. See
	// HeapAccumulator.PrepareWrite's doc comment for the full profiling
	// trail.
	//
	// dst[pol][ch] being CHANNEL-MAJOR (one channel's samples contiguous,
	// not interleaved sample-major as a per-sample synthesis loop would
	// most naturally produce) is unchanged from before: it's also
	// HeapAccumulator's own per-channel storage order, and every
	// downstream consumer (per-channel heap encoding) wants one
	// channel's samples contiguous — see HeapAccumulator's doc comment
	// for the transpose-elimination history this layout choice predates.
	GenerateNextTick(t float64, n int, dst map[string][][]complex64)
}

// HeapSender is anything that can send a ChannelHeap onward (SPEAD/UDP in
// production, a channel-backed queue here). Kept as an interface, not a
// concrete queue type, so ScanRunner doesn't dictate how heaps reach the
// sender.
type HeapSender interface {
	// Send enqueues heap for sending, returning false if it was dropped
	// (queue full) rather than blocking indefinitely.
	Send(heap *ChannelHeap) bool
}

// ScanRunner drives a Streamer on a real-time pacing loop, matching
// Python's ScanRunner: one goroutine ticks at BlockDurationS intervals,
// feeding each tick's output into a HeapAccumulator and forwarding
// popped heaps to a HeapSender.
type ScanRunner struct {
	streamer      Streamer
	sender        HeapSender
	obsTime       float64
	scanDurationS float64

	channelOutputRate float64
	nSamplesPerTick   int
	accumulator       *HeapAccumulator

	started atomic.Bool
	stop    chan struct{}
	done    chan struct{}
}

// NewScanRunner constructs a ScanRunner. The per-channel output sample
// rate is fixed by the ICD (ChannelOutputRateHz), not whichever backend
// is in use.
func NewScanRunner(streamer Streamer, sender HeapSender, obsTime, scanDurationS float64) *ScanRunner {
	nSamplesPerTick := streamer.TickNSamples()
	return &ScanRunner{
		streamer:          streamer,
		sender:            sender,
		obsTime:           obsTime,
		scanDurationS:     scanDurationS,
		channelOutputRate: ChannelOutputRateHz,
		nSamplesPerTick:   nSamplesPerTick,
		accumulator: NewHeapAccumulator(
			streamer.NumChannels(), obsTime, ChannelOutputRateHz, streamer.ChannelIDMap(),
		),
		stop: make(chan struct{}),
		done: make(chan struct{}),
	}
}

// Start runs the scan loop in a new goroutine.
func (r *ScanRunner) Start() {
	r.started.Store(true)
	go r.run()
}

// Done returns a channel that's closed once the scan loop has finished
// (naturally or via Stop) — lets callers (tests, StopScan handlers) wait
// for completion without polling IsRunning.
func (r *ScanRunner) Done() <-chan struct{} {
	return r.done
}

// IsRunning reports whether the scan loop has been started and hasn't
// finished yet.
func (r *ScanRunner) IsRunning() bool {
	if !r.started.Load() {
		return false
	}
	select {
	case <-r.done:
		return false
	default:
		return true
	}
}

// Stop signals the scan loop to stop and waits (up to timeout) for it to
// finish.
func (r *ScanRunner) Stop(timeout time.Duration) {
	select {
	case <-r.stop:
		// already stopped
	default:
		close(r.stop)
	}
	select {
	case <-r.done:
	case <-time.After(timeout):
	}
}

func (r *ScanRunner) run() {
	defer close(r.done)

	blockDuration := time.Duration(BlockDurationS * float64(time.Second))
	wallStart := time.Now()
	nTicks := int(r.scanDurationS / BlockDurationS)

	for tick := 0; tick < nTicks; tick++ {
		select {
		case <-r.stop:
			log.Printf("scan producer finished (stopped=true)")
			return
		default:
		}

		simTime := r.obsTime + float64(tick)*BlockDurationS
		targetWall := wallStart.Add(time.Duration(tick) * blockDuration)
		now := time.Now()
		if now.Before(targetWall) {
			timer := time.NewTimer(targetWall.Sub(now))
			select {
			case <-r.stop:
				timer.Stop()
				log.Printf("scan producer finished (stopped=true)")
				return
			case <-timer.C:
			}
		} else {
			overrun := now.Sub(targetWall).Seconds()
			if overrun > BlockDurationS*OverrunTolerance {
				log.Printf("producer falling behind pacing by %.3fs at tick %d", overrun, tick)
			}
		}

		dst := map[string][][]complex64{
			"V": r.accumulator.PrepareWrite("V", r.nSamplesPerTick),
			"H": r.accumulator.PrepareWrite("H", r.nSamplesPerTick),
		}
		r.streamer.GenerateNextTick(simTime, r.nSamplesPerTick, dst)

		for _, heap := range r.accumulator.PopReadyHeaps() {
			if !r.sender.Send(heap) {
				log.Printf("send queue full — dropping heap ch=%d t=%.4f", heap.ChannelID, heap.HeapStartTime)
			}
		}
	}
	log.Printf("scan producer finished (stopped=false)")
}
