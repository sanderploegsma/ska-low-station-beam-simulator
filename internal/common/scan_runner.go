package common

import (
	"log"
	"math"
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
	// output buffer to hand back. This avoids a second full copy the
	// accumulator would otherwise need to make to move samples out of a
	// caller-returned buffer.
	//
	// dst[pol][ch] is CHANNEL-MAJOR (one channel's samples contiguous,
	// not interleaved sample-major as a per-sample synthesis loop would
	// most naturally produce): this matches HeapAccumulator's own
	// per-channel storage order, and every downstream consumer
	// (per-channel heap encoding) wants one channel's samples contiguous.
	GenerateNextTick(t float64, n int, dst map[string][][]complex64)
}

// ComplexPathChannelIDMapper is an OPTIONAL Streamer capability. If a
// Streamer implements it, ScanRunner sizes its HeapAccumulator (and
// therefore GenerateNextTick's dst) to exactly this channel subset,
// instead of every channel in ChannelIDMap() -- for a Streamer that
// produces SOME channels' heaps a different way (see
// QuantizedHeapProducer) and only needs the complex64 dst-write path for
// the rest. A Streamer that doesn't implement this gets the original
// full-ChannelIDMap sizing, unchanged -- this is why it's a separate,
// optional interface rather than a new required Streamer method: every
// existing Streamer (including test fakes) keeps working with zero
// changes.
type ComplexPathChannelIDMapper interface {
	ComplexPathChannelIDMap() []int
}

// QuantizedHeapProducer is an OPTIONAL Streamer capability: a Streamer
// that can produce some channels' heaps WITHOUT the complex64 dst-write/
// HeapAccumulator path (e.g. pre-quantized noise-only channels, see
// synth.DirectSynthesisStreamer.GenerateQuantizedHeaps) implements this;
// ScanRunner calls it once per tick, alongside (not instead of)
// GenerateNextTick+PrepareWrite/PopReadyHeaps for whatever channels
// aren't covered this way (see ComplexPathChannelIDMapper). A Streamer
// that doesn't implement this is unaffected -- ScanRunner simply doesn't
// call it, and every channel goes through the original complex64 path.
type QuantizedHeapProducer interface {
	// GenerateQuantizedHeaps returns complete, ready-to-send heaps for
	// this tick (t: same absolute epoch time GenerateNextTick receives).
	// Each returned heap is exactly one HeapLen-sample heap -- there is
	// no cross-tick accumulation for this path, unlike HeapAccumulator's
	// general (if, in practice, always-one-tick-per-heap) buffering.
	GenerateQuantizedHeaps(t float64) []*ChannelHeap
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

	// driftBits/tick are updated once per tick from the run() goroutine
	// and read from GetStatus's goroutine (the gRPC handler) -- atomics,
	// not the mu-guarded fields elsewhere in this package, since these
	// are polled at a much higher rate than start/stop and don't need to
	// be consistent with any other field. float64 has no atomic type in
	// this Go version, hence the math.Float64bits/frombits round-trip
	// through an atomic.Uint64 rather than atomic.Value (which would box
	// every store).
	driftBits atomic.Uint64
	tick      atomic.Int64
}

// DriftSeconds returns the most recently observed pacing drift: wall-
// clock time minus that tick's target time, in seconds, at the last
// tick produced. Positive means the producer is running behind
// schedule. 0 before the first tick.
func (r *ScanRunner) DriftSeconds() float64 {
	return math.Float64frombits(r.driftBits.Load())
}

// TickNumber returns the most recently produced tick's 0-based index.
// 0 before the first tick.
func (r *ScanRunner) TickNumber() int64 {
	return r.tick.Load()
}

// NewScanRunner constructs a ScanRunner. The per-channel output sample
// rate is fixed by the ICD (ChannelOutputRateHz), not whichever backend
// is in use.
//
// The HeapAccumulator (and therefore GenerateNextTick's dst) is sized to
// streamer.ComplexPathChannelIDMap() if streamer implements
// ComplexPathChannelIDMapper, otherwise to the full streamer.
// ChannelIDMap() as before -- see that interface's doc comment.
func NewScanRunner(streamer Streamer, sender HeapSender, obsTime, scanDurationS float64) *ScanRunner {
	nSamplesPerTick := streamer.TickNSamples()
	complexChannelIDMap := streamer.ChannelIDMap()
	if m, ok := streamer.(ComplexPathChannelIDMapper); ok {
		complexChannelIDMap = m.ComplexPathChannelIDMap()
	}
	WarmBufferPools(streamer.NumChannels())
	return &ScanRunner{
		streamer:          streamer,
		sender:            sender,
		obsTime:           obsTime,
		scanDurationS:     scanDurationS,
		channelOutputRate: ChannelOutputRateHz,
		nSamplesPerTick:   nSamplesPerTick,
		accumulator: NewHeapAccumulator(
			len(complexChannelIDMap), obsTime, ChannelOutputRateHz, complexChannelIDMap,
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
			now = time.Now()
		}

		drift := now.Sub(targetWall).Seconds()
		r.driftBits.Store(math.Float64bits(drift))
		r.tick.Store(int64(tick))
		if drift > BlockDurationS*OverrunTolerance {
			log.Printf("producer falling behind pacing by %.3fs at tick %d", drift, tick)
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

		// Channels this streamer produces WITHOUT the complex64 dst-write/
		// HeapAccumulator path above (see QuantizedHeapProducer) -- most
		// Streamer implementations don't support this, in which case this
		// is simply a no-op and every channel came from PopReadyHeaps above.
		if qp, ok := r.streamer.(QuantizedHeapProducer); ok {
			for _, heap := range qp.GenerateQuantizedHeaps(simTime) {
				if !r.sender.Send(heap) {
					log.Printf("send queue full — dropping heap ch=%d t=%.4f", heap.ChannelID, heap.HeapStartTime)
				}
			}
		}
	}
	log.Printf("scan producer finished (stopped=false)")
}
