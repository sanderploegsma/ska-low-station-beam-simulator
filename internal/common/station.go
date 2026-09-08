package common

// StationConfig identifies a station in every heap it emits. subarray_id,
// beam_id and scan_id vary per scan; station_id/substation_id identify
// the pod itself.
type StationConfig struct {
	StationID    int32
	SubstationID int32
	SubarrayID   int32
	BeamID       int32
	ScanID       int64
}

// ChannelHeap is one channel's worth of generated samples, ready for
// SPEAD encoding.
//
// Samples are complex64, not complex128: the wire format quantizes every
// component down to int8 in the end, so complex128's ~15-16 decimal
// digits of precision is enormous overkill next to that ~2-digit
// (1/127) final resolution -- complex64's ~7 digits still leaves several
// orders of magnitude of headroom. Halving the sample width halves every
// byte the real-time pipeline has to move: the noise-bank tile copy
// (GenerateNextTick's dominant per-tick cost, profiled at ~45% of total
// CPU on target hardware -- see the go-simulator README's "Real-hardware
// profiling" section) and the SPEAD quantize passes that read this
// buffer straight afterward. Precision-sensitive computation (phase
// accumulation, delay-polynomial evaluation) still happens entirely in
// float64 -- see synth/tone.go -- only the FINAL sample value narrows,
// same principle as any other lossy-on-purpose step in this pipeline.
//
// VQuantized/HQuantized are an ALTERNATIVE to VSamples/HSamples, not an
// addition, per pol: a pol is either "complex path" (VSamples/HSamples
// set, quantized at SPEAD-encode time -- adaptively or via a fixed
// scale) or "pre-quantized path" (VQuantized/HQuantized set instead,
// bytes already fully quantized at noise-tile-bank CONSTRUCTION time).
// This exists because the noise-bank copy remained the single largest
// real-hardware cost even after halving to complex64 -- see the
// go-simulator README's "pre-quantized noise tile bank" section. A
// channel's path is decided once, for the whole scan (see
// synth.DirectSynthesisStreamer.ComplexPathChannelIDMap): a channel with
// no tone source targeting it never needs full-precision samples, since
// quantizing the SAME underlying noise draw at bank-construction time
// (once) instead of encode time (every tick) produces the identical
// wire bytes a fixed-scale complex-path encode would have produced.
type ChannelHeap struct {
	ChannelID     int
	VSamples      []complex64 // len HeapLen; nil if VQuantized is set instead
	HSamples      []complex64 // len HeapLen; nil if HQuantized is set instead
	VQuantized    []byte      // len HeapLen*2 (real,imag int8 pairs, interleaved); nil if VSamples is set instead
	HQuantized    []byte      // len HeapLen*2
	HeapStartTime float64     // sim_time (obs_time-relative seconds) of the first sample
}
