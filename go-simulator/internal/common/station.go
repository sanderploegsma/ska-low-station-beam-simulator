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
type ChannelHeap struct {
	ChannelID     int
	VSamples      []complex128 // len HeapLen
	HSamples      []complex128 // len HeapLen
	HeapStartTime float64      // sim_time (obs_time-relative seconds) of the first sample
}
