// Package common holds shared, backend-agnostic plumbing: ICD-fixed
// constants, delay polynomial handling, station/heap data structures, and
// the producer (ScanRunner) that drives a Streamer — ported from this
// project's Python common.py. See the Python CLAUDE.md's "SPS-CBF ICD
// channelization" section for how these constants were confirmed against
// the real ICD text (not a screenshot, not an earlier guess).
package common

// CHANNEL_WIDTH_HZ: per SPS-CBF ICD coarse channel spacing (channel
// CENTRE spacing, unaffected by the filterbank's oversampling — see
// ChannelOutputRateHz below for the distinct TIME-domain sample rate).
const ChannelWidthHz = 781_250.0

// ChannelStart: the lowest frequency channel is global ID 64, centre
// frequency 50MHz exactly (64 * ChannelWidthHz). Confirmed against the
// real ICD text; a previous assumption of channel 65 was off by one.
const ChannelStart = 64

// Channel count bounds: the band is channelized as 384 equispaced coarse
// channels, configurable from 8 to 384 in steps of 8. 448 (350MHz) was
// used in earlier work as "the full band" but was never a valid SPS
// configuration.
const (
	MinNumChannels  = 8
	MaxNumChannels  = 384
	NumChannelsStep = 8
)

// BaseFreqHz: channel 0's (i.e. global channel ChannelStart's) centre
// frequency. Every per-channel frequency in this codebase is expressed as
// a channel CENTRE, not a band edge.
const BaseFreqHz = 50.0e6

// HeapLen: time samples per heap, per channel, per ICD.
const HeapLen = 2048

// The SPS filterbank oversamples by 32/27 — each channel's real
// TIME-domain sample period is 1080ns (1.25ns ADC period * 1024 * 27/32),
// not the naive 1280ns a critically-sampled channelizer would give.
// ChannelWidthHz remains correct as-is for FREQUENCY-domain channel
// spacing; ChannelOutputRateHz is the distinct time-domain sample rate.
const (
	OversamplingNumerator   = 32
	OversamplingDenominator = 27
)

const ChannelOutputRateHz = ChannelWidthHz * OversamplingNumerator / OversamplingDenominator

// BlockDurationS: the per-tick real-time budget, FIXED regardless of
// channel count — HeapLen samples at the real (oversampled)
// per-channel rate. Also confirms HeapLen satisfies the ICD's separate
// requirement that samples-per-packet be a multiple of the oversampling
// numerator (32): 2048/32 = 64 exactly.
const BlockDurationS = HeapLen / ChannelOutputRateHz

const (
	OverrunTolerance  = 2.0
	QueueMaxSize      = 4096
	QueuePutTimeoutMS = 500
)

// TAI2000 epoch: 2000-01-01T00:00:00 TAI — the SKA epoch for heap_counter.
const TAI2000EpochISO = "2000-01-01T00:00:00"
