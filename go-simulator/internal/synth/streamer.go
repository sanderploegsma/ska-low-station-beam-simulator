package synth

import (
	"fmt"
	"log"
	"runtime"
	"sync"

	"github.com/skao/station-beam-simulator-go/internal/common"
)

// DefaultNTiles matches the Python default (DEFAULT_N_TILES) — see the
// Python CLAUDE.md's Noise section for the fidelity/resource tradeoff
// this controls (birthday-paradox tile repeat vs. build time/memory).
const DefaultNTiles = 256

// ToneSourceConfig is a tone source — required DelayFeed, no
// default/fallback delay (see the Python CLAUDE.md's "Per-source delay"
// section for why: a source with no real delay path would silently
// produce trivially "perfectly aligned" content, exactly the kind of
// thing that could mask a real CBF delay-tracking bug).
type ToneSourceConfig struct {
	DelayFeed *common.DelayFeed
	FreqHz    float64
	Amplitude float64 // 0 is treated as "use 1.0" by NewDirectSynthesisStreamer
}

// NoiseConfig configures the per-pol station (receiver) noise tile bank.
type NoiseConfig struct {
	Std  float64
	Seed int64
}

// StreamerConfig is the full construction-time configuration for a
// DirectSynthesisStreamer. Zero-valued optional fields fall back to the
// same defaults as the Python DirectSynthesisStreamer's constructor
// kwargs.
type StreamerConfig struct {
	Station     *common.StationConfig
	ToneSources []ToneSourceConfig
	ObsTimeRef  float64
	Noise       *NoiseConfig // nil for no noise at all

	NumChannels    int     // 0 -> common.MaxNumChannels
	BaseFreqHz     float64 // 0 -> common.BaseFreqHz
	ChannelWidthHz float64 // 0 -> common.ChannelWidthHz
	NTiles         int     // 0 -> DefaultNTiles
	TileNSamples   int     // 0 -> TickNSamples()

	// NumWorkers: goroutines GenerateNextTick splits its per-tick noise
	// fill across (see that method's doc comment). 0 -> a
	// GOMAXPROCS-based default (defaultParallelism), which automatically
	// respects a Kubernetes pod's CPU request/limit, unlike a hardcoded
	// constant would.
	NumWorkers int
}

// DirectSynthesisStreamer is the numeric core: direct, per-channel
// synthesis of tone and per-pol station noise (no pulsar — see this
// package's doc comment). Implements common.Streamer.
type DirectSynthesisStreamer struct {
	station           *common.StationConfig
	numChannels       int
	baseFreqHz        float64
	channelWidthHz    float64
	channelOutputRate float64
	obsTimeRef        float64

	toneCfgs []ToneSourceConfig

	noiseCfg   *NoiseConfig
	noiseSeedV uint64
	noiseSeedH uint64
	noiseStd   float64

	nTiles       int
	tileNSamples int
	banks        map[string][]complex128 // "V"/"H" -> flat (nTiles, tileNSamples, numChannels)

	numWorkers int
}

// NewDirectSynthesisStreamer validates cfg and builds a streamer,
// including filling the noise tile bank (if configured) — this is the
// same one-time, non-per-tick construction cost the Python version pays
// (see the Python CLAUDE.md's "One-time construction budget" section).
func NewDirectSynthesisStreamer(cfg StreamerConfig) (*DirectSynthesisStreamer, error) {
	numChannels := cfg.NumChannels
	if numChannels == 0 {
		numChannels = common.MaxNumChannels
	}
	if numChannels < common.MinNumChannels || numChannels > common.MaxNumChannels || numChannels%common.NumChannelsStep != 0 {
		return nil, fmt.Errorf(
			"num_channels=%d is not a valid SPS beam configuration -- per the ICD, "+
				"the number of channels assigned to a beam is configurable from %d to %d "+
				"in steps of %d (%d channels * %.0fHz = 300MHz is the real maximum)",
			numChannels, common.MinNumChannels, common.MaxNumChannels, common.NumChannelsStep,
			common.MaxNumChannels, common.ChannelWidthHz,
		)
	}

	baseFreqHz := cfg.BaseFreqHz
	if baseFreqHz == 0 {
		baseFreqHz = common.BaseFreqHz
	}
	channelWidthHz := cfg.ChannelWidthHz
	if channelWidthHz == 0 {
		channelWidthHz = common.ChannelWidthHz
	}
	channelOutputRate := channelWidthHz * common.OversamplingNumerator / common.OversamplingDenominator

	for i, ts := range cfg.ToneSources {
		if ts.DelayFeed == nil {
			return nil, fmt.Errorf("tone source %d (freq_hz=%v) has no delay_feed -- every source must have its own delay path, there is no default/fallback delay", i, ts.FreqHz)
		}
		if ts.Amplitude == 0 {
			cfg.ToneSources[i].Amplitude = 1.0
		}
	}

	s := &DirectSynthesisStreamer{
		station:           cfg.Station,
		numChannels:       numChannels,
		baseFreqHz:        baseFreqHz,
		channelWidthHz:    channelWidthHz,
		channelOutputRate: channelOutputRate,
		obsTimeRef:        cfg.ObsTimeRef,
		toneCfgs:          cfg.ToneSources,
		noiseCfg:          cfg.Noise,
		banks:             make(map[string][]complex128),
	}

	if cfg.Noise != nil {
		s.noiseSeedV = uint64(cfg.Noise.Seed)
		s.noiseSeedH = uint64(cfg.Noise.Seed + 1_000_003)
		s.noiseStd = cfg.Noise.Std
	}

	s.nTiles = cfg.NTiles
	if s.nTiles == 0 {
		s.nTiles = DefaultNTiles
	}
	s.tileNSamples = cfg.TileNSamples
	if s.tileNSamples == 0 {
		s.tileNSamples = s.TickNSamples()
	}

	if cfg.Noise != nil {
		s.banks["V"] = fillNoiseBank(s.noiseSeedV, s.noiseStd, s.nTiles, s.tileNSamples, s.numChannels)
		s.banks["H"] = fillNoiseBank(s.noiseSeedH, s.noiseStd, s.nTiles, s.tileNSamples, s.numChannels)
	}

	s.numWorkers = cfg.NumWorkers
	if s.numWorkers == 0 {
		s.numWorkers = defaultParallelism(numChannels)
	}

	return s, nil
}

// defaultParallelism picks a worker/goroutine count consistently across
// this package: runtime.GOMAXPROCS(0) (which — unlike runtime.NumCPU()
// — respects a Kubernetes pod's CPU request/limit), capped only to n (no
// point spawning more workers than units of independent work).
//
// This used to also cap at a flat 16, matching fillNoiseBank's identical
// cap for its one-time noise-bank *construction* cost -- wrong to share
// here: GenerateNextTick's noise fill runs on EVERY tick under the fixed
// per-tick budget, not once at startup, and the Python CLAUDE.md's own
// EPYC benchmarking already established that this class of
// per-channel-independent work keeps scaling well past 16 threads once
// allocation overhead is out of the way (see its "Target server results"
// section: throughput kept improving monotonically up to 96 threads). A
// real profile on 2-socket EPYC target hardware showed average
// concurrency pinned at ~15.18 -- suspiciously exactly this cap -- while
// the machine had far more cores sitting idle and pacing was still
// falling behind. Removed; GOMAXPROCS is now trusted on its own.
func defaultParallelism(n int) int {
	w := runtime.GOMAXPROCS(0)
	if w > n {
		w = n
	}
	if w < 1 {
		w = 1
	}
	return w
}

// ChannelIDMap implements common.Streamer — identity, since this
// streamer's output columns are always already in external
// ascending-frequency channel_id order.
func (s *DirectSynthesisStreamer) ChannelIDMap() []int {
	m := make([]int, s.numChannels)
	for i := range m {
		m[i] = i
	}
	return m
}

// NumChannels implements common.Streamer.
func (s *DirectSynthesisStreamer) NumChannels() int { return s.numChannels }

// TickNSamples implements common.Streamer — HeapLen by construction
// (BlockDurationS is defined for exactly this).
func (s *DirectSynthesisStreamer) TickNSamples() int {
	return int(s.channelOutputRate*common.BlockDurationS + 0.5)
}

// BankMemoryBytes reports total noise-bank memory across both pols (0 if
// noise is not configured).
func (s *DirectSynthesisStreamer) BankMemoryBytes() int64 {
	if len(s.banks) == 0 {
		return 0
	}
	return bankMemoryBytes(s.nTiles, s.tileNSamples, s.numChannels, len(s.banks))
}

// GenerateNextTick implements common.Streamer. t is the absolute epoch
// time of this tick's first sample; nSamples is the per-channel sample
// count for this tick (at channelOutputRate). Writes directly into
// dst[pol][ch] for each configured pol — see common.Streamer's doc
// comment for why (this replaced an earlier "return a buffer, caller
// copies it into the accumulator" design once profiling on real target
// hardware found that copy dominating CPU time even after being
// parallelized across every available core).
//
// Since noise fill dominates this method's own cost (a bulk copy from
// the tile bank, O(numChannels)) and every channel's copy is
// independent, it's parallelized here across both pols AND channel
// ranges at once -- V and H are already fully independent (separate
// seeds/buffers, separate dst entries), so no worker ever touches
// another's data. Tone injection is NOT parallelized: it's already
// established elsewhere in this file as O(1) per tone, negligible next
// to the noise fill, and splitting it would only complicate the
// out-of-range-tone warning below for no real benefit.
func (s *DirectSynthesisStreamer) GenerateNextTick(t float64, nSamples int, dst map[string][][]complex128) {
	tLocalRelStart := t - s.obsTimeRef

	pols := [...]struct {
		pol       string
		isHPol    bool
		noiseSeed uint64
	}{
		{"V", false, s.noiseSeedV},
		{"H", true, s.noiseSeedH},
	}

	var wg sync.WaitGroup
	for _, p := range pols {
		target := dst[p.pol]
		if target == nil {
			continue
		}
		// Noise first, WRITTEN (not accumulated) so it covers every
		// cell -- that's what lets tone below skip zeroing the buffer.
		s.fillNoiseParallel(&wg, p.pol, p.noiseSeed, target, nSamples, tLocalRelStart)
	}
	wg.Wait()

	for _, p := range pols {
		target := dst[p.pol]
		if target == nil {
			continue
		}
		for _, cfg := range s.toneCfgs {
			poly := cfg.DelayFeed.Get(t)
			polyTRelStart := t - poly.StartValiditySec

			chIdx, samples := synthToneChannel(
				cfg.FreqHz,
				cfg.Amplitude,
				s.baseFreqHz,
				s.channelWidthHz,
				poly.XYPolCoeffsNs,
				polyTRelStart,
				poly.YPolOffsetNs,
				p.isHPol,
				tLocalRelStart,
				s.channelOutputRate,
				nSamples,
			)
			if chIdx < 0 || chIdx >= s.numChannels {
				log.Printf("tone freq_hz=%v maps to channel_idx=%d, outside the configured [0, %d) channel range — skipping", cfg.FreqHz, chIdx, s.numChannels)
				continue
			}
			outCh := target[chIdx]
			for i, sample := range samples {
				outCh[i] += sample
			}
		}
	}
}

// fillNoiseParallel fills dst (per channel, each already nSamples long)
// with this tick's noise-bank tile (or zeros, if noise isn't
// configured), registering s.numWorkers goroutines' worth of work on wg
// -- one goroutine per channel range, split via fillNoiseRange. Does NOT
// call wg.Wait() itself: GenerateNextTick calls this once per pol
// against one shared WaitGroup, so both pols' fills run fully in
// parallel with each other too, not just within a pol.
func (s *DirectSynthesisStreamer) fillNoiseParallel(wg *sync.WaitGroup, pol string, noiseSeed uint64, dst [][]complex128, nSamples int, tLocalRelStart float64) {
	if s.numWorkers <= 1 {
		s.fillNoiseRange(pol, noiseSeed, dst, nSamples, tLocalRelStart, 0, s.numChannels)
		return
	}
	chunkSize := (s.numChannels + s.numWorkers - 1) / s.numWorkers
	for chStart := 0; chStart < s.numChannels; chStart += chunkSize {
		chEnd := chStart + chunkSize
		if chEnd > s.numChannels {
			chEnd = s.numChannels
		}
		wg.Add(1)
		go func(chStart, chEnd int) {
			defer wg.Done()
			s.fillNoiseRange(pol, noiseSeed, dst, nSamples, tLocalRelStart, chStart, chEnd)
		}(chStart, chEnd)
	}
}

// fillNoiseRange fills dst[chStart:chEnd] -- one worker's channel range.
// dst[ch] is its own separately-owned backing array (HeapAccumulator's
// per-channel storage, not one contiguous multi-channel buffer as
// before), so each channel gets its own copy call rather than one bulk
// copy across the whole range -- same total bytes moved, no cross-worker
// synchronization needed beyond the caller's WaitGroup either way, since
// disjoint dst[ch] slices can never alias.
func (s *DirectSynthesisStreamer) fillNoiseRange(pol string, noiseSeed uint64, dst [][]complex128, nSamples int, tLocalRelStart float64, chStart, chEnd int) {
	if s.noiseCfg == nil {
		for ch := chStart; ch < chEnd; ch++ {
			out := dst[ch]
			for i := range out {
				out[i] = 0
			}
		}
		return
	}
	bank := s.banks[pol]
	tileLen := s.tileNSamples * s.numChannels
	n := nSamples
	if n < 1 {
		n = 1
	}
	tickIndex := int64(tLocalRelStart*s.channelOutputRate+0.5) / int64(n)
	tileIdx := int(splitmix64Hash(noiseSeed, uint64(tickIndex)) % uint64(s.nTiles))
	for ch := chStart; ch < chEnd; ch++ {
		bankOffset := tileIdx*tileLen + ch*nSamples
		copy(dst[ch], bank[bankOffset:bankOffset+nSamples])
	}
}
