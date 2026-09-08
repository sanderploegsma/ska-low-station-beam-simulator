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

	outBufs    map[string][]complex128 // reused per tick, per pol
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
		outBufs:           make(map[string][]complex128),
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

// defaultParallelism caps worker/goroutine counts consistently across
// this package: runtime.GOMAXPROCS(0) (which — unlike runtime.NumCPU()
// — respects a Kubernetes pod's CPU request/limit) capped to n (no point
// spawning more workers than units of independent work) and to 16 (this
// project's own established ceiling — see fillNoiseBank's identical cap
// for noise-bank construction).
func defaultParallelism(n int) int {
	w := runtime.GOMAXPROCS(0)
	if w > n {
		w = n
	}
	if w > 16 {
		w = 16
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

func (s *DirectSynthesisStreamer) getOutputBuffer(pol string, nSamples int) []complex128 {
	buf := s.outBufs[pol]
	if len(buf) != nSamples*s.numChannels {
		buf = make([]complex128, nSamples*s.numChannels)
		s.outBufs[pol] = buf
	}
	return buf
}

// GenerateNextTick implements common.Streamer. t is the absolute epoch
// time of this tick's first sample; nSamples is the per-channel sample
// count for this tick (at channelOutputRate). Returns pol -> flat,
// CHANNEL-MAJOR (numChannels, nSamples) complex128 (index =
// channel*nSamples+sample) — see common.Streamer's doc comment for why.
//
// Profiling a real end-to-end run on the target hardware found
// ScanRunner's single producer goroutine (this method plus
// HeapAccumulator's Add/PopReadyHeaps) running at ~99% duty cycle on ONE
// core for the whole scan, unable to keep pace even after every
// allocation-related fix upstream and downstream of it -- unlike
// sending (already parallelized across SenderPool's goroutines),
// generation had never been split across cores at all. Since noise
// fill dominates this method's own cost (a bulk copy from the tile
// bank, O(numChannels)) and every channel's copy is independent, it's
// parallelized here across both pols AND channel ranges at once -- V
// and H are already fully independent (separate seeds/buffers), and
// channel-major layout means a channel range is one contiguous slice on
// both the bank (source) and out (destination), so no worker ever
// touches another's data. Tone injection is NOT parallelized: it's
// already established elsewhere in this file as O(1) per tone,
// negligible next to the noise fill, and splitting it would only
// complicate the out-of-range-tone warning below for no real benefit.
func (s *DirectSynthesisStreamer) GenerateNextTick(t float64, nSamples int) map[string][]complex128 {
	tLocalRelStart := t - s.obsTimeRef

	pols := [...]struct {
		pol       string
		isHPol    bool
		noiseSeed uint64
	}{
		{"V", false, s.noiseSeedV},
		{"H", true, s.noiseSeedH},
	}

	results := make(map[string][]complex128, 2)
	var wg sync.WaitGroup
	for _, p := range pols {
		out := s.getOutputBuffer(p.pol, nSamples)
		results[p.pol] = out
		// Noise first, WRITTEN (not accumulated) so it covers every
		// cell -- that's what lets tone below skip zeroing the buffer.
		s.fillNoiseParallel(&wg, p.pol, p.noiseSeed, out, nSamples, tLocalRelStart)
	}
	wg.Wait()

	for _, p := range pols {
		out := results[p.pol]
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
			// Channel-major out: chIdx's samples are already contiguous,
			// so this is a sequential add, not a strided one.
			outCh := out[chIdx*nSamples : (chIdx+1)*nSamples]
			for i, sample := range samples {
				outCh[i] += sample
			}
		}
	}
	return results
}

// fillNoiseParallel fills out (channel-major, numChannels*nSamples) with
// this tick's noise-bank tile (or zeros, if noise isn't configured),
// registering s.numWorkers goroutines' worth of work on wg -- one
// goroutine per channel range, split via fillNoiseRange. Does NOT call
// wg.Wait() itself: GenerateNextTick calls this once per pol against one
// shared WaitGroup, so both pols' fills run fully in parallel with each
// other too, not just within a pol.
func (s *DirectSynthesisStreamer) fillNoiseParallel(wg *sync.WaitGroup, pol string, noiseSeed uint64, out []complex128, nSamples int, tLocalRelStart float64) {
	if s.numWorkers <= 1 {
		s.fillNoiseRange(pol, noiseSeed, out, nSamples, tLocalRelStart, 0, s.numChannels)
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
			s.fillNoiseRange(pol, noiseSeed, out, nSamples, tLocalRelStart, chStart, chEnd)
		}(chStart, chEnd)
	}
}

// fillNoiseRange fills out[chStart*nSamples : chEnd*nSamples] -- one
// worker's contiguous channel range (channel-major layout makes this a
// single contiguous slice on both out and the noise bank, so no cross-
// worker synchronization is needed beyond the caller's WaitGroup).
func (s *DirectSynthesisStreamer) fillNoiseRange(pol string, noiseSeed uint64, out []complex128, nSamples int, tLocalRelStart float64, chStart, chEnd int) {
	outRange := out[chStart*nSamples : chEnd*nSamples]
	if s.noiseCfg == nil {
		for i := range outRange {
			outRange[i] = 0
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
	bankOffset := tileIdx*tileLen + chStart*nSamples
	copy(outRange, bank[bankOffset:bankOffset+(chEnd-chStart)*nSamples])
}
