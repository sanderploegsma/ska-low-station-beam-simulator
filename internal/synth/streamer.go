package synth

import (
	"fmt"
	"log"
	"runtime"
	"sort"
	"sync"

	"github.com/skao/station-beam-simulator-go/internal/common"
)

// DefaultNTiles matches the Python default (DEFAULT_N_TILES) — see
// docs/history.md for the fidelity/resource tradeoff this controls
// (birthday-paradox tile repeat vs. build time/memory).
const DefaultNTiles = 256

// ToneSourceConfig is a tone source — required DelayFeed, no
// default/fallback delay: a source with no real delay path would silently
// produce trivially "perfectly aligned" content, exactly the kind of
// thing that could mask a real CBF delay-tracking bug.
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
	banks        map[string][]complex64 // "V"/"H" -> flat (nTiles, tileNSamples, len(toneChannelPositions)) -- full-precision, but SUBSET-width, not numChannels-width: see NewDirectSynthesisStreamer
	quantBanks   map[string][]byte      // "V"/"H" -> flat quantized (nTiles, tileNSamples, numChannels) -- 2 bytes/sample, read for noiseOnlyChannelPositions (see GenerateQuantizedHeaps)

	// toneChannelPositions: channel positions (0..numChannels-1) with at
	// least one tone source targeting them -- FIXED for the whole scan
	// (a tone's channel depends only on its configured, static FreqHz,
	// never on per-tick delay), computed once at construction.
	// noiseOnlyChannelPositions is the complement. posToSubsetIdx maps
	// toneChannelPositions[j] -> j, for GenerateNextTick's tone-add loop
	// to find its dst slot (dst is sized to len(toneChannelPositions),
	// not numChannels -- see ComplexPathChannelIDMap). Splitting the two
	// lets noise-only channels (usually most of them) go through a
	// pre-quantized path instead of the full complex64 dst-write path.
	toneChannelPositions      []int
	noiseOnlyChannelPositions []int
	posToSubsetIdx            map[int]int

	numWorkers int
}

// NewDirectSynthesisStreamer validates cfg and builds a streamer,
// including filling the noise tile bank (if configured) — this is the
// same one-time, non-per-tick construction cost the Python version pays
// (see docs/history.md's "One-time construction budget" section).
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

	toneChannelSet := make(map[int]bool, len(cfg.ToneSources))
	for i, ts := range cfg.ToneSources {
		if ts.DelayFeed == nil {
			return nil, fmt.Errorf("tone source %d (freq_hz=%v) has no delay_feed -- every source must have its own delay path, there is no default/fallback delay", i, ts.FreqHz)
		}
		if ts.Amplitude == 0 {
			cfg.ToneSources[i].Amplitude = 1.0
		}
		if chIdx := toneChannelIndex(ts.FreqHz, baseFreqHz, channelWidthHz); chIdx >= 0 && chIdx < numChannels {
			toneChannelSet[chIdx] = true
		}
		// An out-of-range tone is NOT added to toneChannelSet here --
		// GenerateNextTick's own range check logs a warning and skips it
		// every tick instead.
	}

	// toneChannelPositions: sorted so ComplexPathChannelIDMap (and
	// therefore HeapAccumulator's channel ordering/dst indexing) is
	// deterministic run to run, not map-iteration-order-dependent.
	toneChannelPositions := make([]int, 0, len(toneChannelSet))
	for ch := range toneChannelSet {
		toneChannelPositions = append(toneChannelPositions, ch)
	}
	sort.Ints(toneChannelPositions)

	posToSubsetIdx := make(map[int]int, len(toneChannelPositions))
	noiseOnlyChannelPositions := make([]int, 0, numChannels-len(toneChannelPositions))
	for j, ch := range toneChannelPositions {
		posToSubsetIdx[ch] = j
	}
	for ch := 0; ch < numChannels; ch++ {
		if !toneChannelSet[ch] {
			noiseOnlyChannelPositions = append(noiseOnlyChannelPositions, ch)
		}
	}

	s := &DirectSynthesisStreamer{
		station:                   cfg.Station,
		numChannels:               numChannels,
		baseFreqHz:                baseFreqHz,
		channelWidthHz:            channelWidthHz,
		channelOutputRate:         channelOutputRate,
		obsTimeRef:                cfg.ObsTimeRef,
		toneCfgs:                  cfg.ToneSources,
		noiseCfg:                  cfg.Noise,
		banks:                     make(map[string][]complex64),
		quantBanks:                make(map[string][]byte),
		toneChannelPositions:      toneChannelPositions,
		noiseOnlyChannelPositions: noiseOnlyChannelPositions,
		posToSubsetIdx:            posToSubsetIdx,
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
		// Full-precision bank: only needed by channels that have to
		// combine noise with tone before quantizing -- skip entirely if
		// no tone is configured at all, since nothing would ever read it.
		// Sized to len(toneChannelPositions), NOT numChannels: this
		// subset is usually tiny (a handful of tone sources at most), so
		// building it at full channel width would waste construction
		// time/memory on channels nothing ever reads from it, just
		// because ONE channel needed the complex path. fillNoiseRange
		// reads this bank by SUBSET position directly (not the raw
		// channel index) to match.
		if len(toneChannelPositions) > 0 {
			s.banks["V"] = fillNoiseBank(s.noiseSeedV, s.noiseStd, s.nTiles, s.tileNSamples, len(toneChannelPositions))
			s.banks["H"] = fillNoiseBank(s.noiseSeedH, s.noiseStd, s.nTiles, s.tileNSamples, len(toneChannelPositions))
		}
		// Pre-quantized bank: only needed by noise-only channels -- skip
		// if every channel has a tone (nothing left for it to serve).
		// QuantizeScale() is safe to call here: toneCfgs/noiseStd are
		// already set on s above.
		if len(noiseOnlyChannelPositions) > 0 {
			scale := s.QuantizeScale()
			s.quantBanks["V"] = fillQuantizedNoiseBank(s.noiseSeedV, s.noiseStd, scale, s.nTiles, s.tileNSamples, s.numChannels)
			s.quantBanks["H"] = fillQuantizedNoiseBank(s.noiseSeedH, s.noiseStd, scale, s.nTiles, s.tileNSamples, s.numChannels)
		}
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
// ascending-frequency channel_id order. Covers EVERY channel this
// streamer produces, regardless of which path (complex or pre-quantized)
// actually produces it -- see ComplexPathChannelIDMap for the subset
// that uses the former.
func (s *DirectSynthesisStreamer) ChannelIDMap() []int {
	m := make([]int, s.numChannels)
	for i := range m {
		m[i] = i
	}
	return m
}

// ComplexPathChannelIDMap implements common.ComplexPathChannelIDMapper:
// the channel positions with at least one tone source targeting them --
// FIXED for the whole scan, see toneChannelPositions' doc comment on the
// struct. common.ScanRunner uses this to size its HeapAccumulator (and
// therefore GenerateNextTick's dst) to just this subset instead of every
// channel; the complement (GenerateQuantizedHeaps' channels) never goes
// through the complex64 dst-write path at all.
func (s *DirectSynthesisStreamer) ComplexPathChannelIDMap() []int {
	return s.toneChannelPositions
}

// NumChannels implements common.Streamer.
func (s *DirectSynthesisStreamer) NumChannels() int { return s.numChannels }

// TickNSamples implements common.Streamer — HeapLen by construction
// (BlockDurationS is defined for exactly this).
func (s *DirectSynthesisStreamer) TickNSamples() int {
	return int(s.channelOutputRate*common.BlockDurationS + 0.5)
}

// BankMemoryBytes reports total noise-bank memory (both the
// full-precision bank, if any channel needs it, and the pre-quantized
// bank, if any channel needs it) across both pols.
func (s *DirectSynthesisStreamer) BankMemoryBytes() int64 {
	var total int64
	if len(s.banks) > 0 {
		total += bankMemoryBytes(s.nTiles, s.tileNSamples, len(s.toneChannelPositions), len(s.banks))
	}
	if len(s.quantBanks) > 0 {
		total += quantizedBankMemoryBytes(s.nTiles, s.tileNSamples, s.numChannels, len(s.quantBanks))
	}
	return total
}

// quantizeSigmaMargin: how many standard deviations of noise-amplitude
// headroom QuantizeScale reserves above the largest configured tone
// amplitude, chosen so the probability of a real sample EVER clipping
// is negligible across any realistic deployment lifetime, not just "low
// for one heap." A complex sample's magnitude (sqrt(re²+im²), re/im each
// i.i.d. N(0, std²)) follows a Rayleigh(std) distribution, whose tail is
// P(magnitude > k·std) = exp(-k²/2). At k=8: ~1.3e-14 per SAMPLE. Even
// at 384 channels × 2 pols × ~10^8 ticks (a deliberately absurd
// multi-year-continuous-scanning upper bound -- real usage is nowhere
// near this), the expected number of samples that would EVER exceed
// this bound across that whole lifetime is under 0.001. This margin
// does NOT change how noise is GENERATED (still the same full-precision
// float64 Box-Muller draws, same statistics as before) -- it only
// changes how conservatively the already-generated value is digitized
// to fit int8 on the wire, i.e. it's a wire-format precision choice, not
// a physics/statistics one.
const quantizeSigmaMargin = 8.0

// QuantizeScale returns the fixed per-sample quantization scale this
// streamer's configuration implies (see
// spead.SpsPacketizer.SetQuantizeScale, which this feeds): the largest
// configured tone amplitude -- summed across every configured tone
// source, as if they all happened to land in the same channel and add
// exactly in phase, the true worst case, not just the typical one --
// plus quantizeSigmaMargin standard deviations of noise headroom. Using
// this instead of the alternative, per-heap adaptive scale
// (quantize8bitScale re-scanning every heap's actual samples for their
// own max magnitude, every tick) removes that scan from the per-tick hot
// path entirely. Returns 0 if this streamer has neither noise nor tone
// configured (an all-silent config, where the scale value is moot: every
// sample is exactly zero either way) -- SpsPacketizer.SetQuantizeScale
// treats 0 as "use the adaptive scale," which is harmless here since
// 0*anything=0 regardless of scale.
func (s *DirectSynthesisStreamer) QuantizeScale() float64 {
	bound := quantizeSigmaMargin * s.noiseStd
	for _, ts := range s.toneCfgs {
		bound += ts.Amplitude
	}
	if bound == 0 {
		return 0
	}
	return 127.0 / (bound + 1e-12)
}

// GenerateNextTick implements common.Streamer. t is the absolute epoch
// time of this tick's first sample; nSamples is the per-channel sample
// count for this tick (at channelOutputRate). Writes directly into
// dst[pol][j] for each configured pol, j indexing INTO
// toneChannelPositions (NOT a raw channel index -- dst is sized to
// ComplexPathChannelIDMap()'s subset, since noise-only channels never
// reach this method at all; see GenerateQuantizedHeaps for those). See
// common.Streamer's doc comment for why dst is written into directly
// rather than returned.
//
// Noise fill here is parallelized across both pols AND (what's usually a
// small) channel-position range at once -- V and H are already fully
// independent (separate seeds/buffers, separate dst entries), so no
// worker ever touches another's data. Tone injection is NOT
// parallelized: it's already established elsewhere in this file as O(1)
// per tone, negligible next to the noise fill, and splitting it would
// only complicate the out-of-range-tone warning below for no real
// benefit.
func (s *DirectSynthesisStreamer) GenerateNextTick(t float64, nSamples int, dst map[string][][]complex64) {
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
			j, ok := s.posToSubsetIdx[chIdx]
			if !ok {
				// Shouldn't happen: toneChannelPositions was built from
				// exactly these chIdx values at construction (see
				// NewDirectSynthesisStreamer). Guard anyway rather than
				// indexing target out of bounds if that invariant is
				// ever broken by a future edit.
				log.Printf("tone freq_hz=%v channel_idx=%d has no complex-path dst slot (subset invariant violated) — skipping", cfg.FreqHz, chIdx)
				continue
			}
			outCh := target[j]
			for i, sample := range samples {
				outCh[i] += sample
			}
		}
	}
}

// currentTileIndex picks this tick's noise-bank tile for the given pol
// seed -- shared by fillNoiseRange (full-precision bank, complex-path
// channels) and GenerateQuantizedHeaps (pre-quantized bank, noise-only
// channels) so both paths draw from the SAME tile on a given tick,
// keeping noise temporally consistent across the whole channel range
// regardless of which representation a given channel happens to use.
func (s *DirectSynthesisStreamer) currentTileIndex(noiseSeed uint64, tLocalRelStart float64, nSamples int) int {
	n := nSamples
	if n < 1 {
		n = 1
	}
	tickIndex := int64(tLocalRelStart*s.channelOutputRate+0.5) / int64(n)
	return int(splitmix64Hash(noiseSeed, uint64(tickIndex)) % uint64(s.nTiles))
}

// fillNoiseParallel fills dst (per channel-position-subset index j, each
// already nSamples long) with this tick's noise-bank tile (or zeros, if
// noise isn't configured), registering s.numWorkers goroutines' worth of
// work on wg -- one goroutine per subset range, split via fillNoiseRange.
// Does NOT call wg.Wait() itself: GenerateNextTick calls this once per
// pol against one shared WaitGroup, so both pols' fills run fully in
// parallel with each other too, not just within a pol. In practice
// len(toneChannelPositions) is usually small (0 to a handful of tone
// sources), so this rarely spawns more than one goroutine regardless of
// s.numWorkers -- kept structurally identical to the pre-split version
// anyway, since the actual per-tick cost this codebase cares about now
// lives in GenerateQuantizedHeaps, not here.
func (s *DirectSynthesisStreamer) fillNoiseParallel(wg *sync.WaitGroup, pol string, noiseSeed uint64, dst [][]complex64, nSamples int, tLocalRelStart float64) {
	n := len(s.toneChannelPositions)
	if n == 0 {
		return
	}
	if s.numWorkers <= 1 || n <= 1 {
		s.fillNoiseRange(pol, noiseSeed, dst, nSamples, tLocalRelStart, 0, n)
		return
	}
	chunkSize := (n + s.numWorkers - 1) / s.numWorkers
	for start := 0; start < n; start += chunkSize {
		end := start + chunkSize
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			s.fillNoiseRange(pol, noiseSeed, dst, nSamples, tLocalRelStart, start, end)
		}(start, end)
	}
}

// fillNoiseRange fills dst[subsetStart:subsetEnd] -- one worker's
// channel-position-subset range. dst[j] is its own separately-owned
// backing array (HeapAccumulator's per-channel storage, sized to
// ComplexPathChannelIDMap()'s subset -- not one contiguous
// multi-channel buffer), so each channel gets its own copy call rather
// than one bulk copy across the whole range -- same total bytes moved,
// no cross-worker synchronization needed beyond the caller's WaitGroup
// either way, since disjoint dst[j] slices can never alias.
// The bank itself is SUBSET-width too now (see NewDirectSynthesisStreamer's
// doc comment on why), so j is both dst's index AND the bank's column --
// no separate raw-channel lookup needed for the bank read, unlike an
// earlier version of this method.
func (s *DirectSynthesisStreamer) fillNoiseRange(pol string, noiseSeed uint64, dst [][]complex64, nSamples int, tLocalRelStart float64, subsetStart, subsetEnd int) {
	if s.noiseCfg == nil {
		for j := subsetStart; j < subsetEnd; j++ {
			out := dst[j]
			for i := range out {
				out[i] = 0
			}
		}
		return
	}
	bank := s.banks[pol]
	tileLen := s.tileNSamples * len(s.toneChannelPositions)
	tileIdx := s.currentTileIndex(noiseSeed, tLocalRelStart, nSamples)
	for j := subsetStart; j < subsetEnd; j++ {
		bankOffset := tileIdx*tileLen + j*nSamples
		copy(dst[j], bank[bankOffset:bankOffset+nSamples])
	}
}

// GenerateQuantizedHeaps implements common.QuantizedHeapProducer: builds
// complete, ready-to-send heaps DIRECTLY for every noise-only channel
// (no tone source targets it -- see noiseOnlyChannelPositions), bypassing
// GenerateNextTick's dst-write path and HeapAccumulator entirely. Each
// channel's samples are copied PRE-QUANTIZED straight out of
// quantBanks -- no per-tick float64 generation, no per-tick
// scan-and-round-and-clamp, both already done ONCE at construction (see
// fillQuantizedNoiseBank). This doesn't change what noise IS, only how
// it's digitized for the wire.
//
// Returns nil if every channel has a tone source (nothing left for this
// path to produce).
func (s *DirectSynthesisStreamer) GenerateQuantizedHeaps(t float64) []*common.ChannelHeap {
	if len(s.noiseOnlyChannelPositions) == 0 {
		return nil
	}
	n := s.TickNSamples() // == common.HeapLen by construction
	heaps := make([]*common.ChannelHeap, len(s.noiseOnlyChannelPositions))

	if s.noiseCfg == nil {
		// No noise configured -- these channels still emit a real
		// (all-zero) heap every tick, matching what the complex path's
		// zero-fill branch (fillNoiseRange) would have produced for
		// them: "no noise" is a real, silent signal CBF still expects
		// data for, not "no heap." No bank to read from here, so an
		// explicit zero-fill, not a copy.
		for idx, ch := range s.noiseOnlyChannelPositions {
			v := common.GetQuantizedBuffer()
			h := common.GetQuantizedBuffer()
			for i := range v {
				v[i] = 0
			}
			for i := range h {
				h[i] = 0
			}
			heaps[idx] = &common.ChannelHeap{ChannelID: ch, VQuantized: v, HQuantized: h, HeapStartTime: t}
		}
		return heaps
	}

	tLocalRelStart := t - s.obsTimeRef
	tileIdxV := s.currentTileIndex(s.noiseSeedV, tLocalRelStart, n)
	tileIdxH := s.currentTileIndex(s.noiseSeedH, tLocalRelStart, n)
	quantTileLen := s.tileNSamples * s.numChannels * 2 // 2 bytes/sample
	quantBankV := s.quantBanks["V"]
	quantBankH := s.quantBanks["H"]

	for idx, ch := range s.noiseOnlyChannelPositions {
		v := common.GetQuantizedBuffer()
		h := common.GetQuantizedBuffer()
		vOffset := tileIdxV*quantTileLen + ch*n*2
		hOffset := tileIdxH*quantTileLen + ch*n*2
		copy(v, quantBankV[vOffset:vOffset+n*2])
		copy(h, quantBankH[hOffset:hOffset+n*2])
		heaps[idx] = &common.ChannelHeap{ChannelID: ch, VQuantized: v, HQuantized: h, HeapStartTime: t}
	}
	return heaps
}
