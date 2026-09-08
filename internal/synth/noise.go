package synth

import (
	"math/rand/v2"
	"runtime"
	"sync"

	"github.com/skao/station-beam-simulator-go/internal/spead"
)

// golden is the splitmix64 golden-ratio constant — same algorithm and
// constants as the Python codebase's _splitmix64_hash, ported for
// consistency (bit-identical output to the Python version is not
// required: noise only needs to be a well-distributed, deterministic,
// SEEKABLE hash — see splitmix64Hash's doc comment — not
// cross-language-identical bytes).
const golden uint64 = 0x9E3779B97F4A7C15

// splitmix64Hash is a deterministic (seed, index) -> uint64 hash, used
// ONLY to pick each tick's noise tile index
// (splitmix64Hash(seed, tickIndex) % nTiles) — not a statistical
// distribution, just a well-distributed integer. Must be a PURE FUNCTION
// of index alone (no state to track across calls): ScanRunner/tests can
// and do call GenerateNextTick with the same t twice and require
// byte-identical output, which rules out a stateful/sequential generator.
// Go's uint64 arithmetic wraps mod 2^64 on overflow by definition, so no
// explicit masking is needed (unlike the numpy port, which masks
// explicitly since numpy doesn't guarantee wraparound the same way).
func splitmix64Hash(seed, index uint64) uint64 {
	x := seed ^ (index * golden)
	x += golden
	z := x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	z = z ^ (z >> 31)
	return z
}

// fillNoiseBank fills an nTiles-tile bank (flat: tile i's data lives at
// bank[i*tileNSamples*numChannels : (i+1)*tileNSamples*numChannels]) of
// independent complex Gaussian noise. NO delay applied — physically
// correct for receiver noise, which originates locally at each station
// after any signal-path delay would apply.
//
// Each tile's tileNSamples*numChannels elements are filled in plain
// sequential order below, but GenerateNextTick later copies a whole tile
// verbatim into its channel-major (numChannels, tileNSamples) per-tick
// output buffer — that reinterpretation is safe with NO transpose
// needed here, specifically because every element is an independent,
// identically-distributed draw: which (channel, sample) position a given
// value ends up landing on is statistically irrelevant, unlike tone's
// per-channel structure, which does require producing (and adding into)
// data in the right layout position by construction.
//
// Parallelized across goroutines, each with its own independently-seeded
// PCG stream (derived from the base seed via splitmix64Hash, analogous to
// numpy.random.SeedSequence.spawn()) — each worker writes directly into
// its own disjoint slice of one preallocated bank, one tile at a time,
// bounding transient memory to a handful of tiles regardless of the
// bank's total size (mirroring a real memory bug the Python
// implementation hit and fixed: an earlier version there returned one
// freshly-allocated array per worker and concatenated them, peaking at
// several times the bank's own footprint and OOM-killing the process —
// see docs/history.md. Writing into disjoint slices of one
// preallocated slice, as here, avoids that class of bug entirely, by
// construction, not by later measurement.
func fillNoiseBank(seed uint64, std float64, nTiles, tileNSamples, numChannels int) []complex64 {
	tileLen := tileNSamples * numChannels
	bank := make([]complex64, nTiles*tileLen)
	if nTiles == 0 {
		return bank
	}

	nWorkers := nTiles
	if cpus := runtime.NumCPU(); nWorkers > cpus {
		nWorkers = cpus
	}
	if nWorkers > 16 {
		nWorkers = 16
	}
	if nWorkers < 1 {
		nWorkers = 1
	}

	base := nTiles / nWorkers
	remainder := nTiles % nWorkers

	var wg sync.WaitGroup
	start := 0
	for w := 0; w < nWorkers; w++ {
		chunk := base
		if w < remainder {
			chunk++
		}
		if chunk == 0 {
			continue
		}
		workerSeed1 := splitmix64Hash(seed, uint64(2*w))
		workerSeed2 := splitmix64Hash(seed, uint64(2*w+1))

		wg.Add(1)
		go func(start, chunk int, seed1, seed2 uint64) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(seed1, seed2))
			for i := start; i < start+chunk; i++ {
				tile := bank[i*tileLen : (i+1)*tileLen]
				for j := range tile {
					// Box-Muller itself runs in float64 (rand/v2's
					// NormFloat64) -- only the final draw narrows to
					// complex64 on store, same principle as every other
					// precision-sensitive-computation/lossy-final-output
					// split in this codebase (see ChannelHeap's doc
					// comment).
					tile[j] = complex64(complex(rng.NormFloat64()*std, rng.NormFloat64()*std))
				}
			}
		}(start, chunk, workerSeed1, workerSeed2)
		start += chunk
	}
	wg.Wait()
	return bank
}

// bankMemoryBytes returns the total resident memory for a noise bank
// across nPols polarisations (complex64 = 8 bytes).
func bankMemoryBytes(nTiles, tileNSamples, numChannels, nPols int) int64 {
	return int64(nTiles) * int64(tileNSamples) * int64(numChannels) * 8 * int64(nPols)
}

// fillQuantizedNoiseBank is fillNoiseBank's PRE-QUANTIZED counterpart:
// same tile layout, same per-worker seed derivation (so passing the
// SAME seed/std/nTiles/tileNSamples/numChannels as fillNoiseBank
// produces the identical underlying draws, just stored differently --
// see TestFillQuantizedNoiseBank_MatchesFillNoiseBankScaled, which
// checks exactly this), but each draw is quantized to an int8 (real,
// imag) pair (scale: see DirectSynthesisStreamer.QuantizeScale) and
// stored as 2 bytes instead of complex64's 8 -- a quarter the memory,
// and (the actual point) a quarter the per-tick bytes GenerateQuantizedHeaps
// has to copy out of it, on top of skipping the per-tick
// round+clamp entirely (already done here, once).
//
// Bank layout: bank[(i*tileNSamples*numChannels+k)*2 : +2] is tile i's
// k-th (real,imag) sample pair, k = ch*tileNSamples+sample -- same
// (tile, channel, sample) -> flat-index mapping as fillNoiseBank, just
// 2 bytes/sample instead of 8.
func fillQuantizedNoiseBank(seed uint64, std, scale float64, nTiles, tileNSamples, numChannels int) []byte {
	tileLen := tileNSamples * numChannels
	bank := make([]byte, nTiles*tileLen*2)
	if nTiles == 0 {
		return bank
	}

	nWorkers := nTiles
	if cpus := runtime.NumCPU(); nWorkers > cpus {
		nWorkers = cpus
	}
	if nWorkers > 16 {
		nWorkers = 16
	}
	if nWorkers < 1 {
		nWorkers = 1
	}

	base := nTiles / nWorkers
	remainder := nTiles % nWorkers

	var wg sync.WaitGroup
	start := 0
	for w := 0; w < nWorkers; w++ {
		chunk := base
		if w < remainder {
			chunk++
		}
		if chunk == 0 {
			continue
		}
		workerSeed1 := splitmix64Hash(seed, uint64(2*w))
		workerSeed2 := splitmix64Hash(seed, uint64(2*w+1))

		wg.Add(1)
		go func(start, chunk int, seed1, seed2 uint64) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(seed1, seed2))
			for i := start; i < start+chunk; i++ {
				tile := bank[i*tileLen*2 : (i+1)*tileLen*2]
				for j := 0; j < tileLen; j++ {
					re := rng.NormFloat64() * std
					im := rng.NormFloat64() * std
					tile[j*2] = byte(spead.QuantizeComponent(re * scale))
					tile[j*2+1] = byte(spead.QuantizeComponent(im * scale))
				}
			}
		}(start, chunk, workerSeed1, workerSeed2)
		start += chunk
	}
	wg.Wait()
	return bank
}

// quantizedBankMemoryBytes returns the total resident memory for a
// quantized noise bank across nPols polarisations (2 bytes/sample --
// a quarter of bankMemoryBytes' complex64 8 bytes/sample).
func quantizedBankMemoryBytes(nTiles, tileNSamples, numChannels, nPols int) int64 {
	return int64(nTiles) * int64(tileNSamples) * int64(numChannels) * 2 * int64(nPols)
}
