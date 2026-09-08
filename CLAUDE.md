# CBF Station-Beam Simulator

Software simulator generating SKA Low SPS station-beam data, replacing the
CNIC hardware/firmware tool currently used to test the CBF
correlator/beamformer's delay-tracking, correlation, and beamforming
logic. Independence from CBF matters here specifically: CNIC is built by
the same team that builds the CBF firmware being tested, which undermines
the value of the test. This simulator must generate "true" delay
independently of whatever CBF itself computes.

Deployment target: one Tango device server per station, one Kubernetes
pod per device, sending real SPEAD/UDP heaps to CBF per the SPS-CBF ICD.
Currently developed/benchmarked at 96 channels (75 MHz); the real
**CONFIRMED (against the real ICD text, not a screenshot or an
assumption) maximum is 384 channels (300 MHz)** — the band is
channelized as 384 equispaced coarse channels, configurable from 8 to
384 in steps of 8. **448 channels/350MHz, used throughout earlier
sessions as "the full band," was never a real configuration this
hardware supports** — see "SPS-CBF ICD channelization" below for the
full correction (also: the true first/lowest channel is ID 64, not 65,
and each channel's actual per-sample period is 1080ns, not the
1280ns a critically-sampled channelizer would give — a real ~15.6%
tightening of the per-tick budget). `DirectSynthesisStreamer` now
validates `num_channels` against this range and raises otherwise. See
Benchmarking below for updated resource/timing numbers at the real
384-channel maximum.

## Architectural decisions (settled, don't relitigate without new evidence)

- **Independent delay generation, not shared with CBF.** The simulator
  generates "true" delay from its own source geometry; CBF's own real
  delay-poly Tango device (not a reimplementation, not routed through
  TMC) supplies the polynomial CBF actually applies. Two independent code
  paths computing related-but-different quantities from the same
  underlying source position/geometry — generating station data with the
  SAME polynomial CBF corrects with would be tautological.
- **CSP LMC drives the scan, not TMC** — avoids TMC's full subarray
  observation lifecycle. The delay-poly schema
  (`ska-low-csp-delaymodel/1.0`, ADR-88 in `ska-telmodel`) is a
  documented wire-format interface; TMC is the usual producer but not
  architecturally required. CBF's own delay-poly Tango device (used for
  CBF's own testing, commands like `PstOffsetRaDec`) proves a non-TMC
  injection path exists. **Open item:** the exact CSP LMC command for
  this is still not confirmed.
- **Deterministic, clock-independent `sim_time`.** Content generation
  never depends on any pod's own wall clock — only on `obs_time_ref`
  (given once, identically, to every station pod at scan start) plus a
  derived tick index / relative time. This is what makes multi-pod
  generation consistent without PTP or inter-pod coordination beyond the
  initial `obs_time`. Every generation kernel in this codebase (tone
  phase, noise sample index, delay polynomial evaluation) is a pure
  function of a small, `obs_time`-relative `t`, specifically so any pod
  can independently compute any tick without shared state.
- **Each station is represented by one Tango device server, each running in a Kubernetes Pod**, self-sufficient after
  receiving `obs_time` at scan start.
- **TAI2000 is the SKA epoch** for `heap_counter`. Unix→TAI2000 uses
  `astropy` (`common.unix_to_tai2000_seconds`) — a hardcoded-leap-second
  fallback exists but is explicitly unsafe for production; don't deploy
  it without astropy installed.

## Code layout

```
tango/src/ska_low_station_beam_simulator/
  common.py                    shared plumbing, backend-agnostic
  spead.py                      hand-rolled SPEAD-64-48 heap encoding (SpsPacketizer) -- not spead2,
                                see the ICD section below and bug #17 -- split out of common.py so
                                the SPEAD wire-format concern doesn't live alongside common.py's
                                config/delay/producer-sender plumbing
  direct_synthesis.py          the SOLE backend: DirectSynthesisStreamer (tone + tiled noise + pulsar) --
                                still used directly by the scripts/tests below; no longer driven by
                                simulator.py itself (see "Tango device now drives the Go gRPC simulator")
  simulator.py                 Tango device server (StationSimulatorDevice) -- now a gRPC CLIENT of the
                                Go simulator (repo root's cmd/simulator), not a local generator; see
                                "Tango device now drives the Go gRPC simulator" below
  simulatorpb/                 generated gRPC/protobuf stubs (simulator_pb2.py/simulator_pb2_grpc.py)
                                from the repo root's api/simulator.proto -- regenerate per "Setup" below
  benchmark_direct_synthesis.py  benchmarks DirectSynthesisStreamer, incl. tone+noise+pulsar combined
  generate_test_pcap.py        writes a real pcap of a few SPEAD-encoded heaps, for testing the
                                encoding path against an external unpacker (see spead.py's
                                hand-rolled SPEAD-64-48 encoder -- not spead2, see the ICD section
                                below -- and bug #17)
  pulsar_catalog.py             named pulsar catalog: save/load pre-generated templates by name
                                (see "Pulsar catalog" below) -- no dependency on direct_synthesis.py,
                                so DirectSynthesisStreamer can import it without a cycle
  generate_pulsar_catalog.py   builds every entry in pulsar_catalog.CATALOG_ENTRIES and writes it
                                to disk -- the only module that imports both direct_synthesis.py
                                (build_pulsar_template) and pulsar_catalog.py (to save the result)
```

(This lives under `tango/` at the repo root, alongside an experimental
Go port of the same numeric core -- `cmd/`/`internal/`/`api/` at the
repo root, `go.mod`/`go.sum`/`Dockerfile` included, see its own
`README.md` at the repo root. `pyproject.toml`/`uv.lock` stay at the
repo root too, covering only the Python package under `tango/src/`.)

**This used to be two backends plus three separate experimental
prototype modules; all of that has been converged into the one file
above.** `wideband_streamer.py` (the legacy wideband+FFT `StationStreamer`,
kept only as a pulsed-source fallback), `tiled_noise_streamer.py`
(experimental noise tile-bank), `pulsed_source_streamer.py` (experimental
per-channel pulsar), and their dedicated benchmark scripts have all been
**deleted** — see git history if you need to look at how any of them
worked before the merge. `direct_synthesis.py` now handles tone, per-pol
station noise (via a pre-generated tile bank), and pulsed/pulsar sources
directly, with no wideband+FFT fallback path at all. `scipy` was dropped
from `pyproject.toml` as part of this — it was only ever used by the
now-deleted `wideband_streamer.py`.

`common.py` defines a `Streamer` `Protocol` (`channel_id_map`,
`num_channels`, `tick_n_samples()`, `generate_next_tick()`) that
`ScanRunner` drives structurally — it never imports
`DirectSynthesisStreamer` (the sole implementation), so `common.py` stays
testable independent of the generation strategy behind it.
**`simulator.py` no longer constructs a `DirectSynthesisStreamer` at
all — see "Tango device now drives the Go gRPC simulator" below.** This
paragraph still describes `direct_synthesis.py`/`common.py` accurately
(unchanged); only the Tango-facing device server's own wiring moved.

## Signal generation

**`direct_synthesis.DirectSynthesisStreamer` is the sole backend**,
handling tone, per-pol station (receiver) noise, and pulsed/pulsar
sources, all via direct, per-channel synthesis — no per-tick wideband
generation, no per-tick FFT channelization for any source type. This is
the result of converging what used to be two backends (a legacy
wideband+FFT `StationStreamer`, kept only as a pulsed-source fallback)
plus three separate experimental prototype modules into one file — see
"Code layout" above. The legacy wideband path (`wideband_streamer.py`)
is **deleted**, not just superseded; see git history if you need its
old ring-buffer/coarse-fine-delay/FFT-channelizer implementation for
reference.

### Tone

Spectrally sparse — lives almost entirely in one channel bin after
channelization. Synthesized directly via a closed-form complex
exponential at the residual frequency; delay is a **continuous phase
term** in the exponent, exact for a monochromatic tone (not an
approximation of an integer/fractional sample split — that split is an
artifact of time-domain discretization, sidestepped entirely by direct
phase modulation). No ring buffer, no coarse/fine delay split. Cost is
O(1) per tone, independent of channel count. Correctness-verified
(`python -m ska_low_station_beam_simulator.direct_synthesis`): lands in
the correct channel with the correct residual frequency; delay-as-phase
matches the analytic `-2π·freq·τ` shift to ~1e-14.

### Noise — pre-generated tile bank

The DFT of i.i.d. complex Gaussian noise is itself i.i.d. complex
Gaussian (unitary transform), so per-tick noise could in principle be
generated directly at (sample, channel) resolution with no FFT at all —
but at 448 channels that's still O(channels) of live Box-Muller work per
tick, and an earlier version of this codebase needed ~24-48 threads to
clear budget doing it that way (see Benchmarking). The noise strategy
actually used now is a **pre-generated tile bank**: `n_tiles` tiles of
`tile_n_samples` ("tile width", both configurable on
`DirectSynthesisStreamer`) are generated once at construction; each tick
does an O(1) per-(station, pol) index draw plus a memcopy instead. Noise
never enters a delay pipeline either way — physically correct, since
receiver noise originates locally per station, after any signal-path
delay would apply (this was a real bug in the deleted wideband path:
`shared` and `noise` were summed *before* its delay/channelization
pipeline, delay-correcting noise that should never have been
delay-corrected at all).

**Filling the bank is plain `numpy.random.Generator`, not custom numba
Box-Muller, as of this session.** The original hand-rolled splitmix64
hash + Box-Muller kernel existed specifically because live PER-TICK
generation under numba needed it (numba's nopython mode doesn't support
numpy's own Philox/PCG64 — bug #7). Once noise moved to "generate once,
replay per tick" (this section), that constraint stopped applying to the
BUILD step — only the per-tick tile-index pick still needs custom numba
(`_splitmix64_hash`, kept: it's a single integer hash, not a statistical
distribution, and must run with zero allocation from inside
`generate_next_tick`). `fill_noise_bank` now parallelizes across a plain
`ThreadPoolExecutor` over independent `numpy.random.SeedSequence`
children — numpy's Generator releases the GIL during generation, so this
is genuine multi-core speedup with no custom numerical code at all.
**Measured faster than the numba version it replaced, not just
"acceptable"**: 0.67s vs. numba's 2.2s at n_tiles=256, 2.5s vs. 8.8s at
n_tiles=1024 (both pols, 448 channels) — see the updated table below.
`_copy_tile_into` (an optional numba-parallel per-tick memcopy) was
deleted outright as dead weight: `parallel_copy` was never set `True` in
production or the benchmark, and the plain `out[:] = bank[idx]` path
below was already known to be sufficient.

**A real memory bug was caught here, not just a design choice** (bug #16
below): the first replacement had each worker thread return a freshly
allocated chunk array, concatenated at the end — this peaked at several
times the bank's own footprint (chunk arrays + concatenate's
source-and-destination all alive simultaneously) and OOM-killed the
process (and, since this ran on a shared node, other tenants' pods along
with it) building a realistically-sized bank. Fixed by having each
worker write directly into its slice of one preallocated bank array, one
TILE AT A TIME (bounding transient memory to a handful of tiles
regardless of total bank size) — confirmed via `/usr/bin/time -v` under
a `ulimit -v` safety net: peak RSS now matches the bank's own size almost
exactly, no multiplier.

**Correctness constraint, non-negotiable:** the tile index MUST be drawn
from each station's own (station, pol) noise seed — never from a value
shared across stations (e.g. seeded only from `obs_time`). Sharing it
would make every station emit byte-identical "noise" for a given tick,
silently breaking any test that depends on receiver noise being
uncorrelated across the array (beamforming-SNR gain, cross-correlation
baseline noise floor). `DirectSynthesisStreamer` gets this right by
construction; its `__main__` includes a cross-station-independence check
specifically to guard against a future edit "simplifying" this back to a
shared seed.

**The real cost of this approach is fidelity, not CPU or memory usage,
and no bank size that fits in memory fixes it.** By the birthday
paradox, a station's own tile-index sequence hits its first repeat after
~1.25×√n_tiles ticks. Measured on the 2-socket EPYC 9254 target hardware,
448 channels, 8 threads:

| n_tiles | bank size (both pols) | build time (numba, historical) | build time (numpy Generator, current) | first repeat (~ticks) | ~scan time |
|---|---|---|---|---|---|
| 256 | 7.0 GB | 2.2s | **0.67s** | 20 | 52ms |
| 512 | 15.0 GB | — | **1.29s** | 28 | 73ms |
| 1024 | 28.0 GB | 8.8s | **2.54s** | 40 | 105ms |
| 2048 | 56.0 GB | 31.9s | not re-measured | 57 | 148ms |

Going another order of magnitude in `n_tiles` (well past feasible
per-pod memory) would still only push the first repeat into the
low-single-digit seconds — nowhere near long enough for a test that
checks a single station's long-integration noise-floor behavior (total
power should keep averaging down with more integration time; past the
repeat cycle, it won't). **Only use this in place of
`DirectSynthesisStreamer` if CBF's test suite doesn't rely on that** —
per-tick and cross-station statistics (delay-tracking, correlation,
beamforming functional tests) are unaffected by one station's own
periodicity.

**CPU cost turned out to be a non-issue — this is the headline result.**
Once the bank exists, per-tick cost is just an index hash + a ~14.7MB
memcopy, at 448 channels: plain `out[:] = bank[idx]` (single-threaded
numpy, ignores thread count entirely) is **1.51ms/tick (57.5% of
budget) regardless of core count** — even 1 core is enough. (An earlier,
now-deleted `_copy_tile_into` numba-parallel variant got this down to
0.36ms/13.9% at 8 threads, but was never actually used — `parallel_copy`
defaulted `False` everywhere — so it was removed as unused custom code
rather than kept "just in case"; revisit only if the plain-numpy copy
above is ever shown to be the actual per-tick bottleneck, which it isn't
at 448 channels.)

So **the target of ~8 CPU cores/pod is not just achievable but massively
over-provisioned for this approach's steady-state cost** — 1-2 cores
would already clear budget. The actual per-pod resource question this
approach shifts onto is **memory** (linear in `n_tiles`, ~27MB/tile
across both pols at 448 channels) and **one-time startup latency** (see
the build-time columns above — comfortably inside the 10s construction
target even at n_tiles=1024, now that the fill is plain numpy).

**Bottom line**: if CBF's tests tolerate per-station noise periodicity on
the order of tens to hundreds of milliseconds repeating throughout a
scan, this trades a small, one-time memory/startup cost for an
essentially-free steady-state CPU footprint — this is precisely why the
tile bank (not live per-tick Box-Muller generation) was adopted as the
noise strategy for the converged backend: live generation alone needed
~24-48 threads for headroom at 448 channels (see Benchmarking), and the
whole point of converging was fitting comfortably within an ~8-core/pod
budget. Revisit this choice if a test surfaces that specifically needs
per-station long-integration noise-floor accuracy — that's the one thing
this trade doesn't cover.

### Pulsed (pulsar) sources

Answers the "Immediate next steps" item this file had flagged for a
while: a direct per-channel representation for pulsed sources, so the
legacy wideband path could be deleted. Grew directly out of the
tiled-noise-bank work — could "generate once, replay per tick" work for
a pulsar too? Handles `kind='tone'` and `kind='pulsed'` together (plus
per-pol noise) in the same `DirectSynthesisStreamer`.

**Why this is a fundamentally better fit than the noise tile bank**: a
pulsar is genuinely periodic. Replaying one precomputed period isn't a
fidelity compromise the way replaying a finite noise bank is — a real
pulsar's profile repeats exactly, to the precision this simulator needs,
every rotation period. No birthday-paradox tradeoff, no long-integration
correctness caveat: periodicity here is ground truth, not an artifact.

**This module went through three designs before arriving at the current
one** — the history matters because each wrong version looked reasonable
until it was actually built and checked, not just reasoned about:

- **v1 (wrong)**: each channel sees one constant DM delay (evaluated at
  that channel's center frequency only), applied to a real-valued
  achromatic envelope; per-tick geometric delay via a first-order Taylor
  correction (`envelope(t-tau) ≈ envelope(t) - tau*envelope'(t)`).
- **v2 (fixed intra-channel smear, still broken for beamforming)**: a
  channel isn't one frequency, it's a ~781kHz passband — at SKA-Low
  frequencies, even DM=2 pc/cm³ (a low, realistic value) smears the
  dispersion curve across ~81,000 channel-widths at the bottom of the
  band (50MHz) and ~159 at the top (400MHz), a real low-frequency-radio
  effect, not a bug. v2 fixed this by averaging many shifted copies of
  the profile across each channel's own passband (`n_subfreq=1`, the
  broken version, predicted a channel peak of 1.0 with no smearing at
  all; the converged answer was 0.288 — a 3.5x error). But v2 was still
  real-valued (no carrier), which turned out to be a bigger problem:
  CBF's beamformer coherently combines stations by phase-rotating
  already-channelized complex data — physically valid only because a
  real channelizer inherently produces complex output with genuine
  carrier phase. Real-valued content has no phase for that rotation to
  act on, so v2 could not be coherently beamformed across stations at
  all (tone doesn't have this problem — it has a genuine
  residual-frequency carrier by construction).
- **v3 (current)**: generate the wideband, undispersed pulse train as
  one real time series spanning the whole band (a shared "sky carrier" —
  see below), apply the standard coherent-dispersion transfer function
  (Lorimer & Kramer 2006, eq. 5.21) directly to its full complex FFT,
  then channelize via `_channelize_once` (a small standalone one-shot
  channelizer — ported from the legacy `WidebandChannelizer` when that
  module was deleted, since this was its one remaining use) — a
  legitimate ONE-TIME, offline call, not a hot-path FFT, so it doesn't
  violate this codebase's "direct synthesis, no FFT per tick"
  philosophy. This fixes both v1/v2 problems at once: intra-channel
  smear falls out correctly as an emergent property of dispersing at
  full wideband FFT resolution before channelizing (no averaging hack
  needed), and channelizing a real signal via FFT inherently produces
  genuinely complex per-channel content with real carrier phase — which
  lets the per-tick geometric-delay correction use tone's EXACT phase
  trick instead of v1's Taylor approximation.

**Verified against an external, peer-reviewed reference, not just
internal self-consistency**: investigated using NANOGrav's `PsrSigSim`
package directly, but its dependency chain is heavy and partially broken
for this purpose (pulls in PINT, `fitsio`, `emcee`, `nestle`, matplotlib
just to import; its own `BasebandSignal.to_RF`/`to_FilterBank` conversion
methods are unimplemented stubs) — so instead of depending on it, its
`ISM._disperse_baseband` implementation was read directly and its
physics reimplemented in this module. This paid off as a real check: its
dispersion constant (`DM_K = 1/2.41e-4 = 4149.38`) matches this module's
own (`4148.808`) to 0.014% — the same standard literature constant,
cross-validated independently.

**Two real bugs caught only by numerical verification, not by
inspection** — worth internalizing as a general lesson for this module:
1. The wideband dispersion step originally used `np.fft.rfft`/`irfft`
   (correct for PsrSigSim's own real-ADC-sampled convention), but this
   codebase's own convention (see `_channelize_once` and the deleted
   `WidebandChannelizer.channel_center_frequencies` it was ported from)
   treats negative `fftfreq` bins as meaningful, independent channels —
   using `rfft` silently covered only half the intended band and shifted
   every channel's frequency. Caught by injecting a KNOWN tone at a
   known external channel and checking where it actually landed (it was
   off), not by reasoning about the convention. Fixed by using the full
   complex `fft`/`ifft` throughout — see `__main__`'s channel-mapping
   check, which is now a permanent regression test for this.
2. The first version of the per-tick exact-phase correction called
   `cos`/`sin` once per (sample, channel) pair — ~1.8M transcendental
   calls per tick at 448 channels — needing 32+ threads to clear budget,
   4x worse than expected. Since phase is linear in channel index at a
   fixed sample, fixed with a phase-accumulator (NCO-style) recurrence:
   two trig calls per SAMPLE, then one complex multiply per channel to
   step the rotation forward — the same "per-element transcendental
   calls are the expensive part" lesson this session's noise-kernel work
   already established, reapplied here.

**Cross-station coherence, verified numerically, not assumed**: two
streamers with different `DelayPolynomial`s (simulating two stations),
same pulsar, same tick — after each applies its OWN delay-compensating
phase correction (what a beamformer does), their content's normalized
correlation is 1.0000. This is the concrete confirmation that v3 (unlike
v1/v2) actually supports coherent multi-station beamforming — see
`__main__`.

**Why the "sky carrier" is shared across all stations, not per-station —
opposite of the rule for noise, easy to get backwards**: every station
in a real array observes the literal same wavefront from the same
source, just arriving at a different time because of geometry — that's
the entire physical basis of interferometry. So the wideband pulse
train's random carrier is generated from a single fixed seed shared by
every station simulating this pulsar, never from `station.station_id`.
Receiver noise is the opposite: independently seeded per station,
because each station's receiver is a physically separate noise source.

**Resource cost, benchmarked on the EPYC target hardware, 448 channels,
DM=2, 100ms period, 5ms FWHM:**
- One-time build cost (wideband generate + FFT dispersion + channelize)
  is ALL plain numpy as of this session — `generate_wideband_pulse_train`
  (the wideband sky-carrier step) was custom numba until this session
  (the same splitmix64+Box-Muller kernel the noise tile bank used to use
  — see the Noise section above for why that constraint no longer
  applies once generation is one-time, not per-tick); replaced with a
  fully vectorized numpy version, measured FASTER at every period length
  tried (e.g. 6.2s vs. numba's 9.6s at a 1s period, 448 channels) — no
  tradeoff to make, unlike the per-tick kernels below. FFT
  dispersion+channelize was already plain numpy before this session.
  Thread count isn't the lever for any of this — period length is, since
  it scales with the wideband FFT length (`num_channels ×
  n_period_samples`): 0.48s at a 10ms period, 4.4s at 100ms, **46.2s at
  1000ms**.
- **The 1000ms-period figure ALREADY EXCEEDS this project's one-time
  construction budget (target 10s, hard limit 30s)** — a pre-existing
  cost, newly relevant now that a hard budget exists, and NOT something
  the numba→numpy replacement above fixes on its own. Isolating the
  wideband-generate substep alone confirms it's cheap either way (numpy:
  6.2s of the 46.2s total, numba: 9.6s) — the dominant cost is the
  FFT/dispersion step itself.
- **`scipy.fft` (with `workers=`), tried this session, adopted where it
  helps — `_channelize_once`'s batched small FFTs are the clear win
  (~4-5x faster, e.g. 0.14s→0.03s at a 100ms period: embarrassingly
  parallel across independent rows), the two big 1D FFT/IFFT calls in
  `build_pulsar_template` only ~15-20% (bandwidth-bound at these sizes —
  more workers past 8 gave no further benefit, matching this project's
  8-core/pod target). `scipy` is back in `pyproject.toml` for this
  (it was dropped earlier this session when the legacy wideband path —
  its only other user — was deleted; this is a new, independently
  justified reason to bring it back).**
- **The bigger, previously-undocumented finding: FFT SIZE matters far
  more than the numba/numpy/scipy choice.** `n_wide` (the wideband array
  length dispersion FFTs over) is `num_channels × round(period_s ×
  channel_width_hz)` — for most periods this factors into small primes
  (e.g. 100/200/300ms all reduce to `2^a × 5^7 × 7`, largest prime factor
  7) and construction time scales cleanly, near-linearly, with period.
  But some periods land on a size with a large prime factor — e.g. 50ms
  gives `n_wide` with a largest prime factor of **19531** — and measured
  **3-6x slower than neighboring, better-factored periods**, confirmed by
  actually factoring `n_wide` (not guessed): `2^7 × 7 × 1` size at 10ms
  (fine) vs. that one large prime at 50ms (bad), verified via direct
  timing of every construction substep (gen/fft/H-transfer-function/
  multiply/ifft/channelize) with a proper clock-ramp burn-in first (this
  hardware needs one — see CLAUDE.local.md — an earlier pass without it
  showed the same anomaly, ruling that out as the cause). `scipy.fft.
  next_fast_len()` can find a nearby fast size (17,499,776 → 17,500,000
  for the 50ms case, +224 samples) — a genuine, well-characterized
  candidate fix, **not implemented**: it isn't a drop-in swap, since (a)
  `next_fast_len`'s result isn't guaranteed to still be an exact multiple
  of `num_channels`, which `_channelize_once`'s reshape requires (it
  wasn't, for the 50ms case), and (b) padding the wideband array changes
  what "one period" means to the FFT's implicit circular convolution,
  which needs re-validating against the coherence/channel-mapping checks
  before trusting it, not just assuming it's still correct. This is the
  "needs real design work, not a one-line change" boundary — left for a
  future session if multi-second pulsar periods become a real
  requirement.
- **Approximate period limits with scipy.fft as it stands (no
  size-padding), 448 channels**: for a WELL-FACTORED period, roughly
  **~300ms clears the 10s target**, **~900ms (0.9s) clears the 30s hard
  limit** (near-linear fit through 100/200/300ms measurements: 3.21s/
  6.53s/9.84s). For a period that happens to hit a poorly-factored size
  (the 50ms case measured ~3x its linear-trend prediction), the SAME
  nominal period could cost up to ~3x more — so to stay under budget
  **regardless of which period a test picks**, without validating
  size-padding, treat the safe limits as roughly a third of those:
  **~100ms for the 10s target, ~300ms for the 30s hard limit**. Whether a
  given period is "well-factored" is checkable cheaply in advance via
  `scipy.fft.next_fast_len(n_wide) == n_wide` (exact match = already
  fast).
- **Steady-state per-tick cost clears the ~8-core target with room to
  spare**: 1.36ms/tick (51.8% of the 2.621ms budget) at 8 threads; even
  4 threads clears budget (1.93ms, 73.7%). This isolated the pulsar cost
  alone (no `noise_cfg`) — see Benchmarking below for the combined
  tone+tiled-noise+pulsar result, which is what the converged
  `DirectSynthesisStreamer` actually runs.

**Still not modeled**: pulse-to-pulse jitter, scintillation, nulling,
profile evolution with frequency, or realistic flux/SNR calibration
against the receiver noise floor — the last of these matters specifically
if the goal is testing whether PSS/PST can actually detect the injected
pulsar as a candidate (not just exercising delay-tracking), since
detection significance depends on signal-to-noise, not just having the
right shape.

### Per-source delay (each tone/pulsar gets its own DelayPolynomial, required)

Every source used to share ONE station-level `DelayPolynomial`
(`DirectSynthesisStreamer._current_poly`, pulled from a CBF client and
refreshed on expiry) — meaning two tones or two pulsars in the same
streamer were implicitly the same point in the sky. Wrong the moment a
test wants two sources at genuinely different directions in the same
station. Fixed by giving every tone/pulsar its own `common.DelayFeed`
instead: this needed no changes to the synthesis kernels at all —
`synth_tone_channel` and `add_pulsar_tick` already took
`delay_coeffs`/`poly_t_rel_start`/`ypol_offset_ns` as plain arguments,
not something baked into the streamer, so per-source delay was a
data-plumbing change, not a new numerical code path (confirmed by direct
measurement: alternating between two different `delay_coeffs` arrays
across calls costs the same as reusing one shared array, within noise).

**`delay_feed` is REQUIRED on every tone/pulsed `source_cfg` — there is
NO default/fallback delay.** `DirectSynthesisStreamer.__init__` raises
`ValueError` immediately if one is missing. Deliberate, not an oversight:
a source with no real delay path would silently apply zero delay, which
produces content that's trivially "perfectly aligned" — precisely the
kind of thing that could mask a real CBF delay-tracking bug instead of
exercising it, given this simulator's whole reason for existing is
generating true delay independently of CBF. An earlier version of this
design had a `PolledDelayFeed` fallback for convenience; removed once it
became clear a silent implicit zero-delay default fights the simulator's
own purpose more than it helps.

**`common.DelayFeed`** — `update(poly)` is called from whatever thread
learns of a new polynomial (a Tango event callback in production, direct
calls in tests); `get(t)` is called from the generation thread. A plain
reference swap is safe across threads under the GIL since no field of the
swapped-in `DelayPolynomial` is ever mutated in place. Two deliberate
behaviours here too: no polynomial received yet → zero delay, warned once
(not blocking scan start on an external device being up — this is a
*startup-ordering* gap, not the same as a source having no delay path
configured at all, which is rejected outright as above); polynomial
expired with no replacement received → keep applying it as-is, warned
once per staleness episode. **Recovering from a stalled upstream
publisher is explicitly NOT this simulator's job** — it applies whatever
delay it was actually given and logs when that delay is known to be
stale, so the discrepancy is visible to whoever is debugging a test
failure (the same "surface it, don't paper over it" principle as the
Observability section below).

`DirectSynthesisStreamer` caches each source's coefficients as a float64
array keyed by poly *identity*, not recomputed every tick — same
per-tick-allocation discipline as bug #13 below, just applied per source
instead of once station-wide.

**`simulator.py` wiring — REPLACED this session, see "Tango device now
drives the Go gRPC simulator" below for the full picture.** In brief:
`subarray_id`/`beam_id`/`source_cfgs` still arrive dynamically as fields
of the `StartScan` JSON argument (alongside
`obs_time_epoch_s`/`scan_duration_s`/`scan_id`) — unchanged in spirit, a
station can still be reassigned between subarrays/beams across scans
without a pod restart. What changed is everything downstream: EVERY
`source_cfgs` entry still MUST include a `delay_attr_uri` (there is no
default delay), and `StartScan` still opens an `AttributeProxy` per
named attribute and subscribes to its `CHANGE_EVENT`s — but a pushed
update is now forwarded over gRPC (`PushDelayUpdate`) to a separate Go
process instead of feeding a local `DelayFeed` consumed by a locally
constructed streamer, since there is no local streamer anymore.
`dest_ip`/`dest_port` are gone from this device entirely — the Go
process owns the CBF SPEAD/UDP destination now, via its own CLI flags —
and `station_id`/`substation_id` remain, plus a new `grpc_target` device
property naming the Go process's `host:port`. The UNVERIFIED wire-format
caveat is unchanged and still applies: the exact attribute payload shape
(`common.parse_delay_polynomial_from_attr_value` assumes a JSON
string/mapping matching `DelayPolynomial`'s fields) and whether
`AttributeProxy` delivers an immediate `CHANGE_EVENT` with the attribute's
current value on subscribe (vs. only on the next actual change) both
depend on how the real delay-poly emulator is configured — confirm
against it once available, not just against this assumption.

### Tango device now drives the Go gRPC simulator, not a local streamer (new this session)

`simulator.py` used to construct a `DirectSynthesisStreamer` and
`ScanRunner` directly and run the whole producer/sender pipeline in this
process (SPEAD/UDP included). It now does none of that: `StartScan`/
`StopScan`/attribute reads instead call a separate Go process
(`cmd/simulator` at the repo root, `internal/server.Server` — see
`api/simulator.proto`) over gRPC, using generated stubs at
`tango/src/ska_low_station_beam_simulator/simulatorpb/`
(`simulator_pb2.py`/`simulator_pb2_grpc.py`, regenerated from
`api/simulator.proto` — see "Setup" below for the exact command). This
is exactly the split `api/simulator.proto`'s own doc comment already
described before this session actually wired it up: a Tango device
server owns Tango (device properties, `AttributeProxy` subscriptions to
CBF's delay-poly emulator), the Go process owns signal generation and
SPEAD/UDP sending, and has no Tango access of its own.

**Prototype scope, inherited from the Go side, not yet extended**: the
Go backend (see `api/simulator.proto`'s own doc comment) implements
tone + noise only — no pulsar. `simulator.build_tone_source_request`
(replacing the old local-generation `build_source_cfg`) raises
`ValueError` immediately for any `source_cfgs` entry with
`kind != "tone"`, rather than silently dropping it or attempting some
local fallback — there is no local generation path left in this device
at all, so "fall back to Python" isn't an option even if it were
desirable. Same "fail loud instead of silently degrading" principle
this codebase applies to missing delay feeds elsewhere.
`direct_synthesis.py`'s own pulsar support is completely untouched and
still directly exercised by `benchmark_direct_synthesis.py`,
`generate_test_pcap.py`, and `tests/test_direct_synthesis.py` — only
this Tango-facing device no longer drives it for a real scan.

**`source_id` reuses `delay_attr_uri` verbatim, not a new JSON field.**
The gRPC `PushDelayUpdate` RPC routes an update to the right source by
`source_id`, a concept the old local-generation flow never needed (each
`source_cfgs` entry just built its own `DelayFeed` object directly).
Since every source already requires a unique `delay_attr_uri` (there's
no default delay — see "Per-source delay" above), `simulator.py` reuses
that URI string as the source's `source_id` on both ends of the wire —
`build_tone_source_request` sets `ToneSourceConfig.source_id =
spec["delay_attr_uri"]`, and `_make_delay_feed`'s `CHANGE_EVENT`
callback keys its `PushDelayUpdateRequest` the same way — instead of
inventing a new required key in the `source_cfgs` JSON schema. The Go
server independently rejects a duplicate `source_id` within one scan
(`StartScan`'s own validation), which doubles as a duplicate-
`delay_attr_uri` check on this side.

**`StartScan`'s check-then-act race, same shape as before, not fully
closed**: before touching any subscription, `StartScan` calls
`GetStatus` and raises if a scan is already reported running — this runs
BEFORE tearing down the previous scan's delay subscriptions,
specifically so a rejected `StartScan` never destroys the actually-
running scan's ability to receive delay updates. A concurrent second
`StartScan` call between that check and the real gRPC `StartScan` call
could still slip through (the check and the act aren't atomic); the real
backstop is the Go server's own `FailedPrecondition` rejection in
`StartScan` (`internal/server/grpc_server.go`) — the same non-atomicity
the old local-generation flow already tolerated (a
`self._scan_runner.thread.is_alive()` check with no lock around it
either).

**New Tango attributes, next to `queue_depth`**: `StatusResponse` (the
`GetStatus` RPC's response message) gained `drift_seconds` and
`tick_number` this session, both sourced from `ScanRunner`
(`internal/common/scan_runner.go`) on the Go side — `DriftSeconds()`/
`TickNumber()` read two atomics (`driftBits`, via
`math.Float64bits`/`Float64frombits` since this Go version has no atomic
float64 type; `tick`, an `atomic.Int64`) updated once per tick from the
scan loop, both 0 before the first tick or once no scan is running.
`drift_seconds` is wall-clock time minus that tick's target time at the
most recently produced tick — positive means the producer is running
behind its real-time pacing schedule; this is the same value the scan
loop already logged past `OverrunTolerance` (see "Observability" below),
now queryable directly rather than only visible in a log line, and now
computed uniformly on every tick (previously only computed in the
"already behind" branch) rather than only when the loop didn't need to
wait. `simulator.py` exposes both as read-only Tango attributes
(`drift_seconds`: float, `tick_number`: int) alongside `queue_depth`,
each issuing its own `GetStatus` RPC on read — three independent
round-trips per polling cycle rather than one cached call, deliberately
kept simple for a first version; revisit if that's ever shown to matter
(each call is a small, sub-millisecond local RPC).

**Regenerating the Python stubs** (after editing `api/simulator.proto`):
see "Setup" below. The generated `simulator_pb2_grpc.py` needs one
hand-patch after every regeneration — `grpc_tools.protoc`'s Python
plugin always emits `import simulator_pb2 as simulator__pb2` (a bare,
top-level import), which doesn't resolve from inside the
`ska_low_station_beam_simulator.simulatorpb` package; a `sed` rewrite to
`from . import simulator_pb2 as simulator__pb2` is required every time
(the Go side has no equivalent issue — `protoc-gen-go` emits
package-qualified imports directly).

**Verified against a live Go process, not just unit-level mocks**: this
session drove a locally-built `cmd/simulator` binary directly with the
generated Python stub (`StartScan`, `PushDelayUpdate`, `GetStatus`
mid-scan, `StopScan`) — confirmed `drift_seconds`/`tick_number` populate
with real, sane values mid-scan (e.g. `drift_seconds≈0.00018`,
`tick_number` advancing) and both go back to their zero defaults once
`StopScan` completes.

### `source_cfgs`/`noise_cfg` are typed dataclasses, not dicts (new this session)

`DirectSynthesisStreamer.__init__` used to take `source_cfgs: list[dict]`
and `noise_cfg: dict | None` — a `cfg["kind"]` string plus ad hoc keys
validated by hand at construction time (kind in `("tone", "pulsed")`,
`delay_feed` present and a `DelayFeed`, pulsar's `pulsar_name` XOR
`period_s`/`width_s`/`dm_pc_cm3`). Replaced with explicit dataclasses in
`direct_synthesis.py`: `ToneSourceConfig`, `PulsarByNameConfig`,
`PulsarByParamsConfig` (`SourceConfig`/`PulsedSourceConfig` are the
corresponding union aliases), and `NoiseConfig`.

**The two pulsar variants are deliberately SEPARATE types, not one
dataclass with optional fields** — this turns the old "`pulsar_name` XOR
`period_s`/`width_s`/`dm_pc_cm3`" runtime dict-shape check into a
structural property of which type a caller constructs, so
`DirectSynthesisStreamer.__init__` no longer needs that validation at
all. Same for `delay_feed`: it's a required field with no default on
every config type, so a source with no real delay path fails at the
dataclass's own construction (a plain `TypeError` from Python itself, not
custom validation code) — still satisfies this simulator's "no
default/fallback delay" rule (see "Per-source delay" above), just
enforced one level earlier than before.

**The mutual-exclusion/kind-dispatch validation didn't disappear — it
moved to `simulator.build_source_cfg`**, a standalone (Tango-free)
function that turns one raw JSON `source_cfgs` entry plus its
already-resolved `DelayFeed` into the right dataclass. This is a
deliberate "validate at the boundary" move: `StartScan`'s JSON argument
is the actual untyped-data entry point into this system (per-source
`kind`/`pulsar_name`/`period_s` etc. arrive as strings/numbers off the
wire), whereas `DirectSynthesisStreamer`'s own callers (tests,
`benchmark_direct_synthesis.py`) now pass already-typed config objects
directly and get that checked by Python's own type system rather than by
runtime dict-shape assertions. `build_source_cfg` is unit-tested in
`tango/tests/test_simulator_source_cfg.py` (kind dispatch, `delay_attr_uri`
correctly not forwarded as a dataclass field, both/neither pulsar-config
rejection) without standing up a Tango device — this codebase otherwise
doesn't unit test that layer at all (see Setup).

### Pulsar catalog — pre-generated templates, loaded by name (new this session)

`build_pulsar_template`'s one-time construction cost (see Benchmarking's
"CORRECTED pulsar construction budget" — now tighter than ever, since
the oversampling correction made essentially every period factor
poorly) doesn't have to be paid at every scan start. `pulsar_catalog.py`
adds a small catalog of pre-generated pulsar templates
(`CATALOG_ENTRIES`), built once offline by `generate_pulsar_catalog.py`
and loaded by name at `DirectSynthesisStreamer` construction instead —
essentially free startup cost (an `np.load` plus a slice), no FFT.

**Both configuration styles are supported side by side, not one instead
of the other** — a `"pulsed"` `source_cfg` gives either `pulsar_name`
(load a catalog entry, `PulsarByNameConfig`) or
`period_s`/`width_s`/`dm_pc_cm3` (build a custom template at
construction, `PulsarByParamsConfig`) — see "`source_cfgs`/`noise_cfg`
are typed dataclasses, not dicts" above for why these are two separate
types rather than one mutual-exclusion check. This was a deliberate
design choice, not the obvious default: it means clients can pick fast,
fixed-parameter setup for routine tests while still keeping the ability
to dial in an arbitrary period/DM for a test that specifically needs
one, at the cost of the slower construction path.

**`pulsar_catalog.py`'s own entries are typed too, not raw dicts**:
`CATALOG_ENTRIES` is a list of `CatalogEntrySpec`
(`name`/`period_s`/`width_s`/`dm_pc_cm3`/`sky_seed`) —
`generate_pulsar_catalog.py` reads these as attributes, not `entry["..."]`
lookups. `load_pulsar_from_catalog` returns a `LoadedPulsarTemplate`
(`template`/`period_s`/`n_period_samples`/`width_s`/`dm_pc_cm3`/
`sky_seed`) instead of a dict, which is what `DirectSynthesisStreamer`'s
`PulsarByNameConfig` branch and `tango/tests/test_pulsar_catalog.py` both
consume. The on-disk `catalog.json` record itself (`save_pulsar_to_catalog`'s
per-name dict, including `npy_filename`/`num_channels`/`base_freq_hz`/
`channel_width_hz`/`channel_output_rate`) deliberately stays a plain
dict — it's a JSON serialization boundary, the same reasoning that keeps
`StartScan`'s raw JSON argument a dict before `simulator.build_source_cfg`
turns it into a typed config.

**Every catalog entry is generated at the FULL band width**
(`common.MAX_NUM_CHANNELS` = 384 channels, starting at
`common.BASE_FREQ_HZ` = channel 64) — a station simulating a narrower
sub-band just slices the columns it needs out of the same array
(`pulsar_catalog.load_pulsar_from_catalog`'s `station_num_channels`/
`station_base_freq_hz` arguments handle this, validated for both
channel-grid alignment and range). This means ONE `.npy` per named
pulsar serves every valid station configuration, not one per
`(num_channels, first_channel_id)` combination — consistent with this
module's existing principle that a pulsar's "sky carrier" is shared
across every station observing it.

**Stored as complex64 on disk** (halves size vs. the complex128 used
internally — no meaningful fidelity loss for this purpose, since content
is quantized to int8 well downstream anyway), upcast back to complex128
immediately on load so `add_pulsar_tick`'s numba kernel sees the exact
same dtype regardless of which path built the template.
`catalog.json` records the exact `channel_width_hz`/
`channel_output_rate`/`num_channels`/`base_freq_hz` each entry was
generated under, checked against the CURRENT constants on load — a
catalog baked under since-corrected constants (this project has already
been burned once by a wrong `channel_output_rate` assumption, see below)
fails loudly instead of silently misapplying stale data.

**A real, previously-unmeasured cost, discovered generating the actual
default catalog**: `CATALOG_ENTRIES`'s three illustrative entries
(10ms/89.3ms/300ms periods) came to **28MB / 254MB / 853MB respectively
— ~1.1GB total** for just three examples. This scales roughly linearly
with period (longer period → more samples per period → bigger array at
fixed 384-channel width), so a real deployment's catalog size depends
entirely on which periods it actually needs. **Deliberately NOT
committed to git** (`pulsar_catalog_data/` is gitignored) — binary blobs
this large would permanently bloat repository history for every future
clone, with no way to shrink it back down short of a history rewrite.
The recommended flow for actually getting these into an OCI image is to
run `generate_pulsar_catalog.py` as an image BUILD step (after
installing the package, before the image is finalized), not to bundle
pre-generated `.npy` files into a wheel/sdist — this keeps large binaries
out of both git and any package index the project might publish to.
This project has no Dockerfile yet; wire this in when one exists.

## SPS-CBF ICD channelization — corrected this session

Several channelization assumptions this codebase had been running on
turned out to be wrong once checked against the real ICD text (not a
screenshot, not carried forward from an earlier session's guess). All
three corrections below are now reflected in `common.py`'s constants,
`direct_synthesis.py`'s validation, and `simulator.py`'s device
properties.

1. **Maximum channel count is 384 (300MHz), not 448 (350MHz).** The ICD:
   "The bandwidth is channelized as 384 equispaced coarse channels" and
   "the number of frequency channels assigned to the beams is
   configurable from 8 to 384 in steps of 8." 448 channels was used
   throughout earlier sessions as "the full SKA-Low band" and benchmarked
   extensively at that size — **that config was never valid on real
   hardware.** `common.MAX_NUM_CHANNELS = 384`,
   `common.MIN_NUM_CHANNELS = 8`, `common.NUM_CHANNELS_STEP = 8`;
   `DirectSynthesisStreamer.__init__` now validates `num_channels`
   against these and raises `ValueError` otherwise — this codebase can no
   longer be silently pointed at an invalid channel count. Every "448
   channels" figure in this file from before this correction is left as
   historical record of what was actually measured then (not
   retroactively rewritten) but should not be read as a currently-valid
   configuration; see the Benchmarking section for corrected numbers at
   the real 384-channel maximum.

2. **The lowest channel is global ID 64, not 65.** The ICD: "the lowest
   frequency channel is channel 64, centre frequency 50MHz." 64 ×
   `CHANNEL_WIDTH_HZ` (781,250 Hz) = 50,000,000 Hz exactly, confirming
   `BASE_FREQ_HZ` is meant as a channel **centre** frequency (matching
   how it's used everywhere in this codebase — e.g.
   `synth_tone_channel`'s `channel_center = base_freq_hz +
   channel_idx*channel_width_hz` — not a band edge). Previous session's
   channel 65/50.78125MHz was off by one channel.
   `common.BASE_FREQ_HZ = 50.0e6`,
   `StationConfig.first_channel_id` default changed 65 → 64 (also
   `simulator.py`'s matching device_property default). The band's actual
   lower **edge** (distinct from channel 64's centre) is
   `50e6 - CHANNEL_WIDTH_HZ/2` = 49,609,375 Hz, matching the ICD's stated
   "lower edge of this channel is 49.61MHz" — nothing in this codebase
   needs that edge value directly, since every per-channel frequency
   calculation here is already expressed in channel centres, but it's
   recorded here in case a future edge-relative calculation needs it.

3. **The real per-channel sample rate is oversampled by 32/27, not
   critically sampled.** The ICD: "Each channel of data from SPS to Low
   CBF has a sampling period of 1080 ns (1.25 ns per sample (ADC sample
   rate) x 1024 samples x 27/32, wherein 32/27 is the oversampling factor
   of the filterbank)." This codebase had been assuming
   `channel_output_rate = CHANNEL_WIDTH_HZ` (i.e. a 1280ns sample period
   — what a CRITICALLY sampled channelizer would give), missing the
   oversampling factor entirely. `CHANNEL_WIDTH_HZ` itself
   (781.25kHz) remains correct as-is — it's the FREQUENCY-DOMAIN channel
   *spacing*, unaffected by oversampling (channel centres are still
   exactly `CHANNEL_WIDTH_HZ` apart) — what was wrong is treating that
   same number as the TIME-DOMAIN per-channel sample *rate* too.
   `common.CHANNEL_OUTPUT_RATE_HZ = CHANNEL_WIDTH_HZ * 32/27` ≈
   925,925.93 Hz (1080ns period, confirmed exactly:
   `1/CHANNEL_OUTPUT_RATE_HZ = 1.08e-6 s`). **This is a real, numerically
   significant fix, not just a naming correction**: `BLOCK_DURATION_S`
   (the per-tick real-time budget every benchmark number in this file is
   measured against) is `HEAP_LEN / CHANNEL_OUTPUT_RATE_HZ` = **2.21184ms
   — about 15.6% tighter than the 2.621ms this codebase had been
   budgeting against.** `DirectSynthesisStreamer.channel_output_rate`
   and `ScanRunner.channel_output_rate` (`common.py`) both now use the
   corrected rate; every per-tick timing kernel derives its sample
   spacing from `channel_output_rate`, not `CHANNEL_WIDTH_HZ` directly,
   so this one fix propagates correctly through tone/pulsar/noise timing
   without further per-kernel changes. Confirmed compatible with the
   ICD's separate packet-size constraint ("the number of samples per
   packet shall be a multiple of the numerator of the oversampling
   ratio," i.e. a multiple of 32): `HEAP_LEN` (2048) / 32 = 64 exactly.

**A related ICD passage — REASONED THROUGH, not independently verified
against real hardware**: "The oversampled filterbank data shall be
derotated as well as ensuring the first sample of every SPS SPEAD packet
has zero phase." Worked through in detail in `direct_synthesis.py`'s
module docstring (search "OVERSAMPLING AND PER-PACKET PHASE"); short
version: this describes a raw-PFB-output correction step this codebase
never needs, because `synth_tone_channel`/`add_pulsar_tick` synthesize
the already-clean, already-derotated baseband result directly rather
than generating the raw oversampled artifact and correcting it
afterward. The "zero phase at packet start" convention is read as a
NORMALIZATION reference point (for a hypothetical zero-residual-
frequency signal), which this module's phase formula already satisfies
trivially, not as a literal per-heap phase reset for every source
regardless of residual frequency — the latter reading would destroy the
cross-heap phase continuity CBF's own delay-tracking and coherent
beamforming need to recover, which seems physically implausible as the
actual intent. Flagged honestly as reasoned-not-confirmed: revisit if a
delay-tracking test ever shows a phase discontinuity at heap boundaries.

## SPS-CBF ICD heap structure — RESOLVED this session, was the top item

One heap = ONE CHANNEL = 2048 consecutive time-domain samples, both
polarisations interleaved per sample (Vreal, Vimag, Hreal, Himag, each
int8). `pkt_len = 0x2000` (8192 bytes) confirms this: 2048 × 4 bytes.

**The bit-field layout for `channel_info`/`antenna_info` used to be read
off a SCREENSHOT of the ICD diagram, flagged as this project's top
unverified risk.** You've since confirmed the exact item layout against
the real ICD directly — six items total, each an IMMEDIATE SPEAD-64-48
item pointer (48-bit value field):

| item ID | bit layout |
|---|---|
| `0x0001` | 8 bits reserved \| 40 bits `heap_counter` |
| `0x0004` | 48 bits `packet_payload_length` (fixed: `0x2000`) |
| `0x3010` | 48 bits `scan_id` |
| `0x3000` | 16 bits reserved \| 16 bits `beam_id` \| 16 bits `frequency_id` |
| `0x3001` | 8 bits `substation_id` \| 8 bits `subarray_id` \| 16 bits `station_id` \| 16 bits reserved |
| `0x3300` | 48 bits `payload_offset` (fixed: `0x0` — heaps are always exactly one packet) |

...followed immediately by the 8192-byte interleaved V/H I/Q payload —
**no 7th "payload" item pointer**; CBF firmware reads the payload at a
fixed byte offset (56 bytes in: 8-byte SPEAD header + 6×8-byte item
pointers) rather than doing a generic SPEAD parse. `pack_channel_info`/
`pack_antenna_info`'s bit widths turned out to match this exactly, bit
for bit, once confirmed — no change needed there, just the "VERIFY"
caveats removed.

**A real architectural problem surfaced trying to send this via spead2,
and it's why `SpsPacketizer` (now in `spead.py`) no longer uses spead2 at all.**
Testing the SPEAD encoding path end-to-end (see the pcap section below)
showed every heap coming out with 8 items instead of the ICD's 6.
Checked directly against spead2==4.4.1's C++ source (`send_packet.cpp::
packet_generator::next_packet`), not just its Python API: spead2's
packet encoder unconditionally writes 4 reserved item pointers
(`HEAP_CNT`, `HEAP_LENGTH`, `PAYLOAD_OFFSET`, `PAYLOAD_LENGTH`) at the
start of every packet it ever emits, with **no flag, `StreamConfig`
option, or `Heap` method to suppress any of them** — `Heap.
repeat_pointers`, the only heap-level toggle in the public API, controls
something unrelated (whether item pointers repeat across a fragmented
multi-packet heap's packets, not whether this quartet appears at all).
Since CBF's ICD heap has only 6 items total — fewer than spead2's own
mandatory minimum of 4 reserved + any real items — **spead2 cannot
produce a compliant packet no matter how it's configured.** A firmware
receiver that addresses payload bytes directly at fixed offsets (rather
than doing a generic SPEAD parse) has no way to skip items it doesn't
expect, so this isn't cosmetic.

**Fixed by replacing spead2 with a hand-rolled SPEAD-64-48 encoder**
(`common.py`'s `_spead_header_bytes`/`_spead_item_pointer`, used by
`SpsPacketizer.encode_channel_heap`) — scoped to exactly the 6 items
above and nothing else. The wire format itself (8-byte header: magic
`0x53`/version `4`, item-ID field width, heap-address field width, item
count; then N 8-byte item pointers; then payload) matches real SPEAD,
confirmed against spead2's own source — what differs is only which
items get written. `spead2` has been dropped from `pyproject.toml`
entirely; nothing in this codebase uses it anymore (`SpsPacketizer` now
sends over a plain UDP `socket`, with dependency injection for testing —
see its docstring). Verified two ways: an item-by-item pytest check that
parses the encoded bytes back into (ID, value) pairs independent of the
encoder's own logic (`tango/tests/test_spead_packetizer.py`), and a manual
byte-by-byte decode of a real generated pcap's hex dump against the
table above (every field matched, including `channel_info`'s packed
`beam_id`/`frequency_id` and `antenna_info`'s packed station fields).

**CONFIRMED, then CORRECTED again the following session**: `BASE_FREQ_HZ`
was first confirmed as 50.78125MHz (coarse channel 65), replacing an
earlier `0.0` placeholder — this fixed a real, previously-latent gap:
`simulator.py`'s `StartScan` never overrode `base_freq_hz`, so any
pulsed source configured through it would have hit
`DirectSynthesisStreamer`'s `base_freq_hz > 0` check and failed outright
while `BASE_FREQ_HZ` was still `0.0`. `direct_synthesis.py`'s
`DEFAULT_PULSAR_BASE_FREQ_HZ` (an illustrative 50MHz stand-in) was
deleted at the same time, now that `BASE_FREQ_HZ` itself was real. **The
65/50.78125MHz value itself was then found to be off by one channel** —
see "SPS-CBF ICD channelization" above: the ICD names channel **64**
(50.0MHz exactly) as the lowest, not 65. `common.BASE_FREQ_HZ` is now
`50.0e6` and `StationConfig.first_channel_id`'s default is `64`.

## Known bugs — fixed (don't reintroduce), and their current status

Bugs #4, #5, #6, #8, #9, #10, #11, and #12 below were specific to the
now-**deleted** `wideband_streamer.py` — kept for the lessons (some
generalize, e.g. #6's per-task-overhead diagnosis and #11's
repeatability-check lesson), not because the buggy code still exists to
reintroduce a regression into.

1. **V/H noise sharing**: one `NoiseSource` reused for both pols gave
   numerically identical noise (it's a deterministic pure function of
   inputs). Fixed: separate seeds per pol.
2. **Absolute-epoch-time precision collapse (delay polynomial)**:
   evaluating a 5th-order polynomial against raw Unix-epoch-scale `t`
   (~1.8×10⁹) blows up float64 precision. Fixed: evaluate relative to
   `start_validity_sec` (`common.DelayPolynomial.eval_delay_seconds`).
3. **Same bug, tone phase**: absolute `t` fed directly into phase
   computation collapsed float64 precision (~0.1-0.5 rad error). Fixed:
   all generation takes `t_rel = t - obs_time_ref`, bounded by scan
   duration. This is why every kernel in this codebase takes a
   small-magnitude relative time, never raw epoch time — check any new
   kernel against this before assuming it's a minor style choice.
4. *(deleted-file history)* **Ring buffer `np.roll` on the whole buffer
   every tick** — O(capacity) regardless of new-sample count, in the
   legacy wideband path's `RingBuffer`. Fixed there with a real O(1)
   circular buffer; moot now that the whole ring-buffer/coarse-fine-delay
   approach is gone (direct synthesis never buffers).
5. *(deleted-file history)* **Channelizer looping `np.fft.fft()` per
   output row** in the legacy wideband path — thousands of individual FFT
   calls/tick. Fixed there with `sliding_window_view` + one batched FFT
   call; moot now that per-tick FFT channelization doesn't happen
   anywhere in this codebase (pulsar generation's one FFT call is a
   one-time, offline construction step, not per-tick).
6. **`ThreadPoolExecutor`-chunked generation plateaus at ~4-8 workers**,
   confirmed on 3 different CPU architectures (Apple M5, Intel Xeon
   Silver 4410T, AMD EPYC 9254/7443) regardless of core count —
   per-task Python/GIL overhead, not compute/bandwidth. Fixed: moved to
   `numba`+`prange`, near-linear scaling well past that plateau. The
   general lesson (per-task Python overhead, not compute, caps naive
   threading) generalizes beyond the wideband path this was found in.
7. **Numba's `nopython` mode doesn't support `numpy.random.Philox`**
   (confirmed by direct test). Replaced with a from-scratch splitmix64
   hash + Box-Muller, used throughout `direct_synthesis.py` (tone, noise,
   and the pulsar's wideband "sky carrier" all share it). Statistically
   correct, deterministic, seekable, but NOT bit-identical to Philox.
   `_splitmix64_hash` (the one piece of this still needed today, for
   per-tick noise-tile-index selection — see the Noise section) was
   deliberately kept as a hand-rolled `@njit` function rather than
   switched to `Generator(Philox(key=seed, counter=tick_index))` once
   Philox's own `counter=` constructor was found to support exactly the
   "pure function of an arbitrary index" property this needs: benchmarked
   at ~12.7us/call for a fresh `Philox`+`Generator` construction per call
   vs. ~0.13us/call for the numba hash (~100x slower — Python/pybind11
   object-construction overhead, not the underlying algorithm). The
   `@njit` itself is also justified independent of that comparison: ~25x
   faster than the equivalent plain-Python/numpy-scalar version (0.13us
   vs. 3.3us/call), confirmed by direct measurement, not assumed from
   "numba is usually faster."
8. *(deleted-file history)* **`fftshift` was a full-array reorder every
   tick for no numerical reason**, in the legacy wideband channelizer —
   replaced there with natural FFT bin order + a fixed permutation
   applied only to small `channel_id` integers. The successor of that
   permutation logic (`_channelize_once`'s `natural_to_external`
   construction in `direct_synthesis.py`) is used only in the pulsar
   template's one-time, offline construction, never per-tick.
9. *(deleted-file history)* **`OMP_NUM_THREADS=1` cap** — originally
   needed in the legacy wideband path to stop numpy/BLAS oversubscribing
   against `ThreadPoolExecutor`-chunked generation; confirmed
   no-longer-necessary-but-harmless once that path moved to numba. Never
   needed in `direct_synthesis.py`, which never used `ThreadPoolExecutor`.
10. *(deleted-file history)* **`numpy.fft.fft` has no multi-threading at
    any array size** — the legacy wideband channelizer switched to
    `scipy.fft.fft` (`workers=` param) to work around this; `scipy` has
    since been dropped from `pyproject.toml` entirely along with that
    file, since nothing else in this codebase used it (the pulsar
    template's one-time FFT step uses plain `np.fft`, where the lack of
    multi-threading doesn't matter — it's a one-time cost, not per-tick).
11. **Benchmark-script bug** (not production code): a sweep loop left a
    tuning parameter at its last-tried value instead of the sweep's best.
    General lesson, still very much alive in this codebase: don't trust
    a single-pass sweep result on this class of hardware without a 5+
    repeat repeatability check (see this file's own Benchmarking section
    for a case where this exact discipline caught a wrong root-cause
    diagnosis).
12. *(deleted-file history)* **Receiver noise incorrectly
    delay-corrected** in the legacy wideband architecture (`shared` and
    `noise` were summed before its delay/channelization pipeline).
    `DirectSynthesisStreamer` was already immune by construction (noise
    never enters a delay pipeline there); the buggy code itself is gone
    now that the legacy path is deleted.
13. **`DirectSynthesisStreamer.generate_next_tick` allocated a fresh
    14.7MB complex128 array (`np.zeros`) plus another fresh array from
    `synth_noise_all_channels` plus a separate full-array `+=`, twice per
    tick (once per pol)** *(`synth_noise_all_channels`/`_into` — the
    numba Box-Muller kernels named here — no longer exist; noise
    generation moved to plain numpy this session, see the Noise section
    and bug #16, but the reused-output-buffer fix this bug describes is
    still exactly how `generate_next_tick` works)* — this allocation/copy
    churn, not Box-Muller math, was the dominant per-tick cost at high
    channel counts (~30ms of a ~33ms tick at 448 channels). Misdiagnosed
    at first as a compute/thread-dispatch problem — see "Target server
    results" below for the full investigation trail. **Fixed**: noise is
    written directly into a persistent, reused per-pol buffer
    (`DirectSynthesisStreamer._get_output_buffer`); tone then adds on top.
    Verified bit-identical to the old path for the same seed/index. Safe
    to mutate in place because `ScanRunner._run` calls `generate_next_tick`
    synchronously and `HeapAccumulator.add` copies the data out
    (`np.concatenate`) before the next tick runs — nothing downstream
    holds a reference across ticks.
14. **Pulsar dispersion used `np.fft.rfft`/`irfft`** (correct for a
    real-ADC-sampled convention like PsrSigSim's) **instead of the full
    complex `fft`/`ifft` this codebase's own channel-frequency convention
    needs** (negative `fftfreq` bins are meaningful, independent
    channels) — silently covered only half the intended band and shifted
    every channel's frequency. Caught by injecting a known tone at a
    known external channel and checking where it actually landed, not by
    reasoning about the convention — see the Pulsed sources section
    above. Fixed by using `fft`/`ifft` throughout; the known-tone check
    is now a permanent regression test in `direct_synthesis.py`'s
    `__main__`.
15. **The pulsar's per-tick geometric-delay phase correction called
    `cos`/`sin` once per (sample, channel) pair** — ~1.8M transcendental
    calls per tick at 448 channels, needing 4x the threads to clear
    budget. Fixed with a phase-accumulator (NCO-style) recurrence: two
    trig calls per sample, then one complex multiply per channel to step
    the rotation forward. Same "per-element transcendental calls are the
    expensive part" lesson as bug #13's noise-allocation fix, applied to
    a different hot spot — check any new per-channel kernel against this
    before assuming a `cos`/`sin` call per channel is free.
16. **`fill_noise_bank`'s first version (this session) returned one
    freshly allocated array per worker thread and `np.concatenate`'d
    them** — peaked at several times the bank's own memory footprint
    (per-worker return arrays + concatenate's source-and-destination all
    alive at once) and OOM-killed the process building a realistically
    sized bank (n_tiles=1024, 448 channels) — on a SHARED node, this took
    other tenants' pods down with it (`oom.group` kill), not just our own
    process. Caught only by actually running it at realistic size under
    `/usr/bin/time -v`, not by reasoning about the code — the same
    "measure, don't assume" lesson as the allocation-overhead
    investigation in bug #13, now applied to memory instead of time.
    Fixed by having each worker write directly into its slice of ONE
    preallocated bank array, one tile at a time (bounding transient
    memory to a handful of tiles regardless of total bank size) — a first
    attempt at this fix tried using `numpy.random.Generator`'s `out=`
    parameter directly against `bank[start:end].real`/`.imag`, which
    fails outright (`out=` requires a C-contiguous target, and `.real`/
    `.imag` views of a complex array are strided) — caught immediately as
    a clear error, not a silent one. Plain assignment (`bank[i].real =
    arr`), unlike `out=`, does accept a strided target, which is what the
    final per-tile version relies on. Verified: peak RSS now matches the
    bank's own size almost exactly, confirmed via a `ulimit -v` safety
    net before trusting it on this shared node again.
17. **`SpsPacketizer.send_channel_heap`'s `heap_counter` formula
    multiplied `unix_to_tai2000_seconds(...)` by `CHANNEL_WIDTH_HZ` (the
    SAMPLE rate) instead of dividing by `BLOCK_DURATION_S` (the correct
    per-HEAP rate)** — inflated the value by `HEAP_LEN` (2048x). For any
    current-era timestamp this overflows the ICD's 40-bit `heap_counter`
    field (the `0x0001` item — top 8 bits reserved, low 40 bits the
    counter; see the ICD section above), meaning **every real
    `send_channel_heap()` call would have failed outright** — this had
    silently never been caught because nothing exercised
    `send_channel_heap()` end-to-end before `generate_test_pcap.py` (see
    "Code layout") tried to actually send a heap through it for the
    first time. Fixed: `heap_counter = unix_to_tai2000_seconds(t) /
    BLOCK_DURATION_S`, which keeps the counter comfortably within 40 bits
    until roughly year 2091 — `encode_channel_heap` now also raises
    `ValueError` outright if a future change ever pushes it out of range
    again, rather than silently truncating. See
    `tango/tests/test_spead_packetizer.py` for the regression coverage
    (present-day and 30-years-out checks, an out-of-range rejection
    check, plus an actual `send_channel_heap()` call against an injected
    fake socket, not just the arithmetic in isolation). (This bug
    predates, and is unrelated to, the separate spead2 item-count problem
    described in the ICD section above — both were found by the same
    "actually exercise this path end-to-end for the first time" session.)
18. **spead2 cannot produce CBF's real heap format at all** — its packet
    encoder unconditionally writes 4 reserved item pointers with no way
    to suppress any of them, but the ICD heap has only 6 items total.
    Every heap sent through spead2 came out with 8 items instead of 6.
    Not a config problem, confirmed by reading spead2==4.4.1's own C++
    source — see the SPS-CBF ICD section above for the full writeup.
    Fixed by replacing `SpsPacketizer`'s spead2 usage entirely with a
    hand-rolled SPEAD-64-48 encoder scoped to exactly the ICD's 6 items;
    `spead2` has been dropped from `pyproject.toml`, nothing in this
    codebase depends on it anymore.
19. **Three channelization assumptions were wrong, all traced to not
    having checked the real ICD text closely enough**: (a) 448 channels
    was benchmarked and described throughout this file as "the full
    band" — the real ICD maximum is 384 (300MHz), configurable 8-384 in
    steps of 8; 448 was never a valid configuration. (b) the lowest
    channel was assumed to be global ID 65 (50.78125MHz) — the ICD names
    channel 64 (50.0MHz exactly) as the lowest. (c) `channel_output_rate`
    was assumed equal to `CHANNEL_WIDTH_HZ` (781.25kHz, a 1280ns sample
    period) — the ICD's filterbank oversamples by 32/27, giving a real
    1080ns sample period (`CHANNEL_OUTPUT_RATE_HZ` ≈ 925,925.93 Hz).
    (c) is the most consequential: it shrinks `BLOCK_DURATION_S` (the
    per-tick real-time budget) from 2.621ms to 2.21184ms — every
    percentage-of-budget figure in this file measured before this
    correction is ~15.6% more optimistic than the real constraint. Fixed:
    `common.MAX_NUM_CHANNELS`/`MIN_NUM_CHANNELS`/`NUM_CHANNELS_STEP`
    (with `DirectSynthesisStreamer` validating against them),
    `BASE_FREQ_HZ`/`StationConfig.first_channel_id` corrected to channel
    64, `common.CHANNEL_OUTPUT_RATE_HZ` added and used everywhere
    `channel_output_rate` is derived (`DirectSynthesisStreamer`,
    `ScanRunner`) — see "SPS-CBF ICD channelization" above for the full
    writeup, including a related derotation/phase-convention passage in
    the ICD that was reasoned through rather than blindly implemented.

## Benchmarking

**Per-tick budget is fixed regardless of channel count**:
`HEAP_LEN / CHANNEL_WIDTH_HZ = 2048 / 781250 Hz ≈ 2.621ms`
(`common.BLOCK_DURATION_S`). Work scales with channel count; the budget
doesn't. "Comfortable" per your stated bar is ~80% of budget; 95-110% is
not good enough as a baseline.

### Historical laptop results (early session, tone+noise only, pre-convergence)

Machine: 10-core Apple Silicon (arm64), single laptop-class machine —
**not yet run on real target server/Kubernetes hardware, which is the
reason for this SSH move.**

- **96 channels (current target): ~1.3-1.4ms/tick at numba_threads=8-10
  — ~50-55% of budget.** Comfortably within the 80% comfort bar.
  Repeatability-checked (5 repeats, stdev ~0.02ms).
- **448 channels (full band): ~5.2-5.4ms/tick even fully parallelized —
  still ~200-210% of budget.** The 96→448 scaling ratio is ~3.7-4.4x
  (close to linear, matching the algorithmic prediction), a real
  improvement over the legacy path's worse-than-linear ~4.67x+ — but it
  does NOT close the gap on this hardware. **At 448 channels, per-channel
  noise synthesis (not tone, which is O(1) and negligible at ~0.014ms) is
  now the dominant cost, and it's inherently O(channels).** This is the
  real remaining bottleneck for full-band viability — not something the
  direct-synthesis architecture change alone solves.
- **Next things to try, in rough priority order**: (a) run this same
  benchmark on the actual target server hardware — more cores may close
  some or all of the gap, especially since noise generation parallelizes
  near-linearly; (b) if still over budget there, the noise kernel itself
  is the next lever (current one does log/sqrt/sin/cos per sample via
  Box-Muller — a cheaper RNG, e.g. Ziggurat, or a real-only noise
  approximation, are candidates, not yet tried).

*(Historical, file since deleted)* the legacy wideband `StationStreamer`
benchmarked, same machine, 96 channels: ~2.4-2.6ms/tick at
numba_threads=8-10 with FFT_WORKERS=2-8 — right around budget, not
comfortably under it. Consistent with this path's earlier EPYC results
below; was never re-benchmarked at 448 channels given its known
worse-than-linear scaling, and moot now that it's deleted entirely.

### Target server results — 2-socket AMD EPYC 9254 (this session)

The SSH move happened; this is real target-class K8s node hardware (2×
EPYC 9254, 48c/96t total, 2 NUMA nodes of 48 logical CPUs each) — not the
10-core laptop above. Ad hoc sweep script (channel count × thread count ×
NUMA pinning, 3 repeats/config, 15s clock-ramp burn-in before each sweep —
this machine idles at 1.5GHz and needs sustained load before schedutil
ramps to boost clocks, which otherwise confounds thread-count comparisons)
in scratchpad, not committed; raw CSVs + the sweep script itself were not
kept in the repo — re-run if this needs reproducing. Interactive charts:
https://claude.ai/code/artifact/e6212f17-5d39-48e0-9efb-79ff13264959

**Initial diagnosis (WRONG — see fix below, kept for the record since the
investigation trail matters):** first pass found 448 channels still not
viable (best ≈33.0ms/tick, NUMA-pinned, 32 threads — ~12.6x over budget,
worse relatively than the laptop's ~2x despite ~10x more logical CPUs),
throughput saturating at ~24-32 threads and regressing beyond that, and
concluded the bottleneck was Box-Muller's per-sample log/sqrt/sin/cos
(`numba.config.USING_SVML` is `False` on AMD, so no vectorized-libm
speedup) combined with numba parallel-dispatch fork-join overhead.

**Actual root cause (found by isolating raw kernel cost from the full
`generate_next_tick` call and finding a ~14ms unexplained gap): pure
memory-allocation overhead, not compute.** `generate_next_tick` allocated
a fresh 14.7MB `np.zeros((n_samples, num_channels), complex128)` array
AND took a fresh array back from `synth_noise_all_channels` (itself
allocating `out_real`, `out_imag`, then concatenating to complex) AND did
a separate full-array `+=` — all of that twice per tick, once per
polarization, none of it Box-Muller math. **Fixed**: added
`synth_noise_all_channels_into` (writes directly into a caller-provided
buffer, no allocation) and gave `DirectSynthesisStreamer` a persistent
per-pol output buffer reused across ticks (`_get_output_buffer`); noise
now writes first (covering every cell, so no zero-fill needed) and tone
adds on top. Verified bit-identical to the old allocating path for the
same seed/index (see the `__main__` correctness check added alongside
this fix). **Confirmed safe to mutate in place**: `ScanRunner._run` calls
`generate_next_tick` synchronously and `HeapAccumulator.add` immediately
copies the data out via `np.concatenate` before the next tick can run —
nothing downstream ever holds a reference to the buffer across ticks.

**Result: 448 channels (350MHz, full band) is now comfortably viable.**
Re-swept end-to-end after the fix:

| channels | best (single NUMA node, 48 threads) | best (unpinned, 96 threads) |
|---|---|---|
| 96 | 0.54ms (20%) | 0.45ms (17%) |
| 160 | 0.75ms (29%) | 0.65ms (25%) |
| 224 | 0.97ms (37%) | 0.72ms (27%) |
| 288 | 1.19ms (45%) | 0.78ms (30%) |
| 352 | 1.42ms (54%) | 0.89ms (34%) |
| 416 | 1.69ms (65%) | 1.04ms (40%) |
| 448 | **1.84ms (70%)** | **1.10ms (42%)** |

(% is of the 2.621ms budget.) Every channel count now clears the 80%
comfort bar with just one socket's worth of threads (48) — down from
33.0ms/1260% before the fix. Unpinned with all 96 threads is faster in
isolation, but that hands a whole physical node to one station pod;
NUMA-pinning to one node (48 threads) is the better default since it
leaves the other socket free for a second station pod. This flips the
earlier "more threads stop helping past 24-32" finding, too: with the
allocation overhead gone, more threads help monotonically again (up to
96), because the per-dispatch work is now large enough relative to
fork-join overhead to actually benefit from more parallelism.

**Lesson for future benchmarking here**: always isolate a suspiciously
expensive method into its component kernel calls vs. its full
Python-level body before trusting a "this kernel is the bottleneck"
conclusion drawn only from thread-count/channel-count sweeps — the sweep
methodology in this session was rigorous (burn-in, repeats, NUMA control)
and still pointed at the wrong root cause, because it only ever measured
the whole method, never separated allocation from compute.

**Operational implication for the K8s deployment**: since the fix,
throughput scales positively all the way to 96 threads — the earlier
"give each pod ~24-32 threads" advice no longer holds. Give each station
pod one NUMA node's worth of threads (48, pinned via `numactl
--cpunodebind=N --membind=N` or Kubernetes' NUMA-aware Topology Manager)
for a comfortable 70%-of-budget margin at full 448-channel band, fitting
2 station pods per 96-thread physical node — a large improvement over the
pre-fix outlook where 448 channels wasn't viable on any tested config.

**Note: the numbers above are noise-only** (this was the allocation-fix
investigation, done before the backend convergence) — see the next
subsection for tone+noise+pulsar combined, which is what a real scan
actually runs.

### Combined tone + tiled-noise + pulsar (this session, after backend convergence)

The three source types were benchmarked individually in isolation
elsewhere in this file (tone: negligible; noise tile bank: 1.51ms/57.5%
at 448 channels; pulsar: 1.36ms/51.8%) but never together in one
streamer until the convergence into a single `DirectSynthesisStreamer`
made that the actual thing to benchmark — a real scan uses all three at
once, and per-tick costs don't necessarily just add (shared warm caches,
overlapping memory-bandwidth-bound work). Measured via
`benchmark_direct_synthesis.py`, NUMA-pinned to one node (48 logical
CPUs), tone + `n_tiles=256` noise bank + a 10ms/DM=2 pulsar all active
together:

| channels | best config | best mean | 8 threads |
|---|---|---|---|
| 96 | 16 threads | 0.588ms (22.4%) | 0.655ms (25.0%) |
| 448 | 24 threads | 1.884ms (71.9%) | 2.186ms (83.4%) |

(% is of the 2.621ms budget.) **448 channels combined clears budget at
every thread count tested from 8 threads up** — repeatability-checked (5
repeats at the best config, stdev 0.002ms at 448 channels). At exactly 8
threads it's just above the 80%-comfort bar (83.4%), not comfortably
under it the way any single component was alone — 16+ threads gets back
under 80% (74.1%). Noise tile-bank memory at these settings: 7.5GB (both
pols, `n_tiles=256`) — budget this alongside CPU when sizing a pod. If a
real deployment needs the full comfort margin at exactly 8 cores, revisit
`n_tiles`/`tile_n_samples` (both configurable) — smaller tiles or fewer
of them trade memory and noise-repeat frequency for a bit more per-tick
headroom, though the effect size hasn't been swept here.

### CORRECTED combined benchmark — real 384-channel max + oversampled budget (following session)

**The numbers above are superseded — re-run after the "SPS-CBF ICD
channelization" corrections** (384-channel real maximum, not 448;
`BLOCK_DURATION_S` = 2.21184ms, not 2.621ms — see that section for the
full derivation). Same `benchmark_direct_synthesis.py`, same NUMA
pinning (one node, 48 logical CPUs), same workload (tone + `n_tiles=256`
noise bank + a 10ms/DM=2 pulsar):

| channels | best config | best mean | 8 threads |
|---|---|---|---|
| 96 | 24 threads | 0.587ms (26.5%) | 0.689ms (31.2%) |
| 384 | 16 threads | 1.821ms (82.3%) | 2.034ms (92.0%) |

(% is of the corrected 2.212ms budget.) **The headline conclusion
changes for the worse.** At 384 channels — fewer channels than the old
(invalid) 448-channel config — the best-case result is 82.3% of budget,
which is *outside* the 80%-comfort bar this project has been using
throughout, not comfortably under it (the old 448-channel measurement
read 71.9% best-case, comfortably under). This isn't a contradiction:
work scales down only ~14% going from 448→384 channels, but the real
budget is ~15.6% *tighter* than what those old numbers were measured
against — the two effects land close enough to cancel that fewer
channels does NOT translate into more comfortable margin once measured
against the real budget. Repeatability-checked (5 repeats at
`numba_threads=16`): mean 1.863ms (84.2% of budget), spread 1.769-2.074ms
(80.0%-93.8%) — stdev 0.130ms, meaningfully noisier than the 96-channel
case's 0.026ms, and the spread's upper end is close enough to 100% that
an unlucky tick on this hardware could plausibly miss budget. **At
exactly 8 threads (92.0%) this is tight, not comfortable — if a real
deployment is hard-capped at 8 cores/pod, this configuration needs
either fewer channels, a smaller noise bank (`n_tiles`/`tile_n_samples`
tradeoff — not yet swept for its effect size), or accepting a
proportionally larger overrun-tolerance margin, not just "measure once
and move on."** Noise tile-bank memory at these settings: 6.44GB (both
pols, `n_tiles=256`, 384 channels — smaller than the old 448-channel
figure simply because there are fewer channels).

### One-time construction budget (new this session: target 10s, hard limit 30s)

Distinct from the per-TICK budget above — this is the one-time cost of
`DirectSynthesisStreamer.__init__` itself, before a scan can start.
Noise-bank fill and the pulsar's wideband sky-carrier generation are the
only non-trivial construction-time costs (tone is instant; the delay
feed setup is a dict lookup). Measured in isolation via
`benchmark_direct_synthesis.py`, NUMA-pinned, 448 channels:

| noise n_tiles | build (both pols) |
|---|---|
| 256 | 0.67s |
| 512 | 1.29s |
| 1024 | 2.54s |

| pulsar period | build, plain numpy.fft (pre-scipy) | build, scipy.fft workers=8 (current) |
|---|---|---|
| 10ms | 0.48s | 0.32s |
| 100ms | 4.44s | 3.21s (well-factored `n_wide`) |
| 200ms | — | 6.53s (well-factored) |
| 300ms | — | 9.84s (well-factored) |
| 50ms | — | 4.73s (POORLY-factored `n_wide` — see below; slower than 100ms despite a smaller array) |
| 1000ms | 46.17s — OVER the 30s hard limit | not re-measured at this size after scipy (see Pulsed sources section for why: not the per-array-size cost, but memory — retesting at 1s risked node stability again) |

Noise easily clears the 10s target at every tested size, no change here.
**Pulsar construction is a genuinely more complex story than "period vs.
time"** — see the Pulsed sources section above for the full finding:
`scipy.fft` gives a real but modest win (bigger on `_channelize_once`'s
batched FFTs than the two big 1D transforms), but the DOMINANT effect is
whether `n_wide` happens to factor into small primes — a poorly-factored
period can cost 3-6x more than a well-factored neighbor at a similar
nominal length. Run-to-run variance on this shared, contended host is
real too, not just factorization: a clean isolated run measured
300ms/well-factored at 9.84s (just under the 10s target); re-measured
as part of the full `benchmark_direct_synthesis.py` sweep (same host,
more going on around it) at 12.5s (over it) — roughly 25% higher,
plausibly load/thermal/clock-state, not a code change. Treat any of
these numbers as a band, not a precise threshold. Approximate limits,
accounting for BOTH sources of variance: **~100-200ms to reliably clear
the 10s target, up to perhaps ~0.5-0.9s for the 30s hard limit** if
the period's factorization is lucky and the host isn't contended — check
factorization luck cheaply via `scipy.fft.next_fast_len(n_wide) ==
n_wide`, but budget real margin underneath the hard limit rather than
trusting an exact number, and re-measure on the actual target host under
load if a specific period near these boundaries matters. Not tested
combined with a large noise bank at the largest settings simultaneously
(n_tiles=1024 + a long-period pulsar at once) — an earlier attempt at
that combination used enough transient memory to threaten node stability
on this shared host (see bug #16); if a real test scenario needs both
large at once, budget memory as carefully as time before trying it.

### CORRECTED pulsar construction budget — real 384-channel max + oversampled rate (following session)

**The pulsar table above is superseded, and the news is worse, not
better.** `n_wide = num_channels * round(period_s * channel_output_rate)`
depends on BOTH corrected quantities (384 not 448, and
`CHANNEL_OUTPUT_RATE_HZ` ≈ 925,925.93Hz not `CHANNEL_WIDTH_HZ` =
781,250Hz) — the resulting array lengths are different numbers with
noticeably WORSE factorization properties across the board, not just for
an occasional unlucky period like 50ms used to be. Noise-bank fill
numbers are unaffected by any of this (noise doesn't depend on
`channel_output_rate`) and still clear the 10s target easily at 384
channels: 0.55s / 1.18s / 2.22s at n_tiles=256/512/1024.

Re-swept `build_pulsar_template` at 384 channels, the corrected
`CHANNEL_OUTPUT_RATE_HZ`, NUMA-pinned, same host:

| period | n_wide | fast-factored? | build | vs. target/limit |
|---|---|---|---|---|
| 5ms | 1,777,920 | No | 0.36s | OK |
| 10ms | 3,555,456 | No | 0.61s | OK |
| 15ms | 5,333,376 | No | 0.71s | OK |
| 20ms | 7,111,296 | No | 3.26s | OK |
| 25ms | 8,888,832 | No | 3.17s | OK |
| 30ms | 10,666,752 | No | 1.57s | OK |
| 40ms | 14,222,208 | No | 2.03s | OK |
| 50ms | 17,777,664 | No | 5.81s | OK |
| 60ms | 21,333,504 | No | 3.26s | OK |
| 70ms | 24,888,960 | No | 4.52s | OK |
| 80ms | 28,444,416 | No | 4.16s | OK |
| 90ms | 31,999,872 | No | 11.21s | **OVER 10s TARGET** |
| 100ms | 35,555,712 | No | 16.94s | **OVER 10s TARGET** |
| 200ms | 71,111,040 | No | 10.38s | **OVER 10s TARGET** (barely) |
| 300ms | 106,666,752 | No | 51.07s | **OVER 30s HARD LIMIT** |

**Every single period tested comes back `fast-factored: No`** — under
the old (uncorrected) constants, only isolated unlucky periods like 50ms
had this problem while 100/200/300ms were well-factored and fast; under
the corrected oversampled rate, NOTHING tested is well-factored anymore.
This is a direct, structural consequence of the correction, not
something a different period choice dodges: `CHANNEL_OUTPUT_RATE_HZ` is
`CHANNEL_WIDTH_HZ * 32/27`, and that `/27` (`27 = 3^3`) means
`round(period_s * channel_output_rate)` lands on far fewer
small-prime-friendly integers than rounding against the old, cleaner
`CHANNEL_WIDTH_HZ` did — `scipy.fft.next_fast_len()`'s previously-
deferred padding fix (see the Pulsed sources section) is now a much
higher-value target than it looked before this correction, since it
would help essentially every period, not just occasional unlucky ones.

**Approximate safe limits are now substantially tighter than previously
documented** (still a band, not a precise threshold — see the
run-to-run variance discussion above, unaffected by this correction):
roughly **~80ms reliably clears the 10s target** (a real drop from the
old ~100-200ms figure — note the non-monotonic 15ms→20ms jump, 0.71s to
3.26s, underscoring that factorization noise dominates at these sizes
too, not just a smooth period-vs-time curve), and the 30s hard limit is
crossed somewhere between 200ms (10.38s, still under 30s) and 300ms
(51.07s, well over) — call it **~200ms as the practical ceiling** until
that range gets swept more finely. If a real test scenario needs a
pulsar period longer than ~80ms, budget real margin and re-measure
rather than trusting either this table or the old one.

### Historical hardware notes (legacy wideband path only, from earlier sessions)

- **Apple M5** (4P+6E cores): `ThreadPoolExecutor` threading plateaus at
  ~2.6-2.7x speedup — the 4-P-core topology explains why
  `time_chunks=4` used to be the old design's sweet spot.
  *(Superseded — that design no longer exists; kept for the general
  lesson that this laptop's P-core count is the relevant number, not its
  total core count.)*
- **Intel Xeon Silver 4410T** (10c/20t, dual-socket, NUMA, AVX-512):
  same ~3-4x `ThreadPoolExecutor` ceiling. An AVX-512-throttling
  hypothesis was proposed but the diagnostic tool built to test it
  (`diagnose_cpu.py`, since removed) averaged `/proc/cpuinfo` MHz across
  idle+busy cores, biasing low-thread-count readings down —
  inconclusive, never resolved either way.
- **AMD EPYC 9254/7443** (24-48c, NUMA): `numba`/`prange` scaled
  near-linearly well past where `ThreadPoolExecutor` plateaued — this is
  what confirmed the per-task-overhead diagnosis in bug #6.
- **NUMA pinning**: didn't change best-case mean throughput for the old
  `ThreadPoolExecutor` design, but meaningfully improved tail-latency
  *stability* for numba's thread pool (unpinned: stdev >10ms, spikes to
  50+ms on a ~3ms workload; pinned: no such spikes). **Recommend pinning
  on the target server regardless of mean-throughput reasoning** — tail
  latency is what matters for a per-tick real-time budget.

## Observability (not yet built)

Goal: distinguish "the simulator is falling behind its performance
budget" from "CBF is genuinely failing to apply the delay polynomial
correctly" when a delay-tracking test fails. Since `sim_time` is
deterministic and clock-independent, a slow tick never corrupts CONTENT —
it just arrives late. Plan: dropped-heap counters (queue-full events,
already logged via `log.warning` in `common.ScanRunner._run` but not
exported as a metric) and "falling behind pacing" event counters
(likewise currently just a log line, `OVERRUN_TOLERANCE` in common.py),
exported per-scan alongside the existing `queue_depth` Tango attribute. A
test harness should then correlate CBF's phase-gradient test failure
against this telemetry: clean telemetry + failing gradient → real CBF
bug; backed-up queue/dropped heaps → simulator artifact.

## Immediate next steps, in priority order

1. ~~Benchmark `DirectSynthesisStreamer` on real target server hardware~~
   **Done this session** on a 2-socket AMD EPYC 9254 — see "Target server
   results" above.
2. ~~Optimize the noise kernel~~ **Done this session, but not the way this
   item originally predicted.** The dominant cost turned out to be
   per-tick array allocation (bug #13), not Box-Muller math — fixing the
   allocation (persistent reused buffer, `synth_noise_all_channels_into`)
   made 448 channels viable (1.84ms/70% of budget, one NUMA node, 48
   threads) without touching the RNG at all. The Ziggurat/cheaper-RNG
   idea is no longer necessary; leave it as a future lever only if a
   future channel-count increase reopens the budget gap.
3. **Re-benchmark end-to-end under real Kubernetes pod co-scheduling**,
   not just one workload at a time on a bare node — this session's
   post-fix numbers (2 pods/node comfortably, or 1 pod/node unpinned
   using all 96 threads for even more margin) assumed no other pod
   contending for the same physical cores/memory bandwidth. Confirm this
   holds when multiple station pods actually run concurrently on one
   node, since noise generation is now memory-bandwidth-bound rather than
   allocation-bound, and bandwidth is a shared resource across pods on
   the same socket.
4. **Verify `channel_info`/`antenna_info` bit-packing against the real
   ICD document** (not the screenshot) — see the ICD section above, this
   is the single highest-priority correctness gap regardless of
   benchmarking outcomes.
5. Build the observability/telemetry addition described above.
6. ~~Derive a direct per-channel representation for pulsed sources, wire
   it in, converge the prototypes, delete the legacy backend~~ **All done
   this session.** The pulsar went through three designs (v1/v2/v3 — see
   the Pulsed sources section above) before being verified against an
   external reference (dispersion constant cross-validated against
   NANOGrav's PsrSigSim to 0.014%) and against itself numerically
   (known-tone channel-mapping check; cross-station coherence after delay
   compensation measured at 1.0000, confirming actual support for
   coherent multi-station beamforming). It was then converged with the
   noise tile bank and the existing tone kernel into one
   `DirectSynthesisStreamer`, wired into `simulator.py` as the sole
   backend, and `wideband_streamer.py` plus the three separate prototype
   modules and their benchmarks were deleted (see "Code layout").
   Combined tone+noise+pulsar clears budget at 448 channels (see
   Benchmarking's "Combined tone + tiled-noise + pulsar" subsection) --
   **since superseded**: 448 channels was never a valid configuration
   (the real ICD maximum is 384, see "SPS-CBF ICD channelization"); see
   Benchmarking's corrected-numbers subsection for the real 384-channel
   result.
   Remaining open items: (a) ~~decide on a real `base_freq_hz`~~ **done,
   then corrected the following session**: channel 64 / 50.0 MHz exactly
   (was briefly 65/50.78125MHz, off by one channel) — see the SPS-CBF
   ICD channelization section above, (b) flux/SNR
   calibration against the noise floor if the goal extends to testing
   whether PSS/PST can actually detect the injected pulsar, not just
   exercising delay-tracking, (c) confirm with whoever owns CBF's
   pulsed-source test cases whether this level of astrophysical
   approximation (achromatic profile, Gaussian shape, no pulse-to-pulse
   jitter) is sufficient, (d) ~~`StartScan` still hardcodes `source_cfgs`/
   `noise_cfg` rather than accepting them as scan parameters~~ **done**:
   `StartScan` now takes a single JSON string argument
   (`{obs_time_epoch_s, scan_duration_s, scan_id, subarray_id, beam_id,
   source_cfgs}`) — `subarray_id`, `beam_id`, and `source_cfgs` (including
   a required per-source `delay_attr_uri`, see "Per-source delay" above)
   are no longer device properties, so a station can be reassigned
   between subarrays/beams/sources across scans without a pod restart.
   There is still no convenience default tone (an empty `source_cfgs`
   list means no tone/pulsar sources at all, since a fabricated default
   would need a fabricated delay too, which is exactly what "delay is
   required" is meant to rule out). `noise_cfg` is still hardcoded in
   `StartScan` — revisit if a test needs to vary noise parameters
   per-scan too.
7. Confirm the exact CSP LMC command for pushing a delay model without
   going through TMC.
8. ~~Long-period pulsars violate the one-time construction budget~~
   **`scipy.fft` (with `workers=`) adopted this session** — real but
   modest win (biggest on `_channelize_once`'s batched FFTs, ~4-5x;
   ~15-20% on the two big 1D transforms, bandwidth-bound). **Bigger
   finding along the way, not yet acted on**: construction time depends
   far more on whether the wideband array length (`n_wide`) factors into
   small primes than on period length itself — a poorly-factored period
   measured 3-6x slower than a well-factored neighbor of similar size
   (see the Pulsed sources section for the verified example: 50ms vs.
   100ms). ~~Approximate safe limits as of that session: ~100-300ms for
   the 10s target, ~300ms-0.9s for the 30s hard limit~~ **SUPERSEDED,
   and worse, after the channelization corrections** (see "SPS-CBF ICD
   channelization" and Benchmarking's "CORRECTED pulsar construction
   budget" sections): every period tested now comes back
   poorly-factored, not just occasional unlucky ones — real limits are
   now roughly **~80ms for the 10s target, ~200ms for the 30s hard
   limit**. `scipy.fft.next_fast_len()` is a promising, already-
   identified candidate for fixing the factorization sensitivity
   directly, but padding correctness (keeping `_channelize_once`'s
   exact-multiple-of-`num_channels` requirement, and not silently
   changing what "one period" means to the dispersion FFT's implicit
   circular convolution) needs real validation — not implemented. **Now
   a higher-priority item than before this correction**: it would help
   essentially every period, not just the occasional unlucky one.
9. ~~Minimize custom numba/RNG code where the per-tick budget doesn't
   require it~~ **Done this session for noise and the pulsar's wideband
   carrier** (see the Noise and Pulsed sources sections, and bug #16 for
   a real memory bug caught along the way) — both are now plain
   (thread-parallelized, for noise) numpy, measured FASTER than the numba
   they replaced, not just "acceptable." What's still numba, deliberately:
   `synth_tone_channel` and `add_pulsar_tick` (genuinely per-tick,
   2.621ms budget, already needed real numerical tricks — NCO
   phase-accumulator, bug #15 — just to clear it) and `_splitmix64_hash`
   (a single per-tick integer hash for tile-index selection, not a
   statistical distribution, kept small and separate from the
   Box-Muller machinery it used to feed).
10. **The 8-core/pod comfort margin no longer holds at the real
    384-channel maximum** — see Benchmarking's "CORRECTED combined
    benchmark" subsection: best-case is 82.3% of the corrected 2.212ms
    budget (outside this project's own 80%-comfort bar), and exactly 8
    threads is 92.0% (tight, not comfortable). Re-sweep
    `n_tiles`/`tile_n_samples`'s effect on per-tick headroom (not yet
    done), and/or confirm with whoever owns the deployment whether an
    8-core/pod hard cap is actually fixed or has some flexibility, before
    treating full-band + 8 cores as a settled, comfortable configuration.
11. Confirm the "zero phase at first sample of every SPS SPEAD packet"
    ICD passage's actual intent against real hardware/firmware or
    whoever owns the SPS filterbank spec — this session reasoned through
    a NORMALIZATION-convention reading (see `direct_synthesis.py`'s
    "OVERSAMPLING AND PER-PACKET PHASE" section) and implemented nothing
    further on that basis, but explicitly flagged it as reasoned-not-
    verified. Revisit if a delay-tracking test ever shows a phase
    discontinuity at heap boundaries this reasoning didn't predict.
12. ~~Pre-generate pulsar templates offline instead of building at scan
    construction~~ **Done this session** — see "Pulsar catalog" above:
    `pulsar_catalog.py`/`generate_pulsar_catalog.py`, both source_cfg
    styles supported side by side. Open follow-ups: (a) decide the
    ACTUAL set of named pulsars a real deployment needs (the three
    shipped are illustrative, not curated for any specific test
    campaign) — periods/DMs pulled from CBF's own pulsed-source test
    requirements once known, (b) wire `generate_pulsar_catalog.py` into
    an actual OCI image build step once a Dockerfile exists for this
    project (none does yet), (c) the catalog data is ~1.1GB for just
    three illustrative entries and scales with period length — budget
    image size deliberately once the real named-pulsar set is decided,
    not just build time.

## Setup

Python 3.10 (`.python-version`), dependency management via `uv`
(`pyproject.toml`/`uv.lock`). One private index configured
(`artefact.skao.int`, for `ska-tango-base`) — confirm network access to
it from wherever you're running this before `uv sync`.

```
uv sync                                              # installs everything, incl. dev group
uv run pytest                                        # ALL correctness checks: tone, noise (kernel + tile bank), pulsar, delay feeds
python -m ska_low_station_beam_simulator.benchmark_direct_synthesis  # real multi-core timing, 96 + 384 (real ICD max) channels, tone+noise+pulsar combined
python -m ska_low_station_beam_simulator.generate_pulsar_catalog     # writes pulsar_catalog_data/ (gitignored, ~1.1GB for the default 3 entries) -- see "Pulsar catalog"
```

Correctness checks used to live in `direct_synthesis.py`'s
`if __name__ == "__main__":` block (`python -m
ska_low_station_beam_simulator.direct_synthesis`) — converted to real
`pytest` tests under `tango/tests/` this session, one assertion-group per test
instead of one long script, so a failure identifies exactly which
property broke. `tango/tests/test_direct_synthesis.py` covers tone/noise/pulsar
(the old `__main__` checks); `tango/tests/test_delay_feeds.py` covers
`DelayFeed`/the required-delay_feed validation/the per-source-delay-divergence
integration check (`direct_synthesis.py`'s own local-generation path,
still exercised directly, independent of `simulator.py`);
`tango/tests/test_simulator_delay_wiring.py` covers the
Tango attribute subscription **and gRPC-forwarding** plumbing in
`simulator.py` (against fake `AttributeProxy`/gRPC-stub objects, not a
live Tango context or a live Go process — this codebase still doesn't
unit test the actual Tango device server layer; see "Tango device now
drives the Go gRPC simulator" above for a live-process smoke test that
DID exercise a real `cmd/simulator` binary, not committed as an
automated test); `tango/tests/test_simulator_tone_source_request.py`
covers `build_tone_source_request`'s JSON-boundary validation (rejects
non-'tone' kinds, missing `delay_attr_uri`).

**Regenerating the Python gRPC stubs** (after editing the repo root's
`api/simulator.proto` — see the Go side's own regeneration command in
`README.md`):

```
mkdir -p tango/src/ska_low_station_beam_simulator/simulatorpb   # first time only
uv run python -m grpc_tools.protoc \
    --proto_path=api \
    --python_out=tango/src/ska_low_station_beam_simulator/simulatorpb \
    --grpc_python_out=tango/src/ska_low_station_beam_simulator/simulatorpb \
    --pyi_out=tango/src/ska_low_station_beam_simulator/simulatorpb \
    api/simulator.proto

# REQUIRED every time -- grpc_tools.protoc's Python plugin always emits
# a bare `import simulator_pb2 as simulator__pb2`, which doesn't resolve
# from inside the simulatorpb package (see "Tango device now drives the
# Go gRPC simulator" above):
sed -i '' 's/^import simulator_pb2 as simulator__pb2$/from . import simulator_pb2 as simulator__pb2/' \
    tango/src/ska_low_station_beam_simulator/simulatorpb/simulator_pb2_grpc.py
```

`grpcio-tools` (the `dev` dependency group) provides `grpc_tools.protoc`
— no separate `protoc`/plugin binaries to install beyond what `uv sync`
already pulls in, unlike the Go side's regeneration command.

`pytango` isn't required to run the above — `simulator.py` degrades to
stub Tango classes if `pytango` isn't installed (importable, not
deployable). `spead2` is no longer a dependency at all (see the SPS-CBF
ICD section and bug #18 above): `spead.SpsPacketizer` hand-rolls its
own minimal SPEAD-64-48 encoder instead, since spead2's own packet
encoder cannot produce CBF's real 6-item heap format.

If this test server has machine-specific setup notes you don't want
committed to the shared `CLAUDE.md` (paths, credentials, which NUMA nodes
to pin to, etc.), put them in a `CLAUDE.local.md` alongside this file —
Claude Code loads it automatically, appended after this file, in any
session started from this directory (including over SSH), and it isn't
meant to be checked in.
