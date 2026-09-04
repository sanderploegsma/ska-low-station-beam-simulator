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
Currently developed/benchmarked at 96 channels (75 MHz); the real target
is the full SKA-Low band, 448 channels (350 MHz) — **viable as of this
session's allocation-overhead fix, comfortably clearing budget on target
server hardware even with tone + noise + a pulsar combined, see
Benchmarking below.**

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
src/ska_low_station_beam_simulator/
  common.py                    shared plumbing, backend-agnostic
  direct_synthesis.py          the SOLE backend: DirectSynthesisStreamer (tone + tiled noise + pulsar)
  simulator.py                 Tango device server (StationSimulatorDevice)
  benchmark_direct_synthesis.py  benchmarks DirectSynthesisStreamer, incl. tone+noise+pulsar combined
```

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
`simulator.py` always constructs a `DirectSynthesisStreamer`; there is no
backend-selection branching left.

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
`DirectSynthesisStreamer`) are generated once at construction (still
using the same Box-Muller kernel, just called `n_tiles` times upfront
instead of once per tick); each tick does an O(1) per-(station, pol)
index draw plus a memcopy instead. Noise never enters a delay pipeline
either way — physically correct, since receiver noise originates locally
per station, after any signal-path delay would apply (this was a real
bug in the deleted wideband path: `shared` and `noise` were summed
*before* its delay/channelization pipeline, delay-correcting noise that
should never have been delay-corrected at all).

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

| n_tiles | bank size (both pols) | build time | first repeat (~ticks) | ~scan time |
|---|---|---|---|---|
| 256 | 7.0 GB | 2.2s | 20 | 52ms |
| 1024 | 28.0 GB | 8.8s | 40 | 105ms |
| 2048 | 56.0 GB | 31.9s | 57 | 148ms |

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
memcopy, at 448 channels:
- Plain `out[:] = bank[idx]` (single-threaded numpy, ignores thread
  count entirely): **1.51ms/tick (57.5% of budget) regardless of core
  count** — even 1 core is enough.
- A numba `prange`-parallelized copy does better still: 0.36ms (13.9%)
  at just 8 threads, versus 1.38ms (52.6%) single-threaded.

So **the target of ~8 CPU cores/pod is not just achievable but massively
over-provisioned for this approach's steady-state cost** — 1-2 cores
would already clear budget. The actual per-pod resource question this
approach shifts onto is **memory** (linear in `n_tiles`, ~27MB/tile
across both pols at 448 channels) and **one-time startup latency**
(scales with `n_tiles`, improves with more threads at build time only —
8 threads builds a 256-tile/7GB bank in 2.2s, a 1024-tile/28GB bank in
8.8s).

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
- One-time build cost is now plain numpy (wideband generate + FFT
  dispersion + channelize), not numba — thread count isn't the lever,
  period length is, since it scales with the wideband FFT length
  (`num_channels × n_period_samples`): 0.16s at a 1ms period, 4.5s at
  100ms, 46.7s at 1000ms (roughly 2x the older, wrong sub-frequency-averaging
  approach's build cost at the same period — a fair trade for actually
  being correct on two counts).
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

## SPS-CBF ICD heap structure — HIGHEST-PRIORITY UNVERIFIED ITEM

One heap = ONE CHANNEL = 2048 consecutive time-domain samples, both
polarisations interleaved per sample (Vreal, Vimag, Hreal, Himag, each
int8). `pkt_len = 0x2000` (8192 bytes) confirms this: 2048 × 4 bytes.

**The bit-field layout for `channel_info` and `antenna_info`
(`common.pack_channel_info`/`pack_antenna_info`) was read off a
SCREENSHOT of the ICD diagram, never the source document**, and the
screenshot's own column numbering looked internally inconsistent
(possible rendering artifact). Get this wrong and packets are malformed
in a way that may not even error, just silently misparse. **Verify
against the real ICD table before this touches real hardware** — nothing
else in this codebase should be prioritized above this.

Also unverified: whether `spead2`'s Python send API actually supports an
explicit `heap_counter`/`cnt` override the way `common.SpsPacketizer`
assumes.

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
    tick (once per pol)** — this allocation/copy churn, not Box-Muller
    math, was the dominant per-tick cost at high channel counts (~30ms of
    a ~33ms tick at 448 channels). Misdiagnosed at first as a
    compute/thread-dispatch problem — see "Target server results" below
    for the full investigation trail. **Fixed**: `synth_noise_all_channels_into`
    writes noise directly into a persistent, reused per-pol buffer
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
   Benchmarking's "Combined tone + tiled-noise + pulsar" subsection).
   Remaining open items: (a) decide on a real `base_freq_hz` (currently a
   50MHz placeholder default passed explicitly to pulsed configs, same
   unverified-ICD problem as `common.BASE_FREQ_HZ`) since dispersion
   physics is highly sensitive to the true band edges, (b) flux/SNR
   calibration against the noise floor if the goal extends to testing
   whether PSS/PST can actually detect the injected pulsar, not just
   exercising delay-tracking, (c) confirm with whoever owns CBF's
   pulsed-source test cases whether this level of astrophysical
   approximation (achromatic profile, Gaussian shape, no pulse-to-pulse
   jitter) is sufficient, (d) `StartScan` still hardcodes `source_cfgs`/
   `noise_cfg` rather than accepting them as scan parameters — wiring
   pulsed/noise-tile config through the actual Tango command interface is
   unstarted.
7. Confirm the exact CSP LMC command for pushing a delay model without
   going through TMC.

## Setup

Python 3.10 (`.python-version`), dependency management via `uv`
(`pyproject.toml`/`uv.lock`). One private index configured
(`artefact.skao.int`, for `ska-tango-base`) — confirm network access to
it from wherever you're running this before `uv sync`.

```
uv sync                                              # installs everything, incl. dev group
python -m ska_low_station_beam_simulator.direct_synthesis   # ALL correctness checks: tone, noise (kernel + tile bank), pulsar
python -m ska_low_station_beam_simulator.benchmark_direct_synthesis  # real multi-core timing, 96 + 448 channels, tone+noise+pulsar combined
```

`pytango` and `spead2` aren't required to run the above — `simulator.py`
degrades to stub Tango classes if `pytango` isn't installed (importable,
not deployable), and nothing except `common.SpsPacketizer`/the actual
device server touches `spead2`.

If this test server has machine-specific setup notes you don't want
committed to the shared `CLAUDE.md` (paths, credentials, which NUMA nodes
to pin to, etc.), put them in a `CLAUDE.local.md` alongside this file —
Claude Code loads it automatically, appended after this file, in any
session started from this directory (including over SSH), and it isn't
meant to be checked in.
