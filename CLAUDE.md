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
is the full SKA-Low band, 448 channels (350 MHz) — **not yet viable at
448 channels, see Benchmarking below.**

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
  direct_synthesis.py          fast path: DirectSynthesisStreamer (default)
  wideband_streamer.py         legacy path: StationStreamer (pulsed fallback only)
  simulator.py                 Tango device server (StationSimulatorDevice)
  benchmark.py                 benchmarks wideband_streamer.StationStreamer
  benchmark_direct_synthesis.py  benchmarks direct_synthesis.DirectSynthesisStreamer
  tiled_noise_streamer.py       EXPERIMENTAL: pre-generated noise bank variant, not wired into simulator.py
  benchmark_tiled_noise_streamer.py  benchmarks tiled_noise_streamer.TiledNoiseStreamer
resources/notebooks/benchmark.ipynb   STALE — predates the numba port, ignore/delete
```

Two independent, deliberately non-sharing signal-generation backends live
side by side rather than merged into one module, because the legacy
wideband+FFT path is expected to be dropped once direct synthesis covers
pulsed sources too — keeping them separate means deleting one later
doesn't touch the other. Each has its own private copy of small internals
(splitmix64 hash, Box-Muller) rather than sharing code between them.

`common.py` defines a `Streamer` `Protocol` (`channel_id_map`,
`num_channels`, `tick_n_samples()`, `generate_next_tick()`) that
`ScanRunner` drives structurally — it never imports or `isinstance`-checks
either concrete streamer class, so `common.py` has zero dependency on
which backend is in use. `simulator.py` (the only place that imports both
backends) picks one per scan: `DirectSynthesisStreamer` unless the scan's
`source_cfgs` contains a `kind='pulsed'` entry, in which case it falls
back to the legacy `StationStreamer`.

## Signal generation: two backends

### `direct_synthesis.DirectSynthesisStreamer` — the default, fast path

Bypasses wideband time-domain generation and FFT channelization
entirely, for **tone and per-pol station (receiver) noise only**:

- **Tone**: spectrally sparse — lives almost entirely in one channel bin
  after channelization. Synthesized directly via a closed-form complex
  exponential at the residual frequency; delay is a **continuous phase
  term** in the exponent, exact for a monochromatic tone (not an
  approximation of an integer/fractional sample split — that split is an
  artifact of time-domain discretization, sidestepped entirely by direct
  phase modulation). No ring buffer, no coarse/fine delay split. Cost is
  O(1) per tone, independent of channel count.
- **Noise**: the DFT of i.i.d. complex Gaussian noise is itself i.i.d.
  complex Gaussian (unitary transform) — generating independent Gaussian
  samples directly at (sample, channel) resolution reproduces the exact
  statistics of "wideband noise, then channelized" without ever computing
  an FFT. Never enters a delay pipeline, which is also physically
  correct: receiver noise originates locally per station, after any
  signal-path delay would apply (see bug #12 below).
- **Pulsed sources are explicitly PARKED.** Broadband by nature, so the
  "lives in one bin" shortcut doesn't apply; no closed-form per-channel
  representation has been derived. `DirectSynthesisStreamer` raises
  `ValueError` on `kind='pulsed'` rather than mishandling it silently.

Correctness-verified (`python -m ska_low_station_beam_simulator.direct_synthesis`
runs the checks): tone lands in the correct channel with the correct
residual frequency; delay-as-phase matches the analytic `-2π·freq·τ`
shift to ~1e-14; noise has correct mean/std and cross-channel
independence; both kernels are deterministic/seekable from `t` alone.

Numba is used for both kernels — deliberately re-verified rather than
assumed, since it's easy for an optimization to stop pulling its weight
as code evolves. At 448 channels, numba+prange noise generation beats a
vectorized numpy/Philox equivalent by ~3x at 8-10 threads (2.3ms vs
7.3ms), because `numpy.random.Generator` has no built-in
multi-threading — numba is the only way to actually parallelize that
stage, and noise is the dominant cost at scale (see Benchmarking). Tone
also wins with numba (14.5μs vs 25.1μs vectorized numpy) by avoiding
several full-array temporaries, though it barely matters for the budget
either way.

### `wideband_streamer.StationStreamer` — legacy, pulsed-source fallback only

Generates a wideband time-domain signal, ring-buffers it for coarse/fine
delay correction, then FFT-channelizes. Superseded by direct synthesis
for tone+noise; kept only because pulsed sources aren't covered there
yet. **Per the scaling analysis below, this path is very unlikely to be
viable at 448 channels regardless of further optimization** — going from
96→448 channels is worse-than-linear work growth (FFT is N log N)
against a fixed per-tick budget.

**Bug #12, still unfixed in this class only:** station (receiver) noise
gets delay-corrected identically to the sky signal (`shared` and `noise`
are summed before the ring-buffer/delay/channelization pipeline) — wrong,
since receiver noise originates locally per station, after any
signal-path delay would apply. `DirectSynthesisStreamer` fixes this by
construction (noise never touches the delay pipeline). Only fix it here
too if this legacy path ends up seeing real use beyond pulsed-only
fallback duty — not worth the restructuring otherwise.

### `tiled_noise_streamer.TiledNoiseStreamer` — EXPERIMENTAL, not wired into `simulator.py`

Explored to answer a specific question: could each station pod get by
with far fewer CPU cores (~8, to fit more stations per node) than
`DirectSynthesisStreamer` needs, by pre-generating a bank of `n_tiles`
noise tiles once at scan setup and having each tick do an O(1)
per-(station, pol) index draw + memcopy instead of fresh Box-Muller
generation? Reuses `direct_synthesis.py`'s tone/noise kernels rather than
forking them again — this is an experimental variant OF the
direct-synthesis backend, not a third independent backend in the sense
`wideband_streamer.py` and `direct_synthesis.py` are kept apart.

**Correctness constraint, non-negotiable:** the tile index MUST be drawn
from each station's own (station, pol) noise seed — never from a value
shared across stations (e.g. seeded only from `obs_time`). Sharing it
would make every station emit byte-identical "noise" for a given tick,
silently breaking any test that depends on receiver noise being
uncorrelated across the array (beamforming-SNR gain, cross-correlation
baseline noise floor). `TiledNoiseStreamer` gets this right by
construction; its own `__main__` includes a cross-station-independence
check specifically to guard against a future edit "simplifying" this
back to a shared seed.

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
scan, this approach trades a small, one-time memory/startup cost for an
essentially-free steady-state CPU footprint — a very different profile
from `DirectSynthesisStreamer`, which needs ~24-48 threads for headroom
at 448 channels (see Benchmarking) but has no periodicity at all. Pick
based on what the test suite actually needs, not on which one benchmarks
better in isolation.

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

1. **V/H noise sharing**: one `NoiseSource` reused for both pols gave
   numerically identical noise (it's a deterministic pure function of
   inputs). Fixed: separate seeds per pol — preserved in both backends.
2. **Absolute-epoch-time precision collapse (delay polynomial)**:
   evaluating a 5th-order polynomial against raw Unix-epoch-scale `t`
   (~1.8×10⁹) blows up float64 precision. Fixed: evaluate relative to
   `start_validity_sec` (`common.DelayPolynomial.eval_delay_seconds`).
3. **Same bug, tone phase**: absolute `t` fed directly into phase
   computation collapsed float64 precision (~0.1-0.5 rad error). Fixed:
   all generation takes `t_rel = t - obs_time_ref`, bounded by scan
   duration. This is why every kernel in both backends takes a
   small-magnitude relative time, never raw epoch time — check any new
   kernel against this before assuming it's a minor style choice.
4. **Ring buffer `np.roll` on the whole buffer every tick** — O(capacity)
   regardless of new-sample count. Fixed: real O(1) circular buffer
   (`wideband_streamer.RingBuffer`).
5. **Channelizer looping `np.fft.fft()` per output row** — thousands of
   individual FFT calls/tick. Fixed: vectorized via `sliding_window_view`
   + one batched FFT call.
6. **`ThreadPoolExecutor`-chunked generation plateaus at ~4-8 workers**,
   confirmed on 3 different CPU architectures (Apple M5, Intel Xeon
   Silver 4410T, AMD EPYC 9254/7443) regardless of core count —
   per-task Python/GIL overhead, not compute/bandwidth. Fixed: moved to
   `numba`+`prange`, near-linear scaling well past that plateau.
7. **Numba's `nopython` mode doesn't support `numpy.random.Philox`**
   (confirmed by direct test). Replaced with a from-scratch splitmix64
   hash + Box-Muller — separate copies in `wideband_streamer.py` and
   `direct_synthesis.py`. Statistically correct, deterministic, seekable,
   but NOT bit-identical to the old Philox sequence.
8. **`fftshift` was a full-array reorder every tick for no numerical
   reason** — replaced with natural FFT bin order + a fixed permutation
   applied only to small `channel_id` integers
   (`WidebandChannelizer._natural_to_external`, exposed uniformly via
   `channel_id_map` on both streamer classes — identity for direct
   synthesis, since it has no FFT bin order to undo).
9. **`OMP_NUM_THREADS=1` cap** — originally needed to stop numpy/BLAS
   oversubscribing against `ThreadPoolExecutor`-chunked generation;
   confirmed no-longer-necessary-but-harmless once generation moved to
   numba. Still set in `wideband_streamer.py` (only place it's relevant).
10. **`numpy.fft.fft` has no multi-threading at any array size** —
    switched to `scipy.fft.fft` (`workers=` param). `FFT_WORKERS > 1`
    empirically made things *worse* on EPYC (contention with numba's
    thread pool); later repeats softened this to "1-8 are roughly
    equivalent, not worth tuning further."
11. **Benchmark-script bug** (not production code): a sweep loop left
    `FFT_WORKERS` at its last-tried value instead of the sweep's best —
    fixed by pinning explicitly in `benchmark.py`. General lesson: don't
    trust a single-pass sweep result on this class of hardware without a
    5+ repeat repeatability check.
12. **Receiver noise incorrectly delay-corrected** in the wideband
    architecture. **Fixed in `DirectSynthesisStreamer` by construction**
    (noise never enters a delay pipeline); **still present in
    `StationStreamer`**, deliberately not fixed there since it's now
    pulsed-fallback-only (see backend section above).
13. **`pyproject.toml` was missing `scipy`** even though
    `wideband_streamer.py` imports `scipy.fft` — the module couldn't even
    be imported. Fixed via `uv add scipy`; if you hit `ModuleNotFoundError:
    scipy` on a fresh checkout, run `uv sync`.
14. **`DirectSynthesisStreamer.generate_next_tick` allocated a fresh
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

## Benchmarking

**Per-tick budget is fixed regardless of channel count**:
`HEAP_LEN / CHANNEL_WIDTH_HZ = 2048 / 781250 Hz ≈ 2.621ms`
(`common.BLOCK_DURATION_S`). Work scales with channel count; the budget
doesn't. "Comfortable" per your stated bar is ~80% of budget; 95-110% is
not good enough as a baseline.

### Current results (this session, `python -m ...benchmark_direct_synthesis`)

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

Legacy `wideband_streamer.StationStreamer` (`python -m ...benchmark`), same
machine, 96 channels: ~2.4-2.6ms/tick at numba_threads=8-10 with
FFT_WORKERS=2-8 — right around budget, not comfortably under it.
Consistent with this path's earlier EPYC results below; not worth
re-benchmarking at 448 channels given its known worse-than-linear
scaling and its now-secondary (pulsed-fallback-only) role.

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
   per-tick array allocation (bug #14), not Box-Muller math — fixing the
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
6. Derive (or explicitly decide to defer further) a direct per-channel
   representation for pulsed sources, to let `wideband_streamer.py` be
   deleted entirely.
7. Confirm the exact CSP LMC command for pushing a delay model without
   going through TMC.

## Setup

Python 3.10 (`.python-version`), dependency management via `uv`
(`pyproject.toml`/`uv.lock`). One private index configured
(`artefact.skao.int`, for `ska-tango-base`) — confirm network access to
it from wherever you're running this before `uv sync`.

```
uv sync                                              # installs everything, incl. dev group
python -m ska_low_station_beam_simulator.direct_synthesis   # kernel correctness checks + cost comparison
python -m ska_low_station_beam_simulator.benchmark_direct_synthesis  # real multi-core timing, 96 + 448 channels
python -m ska_low_station_beam_simulator.benchmark   # legacy wideband path timing, 96 channels
python -m ska_low_station_beam_simulator.tiled_noise_streamer   # EXPERIMENTAL noise-bank variant: correctness checks
python -m ska_low_station_beam_simulator.benchmark_tiled_noise_streamer  # EXPERIMENTAL: memory/startup/per-tick cost sweep, 448 channels
```

`pytango` and `spead2` aren't required to run the above — `simulator.py`
degrades to stub Tango classes if `pytango` isn't installed (importable,
not deployable), and nothing except `common.SpsPacketizer`/the actual
device server touches `spead2`.

**Nothing in this repo has been committed to git yet** (working tree
only, no commits on `main`) — worth doing before or as part of moving it
to another machine, rather than copying files by hand each time.

If this test server has machine-specific setup notes you don't want
committed to the shared `CLAUDE.md` (paths, credentials, which NUMA nodes
to pin to, etc.), put them in a `CLAUDE.local.md` alongside this file —
Claude Code loads it automatically, appended after this file, in any
session started from this directory (including over SSH), and it isn't
meant to be checked in.
