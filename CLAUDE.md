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
- **Kubernetes Indexed Job, one pod per station**, self-sufficient after
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

1. **Benchmark `DirectSynthesisStreamer` on real target server hardware**
   (the point of this SSH session) — confirm whether more cores close the
   96-channel comfort margin further and, more importantly, whether they
   close the 448-channel gap (currently ~2x over budget on a 10-core
   laptop). Use `benchmark_direct_synthesis.py` as-is; consider NUMA
   pinning per the historical note above if the server is multi-socket.
2. If 448 channels is still over budget on real hardware: optimize the
   noise kernel specifically (it's now the dominant cost at scale) —
   candidates include a cheaper RNG than Box-Muller, or restructuring to
   avoid recomputing `_splitmix64_hash` twice per complex sample.
3. **Verify `channel_info`/`antenna_info` bit-packing against the real
   ICD document** (not the screenshot) — see the ICD section above, this
   is the single highest-priority correctness gap regardless of
   benchmarking outcomes.
4. Build the observability/telemetry addition described above.
5. Derive (or explicitly decide to defer further) a direct per-channel
   representation for pulsed sources, to let `wideband_streamer.py` be
   deleted entirely.
6. Confirm the exact CSP LMC command for pushing a delay model without
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
