# Project history

This document is the archive of *how this project got to its current state*:
design iterations that were tried and rejected, bugs that were found and
fixed, benchmark investigations, and corrections to earlier assumptions.

It exists so that inline code documentation, `README.md`, and `CLAUDE.md` can
stay focused on describing the **current** implementation without losing the
reasoning and evidence behind it. If you're trying to understand why the
code looks the way it does, or whether an idea has already been tried and
found wanting, look here. If you're trying to understand what the code
currently does, see `README.md` (and the code itself) instead.

Sections are organized by topic, not strictly chronologically, since later
sessions frequently corrected earlier ones (most notably the SPS-CBF ICD
channelization numbers). Where a correction happened, both the original and
the corrected version are recorded, since the trail of "this looked right
until it was checked" is itself a useful lesson.

## Contents

1. [Architectural decisions: background and rationale](#architectural-decisions-background-and-rationale)
2. [Backend convergence: from three prototypes to one `DirectSynthesisStreamer`](#backend-convergence-from-three-prototypes-to-one-directsynthesisstreamer)
3. [SPS-CBF ICD corrections](#sps-cbf-icd-corrections)
4. [Known bugs (fixed)](#known-bugs-fixed)
5. [Python/Tango benchmarking history](#pythontango-benchmarking-history)
6. [Tango device → Go gRPC simulator migration](#tango-device--go-grpc-simulator-migration)
7. [Go simulator: real-hardware profiling and pacing investigation](#go-simulator-real-hardware-profiling-and-pacing-investigation)
8. [Historical hardware notes](#historical-hardware-notes)

---

## Architectural decisions: background and rationale

The project's settled architectural decisions are listed briefly in
`CLAUDE.md`. Their background:

- **Independent delay generation, not shared with CBF.** Independence from
  CBF matters specifically because CNIC (the hardware/firmware tool this
  simulator replaces) is built by the same team that builds the CBF firmware
  being tested — using CBF's own delay polynomial to generate the test data
  would be tautological, since a bug in CBF's delay computation could then
  never show up in a test that used the same buggy computation to generate
  the "true" signal. The simulator therefore generates delay from its own
  source geometry, while CBF's own real delay-poly Tango device (not a
  reimplementation, not routed through TMC) supplies the polynomial CBF
  actually applies. These are two independent code paths computing
  related-but-different quantities from the same underlying source
  position/geometry.
- **CSP LMC drives the scan, not TMC.** This avoids TMC's full subarray
  observation lifecycle. The delay-poly schema
  (`ska-low-csp-delaymodel/1.0`, ADR-88 in `ska-telmodel`) is a documented
  wire-format interface; TMC is the usual producer but not architecturally
  required. CBF's own delay-poly Tango device (used for CBF's own testing,
  commands like `PstOffsetRaDec`) proves a non-TMC injection path exists.
  The exact CSP LMC command for this was never confirmed during the
  sessions that established this design — treat it as still open if it
  matters for current work.
- **Deterministic, clock-independent `sim_time`.** Content generation never
  depends on any pod's own wall clock — only on `obs_time_ref` (given once,
  identically, to every station pod at scan start) plus a derived tick index
  / relative time. This is what makes multi-pod generation consistent
  without PTP or inter-pod coordination beyond the initial `obs_time`. Every
  generation kernel (tone phase, noise sample index, delay polynomial
  evaluation) is a pure function of a small, `obs_time`-relative `t`,
  specifically so any pod can independently compute any tick without shared
  state. This was reinforced by a real bug (see "Known bugs" below):
  evaluating against raw Unix-epoch-scale time collapses float64 precision,
  so every kernel takes a small-magnitude relative time instead.
- **One Tango device server per station, one Kubernetes pod per device.**
  Self-sufficient after receiving `obs_time` at scan start.
- **TAI2000 as the SKA epoch for `heap_counter`.** Unix→TAI2000 uses
  `astropy` (`common.unix_to_tai2000_seconds`); a hardcoded-leap-second
  fallback exists (Python) but is explicitly unsafe for production without
  astropy. The Go port's `internal/common/tai2000.go` currently only has the
  hardcoded-offset fallback, since Go has no drop-in equivalent of
  `astropy` — a real deployment needs a proper leap-second source (or to
  share one with the Python side) before this matters.

## Backend convergence: from three prototypes to one `DirectSynthesisStreamer`

The Python simulator used to have two backends plus three separate
experimental prototype modules:

- `wideband_streamer.py` — the legacy wideband+FFT `StationStreamer`. Used a
  ring buffer with `np.roll` on the whole buffer every tick (O(capacity)
  regardless of new-sample count), a channelizer that looped `np.fft.fft()`
  per output row (thousands of individual FFT calls/tick), and a `fftshift`
  every tick for no numerical reason. It also had a real bug: `shared` and
  `noise` were summed *before* the delay/channelization pipeline, meaning
  receiver noise was incorrectly delay-corrected (noise originates locally
  per station, after any signal-path delay would apply).
- `tiled_noise_streamer.py` — an experimental noise tile-bank prototype.
- `pulsed_source_streamer.py` — an experimental per-channel pulsar
  prototype.

All of this was converged into `direct_synthesis.py`'s single
`DirectSynthesisStreamer`, which handles tone, per-pol station noise (via a
pre-generated tile bank), and pulsed/pulsar sources directly, with no
per-tick wideband generation or per-tick FFT channelization for any source
type. The legacy files and their dedicated benchmark scripts were deleted
outright (not kept behind a flag) once the converged backend covered their
functionality; `scipy` was dropped from `pyproject.toml` at the same time
since the legacy wideband path was its only user (it was later reinstated
for an unrelated reason — see the pulsar section below).

### Tone

Synthesized directly via a closed-form complex exponential at the residual
frequency, with delay as a continuous phase term in the exponent — exact
for a monochromatic tone, unlike a time-domain coarse/fine delay split
(which is itself an artifact of time-domain discretization). Cost is O(1)
per tone, independent of channel count. Verified: lands in the correct
channel with the correct residual frequency; delay-as-phase matches the
analytic `-2π·freq·τ` shift to ~1e-14.

### Noise: why a pre-generated tile bank, not live per-tick generation

The DFT of i.i.d. complex Gaussian noise is itself i.i.d. complex Gaussian
(unitary transform), so per-tick noise could in principle be generated
directly at (sample, channel) resolution with no FFT at all — but at full
channel counts that's still O(channels) of live Box-Muller work per tick,
and an earlier version of this codebase needed ~24-48 threads to clear
budget doing it that way.

The strategy adopted instead: `n_tiles` tiles of `tile_n_samples` are
generated once at construction; each tick does an O(1) per-(station, pol)
index draw plus a memcopy. Noise never enters a delay pipeline, matching
the physical requirement described above.

Filling the bank was originally a hand-rolled splitmix64 hash + Box-Muller
kernel under numba, because live PER-TICK generation under numba needed it
(numba's nopython mode doesn't support numpy's own Philox/PCG64 — see bug
#7 below). Once noise generation moved to "generate once, replay per tick",
that constraint stopped applying to the *build* step, and `fill_noise_bank`
was rewritten to use plain `numpy.random.Generator` parallelized across a
`ThreadPoolExecutor` over independent `numpy.random.SeedSequence` children
(numpy's Generator releases the GIL during generation, so this is genuine
multi-core speedup with no custom numerical code). Measured faster than the
numba version it replaced: 0.67s vs. numba's 2.2s at n_tiles=256, 2.5s vs.
8.8s at n_tiles=1024 (both pols, 448 channels — see the note on the
448-channel figures being superseded, under "SPS-CBF ICD corrections"
below). An optional numba-parallel per-tick memcopy (`_copy_tile_into`) was
deleted outright as dead weight — it was never actually enabled in
production or benchmarks.

A real memory bug was caught building a realistically-sized bank (bug #16
below): see "Known bugs."

**Correctness constraint, non-negotiable:** the tile index must be drawn
from each station's own (station, pol) noise seed — never from a value
shared across stations. Sharing it would make every station emit
byte-identical "noise" for a given tick, silently breaking any test that
depends on receiver noise being uncorrelated across the array.

**The real cost of this approach is fidelity, not CPU or memory, and no
bank size that fits in memory fixes it.** By the birthday paradox, a
station's own tile-index sequence hits its first repeat after
~1.25×√n_tiles ticks. Measured on the 2-socket EPYC 9254 target hardware,
448 channels, 8 threads:

| n_tiles | bank size (both pols) | build time (numba, historical) | build time (numpy Generator) | first repeat (~ticks) | ~scan time |
|---|---|---|---|---|---|
| 256 | 7.0 GB | 2.2s | 0.67s | 20 | 52ms |
| 512 | 15.0 GB | — | 1.29s | 28 | 73ms |
| 1024 | 28.0 GB | 8.8s | 2.54s | 40 | 105ms |
| 2048 | 56.0 GB | 31.9s | not re-measured | 57 | 148ms |

Going another order of magnitude in `n_tiles` would still only push the
first repeat into the low-single-digit seconds — nowhere near long enough
for a test that checks a single station's long-integration noise-floor
behavior. This trade is only valid if CBF's test suite doesn't rely on
long-integration per-station noise-floor accuracy; per-tick and
cross-station statistics (delay-tracking, correlation, beamforming
functional tests) are unaffected by one station's own periodicity.

**CPU cost turned out to be a non-issue.** Once the bank exists, per-tick
cost is an index hash plus a memcopy: at 448 channels, plain `out[:] =
bank[idx]` measured 1.51ms/tick (57.5% of the then-current 2.621ms budget)
*regardless of thread count* — even 1 core was enough. This is why the
target of ~8 CPU cores/pod turned out to be over-provisioned for this
approach's steady-state cost; the actual resource question this approach
shifts onto is memory (linear in `n_tiles`) and one-time startup latency
(build time, see above — comfortably inside a 10s construction target even
at n_tiles=1024).

### Pulsed (pulsar) sources: three design iterations

A pulsar is genuinely periodic, so "generate once, replay per tick" is a
better fit here than for noise — a real pulsar's profile repeats exactly
every rotation period; there's no birthday-paradox tradeoff or
long-integration correctness caveat, since periodicity here is ground
truth, not an artifact.

This module went through three designs before arriving at the current one
(`direct_synthesis.py`'s `build_pulsar_template`/`add_pulsar_tick`). Each
wrong version looked reasonable until it was actually built and checked:

- **v1 (wrong)**: each channel sees one constant DM delay (evaluated at
  that channel's center frequency only), applied to a real-valued
  achromatic envelope; per-tick geometric delay via a first-order Taylor
  correction (`envelope(t-tau) ≈ envelope(t) - tau*envelope'(t)`).
- **v2 (fixed intra-channel smear, still broken for beamforming)**: a
  channel isn't one frequency, it's a ~781kHz passband — at SKA-Low
  frequencies, even DM=2 pc/cm³ smears the dispersion curve across
  ~81,000 channel-widths at the bottom of the band (50MHz) and ~159 at the
  top (400MHz), a real low-frequency-radio effect. v2 averaged many
  shifted copies of the profile across each channel's own passband
  (`n_subfreq=1`, the broken version, predicted a channel peak of 1.0 with
  no smearing at all; the converged answer was 0.288 — a 3.5x error). But
  v2 was still real-valued (no carrier): CBF's beamformer coherently
  combines stations by phase-rotating already-channelized complex data,
  which is only physically valid because a real channelizer inherently
  produces complex output with genuine carrier phase. Real-valued content
  has no phase for that rotation to act on, so v2 could not be coherently
  beamformed across stations at all.
- **v3 (current)**: generate the wideband, undispersed pulse train as one
  real time series spanning the whole band (a shared "sky carrier"), apply
  the standard coherent-dispersion transfer function (Lorimer & Kramer
  2006, eq. 5.21) directly to its full complex FFT, then channelize via a
  one-time, offline `_channelize_once` call. This fixes both v1/v2 problems
  at once: intra-channel smear falls out correctly as an emergent property
  of dispersing at full wideband FFT resolution before channelizing, and
  channelizing a real signal via FFT inherently produces genuinely complex
  per-channel content with real carrier phase, which lets the per-tick
  geometric-delay correction use tone's exact phase trick instead of v1's
  Taylor approximation.

**Verified against an external, peer-reviewed reference.** NANOGrav's
`PsrSigSim` package was investigated directly, but its dependency chain is
heavy and partially broken for this purpose (pulls in PINT, `fitsio`,
`emcee`, `nestle`, matplotlib just to import; its own
`BasebandSignal.to_RF`/`to_FilterBank` conversion methods are unimplemented
stubs). Instead of depending on it, its `ISM._disperse_baseband`
implementation was read directly and its physics reimplemented in this
module. Its dispersion constant (`DM_K = 1/2.41e-4 = 4149.38`) matches this
module's own (`4148.808`) to 0.014%.

**Two real bugs caught only by numerical verification, not by
inspection:**

1. The wideband dispersion step originally used `np.fft.rfft`/`irfft`
   (correct for PsrSigSim's own real-ADC-sampled convention), but this
   codebase's own convention treats negative `fftfreq` bins as meaningful,
   independent channels — using `rfft` silently covered only half the
   intended band and shifted every channel's frequency. Caught by
   injecting a known tone at a known external channel and checking where
   it actually landed (it was off), not by reasoning about the
   convention. Fixed by using full complex `fft`/`ifft` throughout.
2. The first version of the per-tick exact-phase correction called
   `cos`/`sin` once per (sample, channel) pair (~1.8M transcendental calls
   per tick at 448 channels), needing 32+ threads to clear budget, 4x
   worse than expected. Fixed with a phase-accumulator (NCO-style)
   recurrence: two trig calls per sample, then one complex multiply per
   channel to step the rotation forward.

**Cross-station coherence, verified numerically**: two streamers with
different `DelayPolynomial`s (simulating two stations), same pulsar, same
tick — after each applies its own delay-compensating phase correction
(what a beamformer does), their content's normalized correlation measured
1.0000, confirming v3 (unlike v1/v2) actually supports coherent
multi-station beamforming.

**Why the "sky carrier" is shared across stations while noise is
not:** every station in a real array observes the same wavefront from the
same source, arriving at a different time only because of geometry — the
entire physical basis of interferometry. So the wideband pulse train's
random carrier is generated from one fixed seed shared by every station
simulating a given pulsar, never from `station.station_id`. Receiver noise
is the opposite: independently seeded per station, because each station's
receiver is a physically separate noise source.

**Resource cost** (benchmarked on the EPYC target, 448 channels, DM=2,
100ms period, 5ms FWHM — see "SPS-CBF ICD corrections" for why these
448-channel/pre-oversampling-correction figures are since superseded):
one-time build cost (wideband generate + FFT dispersion + channelize) was
originally all custom numba, replaced with fully vectorized numpy
(measured faster at every period tried, e.g. 6.2s vs. numba's 9.6s at a 1s
period). `scipy.fft` (with `workers=`) was adopted where it helped:
`_channelize_once`'s batched small FFTs saw a ~4-5x win; the two big 1D
FFT/IFFT calls in `build_pulsar_template` only ~15-20% (bandwidth-bound at
these sizes). `scipy` was reinstated in `pyproject.toml` for this reason,
independent of the earlier removal.

**A previously-undocumented finding, more consequential than the
numba/numpy/scipy choice: FFT size matters far more.** `n_wide` (the
wideband array length dispersion FFTs run over) is `num_channels ×
round(period_s × channel_width_hz)`. Some periods factor into small primes
(fast) and some land on a size with a large prime factor (e.g. one 50ms
case had a largest prime factor of 19531, and measured 3-6x slower than
neighboring, better-factored periods). `scipy.fft.next_fast_len()` can find
a nearby fast size, but this was never implemented as a fix: (a) its result
isn't guaranteed to remain an exact multiple of `num_channels`, which
`_channelize_once`'s reshape requires, and (b) padding the wideband array
changes what "one period" means to the FFT's implicit circular
convolution, which would need re-validating against the
coherence/channel-mapping checks. Left as a real design task for a future
session if multi-second pulsar periods become a requirement. See the
"CORRECTED pulsar construction budget" subsection below for how this
finding got *worse*, not better, once the real oversampled sample rate was
corrected.

**Still not modeled**: pulse-to-pulse jitter, scintillation, nulling,
profile evolution with frequency, or realistic flux/SNR calibration
against the receiver noise floor. The last of these matters specifically
if the goal is testing whether PSS/PST can actually detect the injected
pulsar as a candidate, since detection significance depends on
signal-to-noise, not just having the right shape.

### Per-source delay: from one shared polynomial to a required `DelayFeed` per source

Every source used to share ONE station-level `DelayPolynomial`, meaning two
tones or two pulsars in the same streamer were implicitly at the same point
in the sky — wrong the moment a test wants two sources at genuinely
different directions in the same station. Fixed by giving every
tone/pulsar its own `common.DelayFeed`, which needed no changes to the
synthesis kernels at all (`synth_tone_channel`/`add_pulsar_tick` already
took delay arguments as plain parameters, not something baked into the
streamer — confirmed by direct measurement that alternating between two
different delay arrays across calls costs the same as reusing one, within
noise).

`delay_feed` is required on every tone/pulsed `source_cfg` — there is no
default/fallback delay; `DirectSynthesisStreamer.__init__` raises
immediately if one is missing. This is deliberate: a source with no real
delay path would silently apply zero delay, producing content that's
trivially "perfectly aligned" — exactly the kind of thing that could mask a
real CBF delay-tracking bug, given this simulator's whole reason for
existing is generating true delay independently of CBF. An earlier design
had a `PolledDelayFeed` fallback for convenience; removed once it became
clear a silent implicit zero-delay default fights the simulator's own
purpose more than it helps.

### `source_cfgs`/`noise_cfg`: from dicts to typed dataclasses

`DirectSynthesisStreamer.__init__` used to take `source_cfgs: list[dict]`
and `noise_cfg: dict | None`, validated by hand at construction time (kind
in `("tone", "pulsed")`, `delay_feed` present, pulsar's `pulsar_name` XOR
`period_s`/`width_s`/`dm_pc_cm3`). Replaced with explicit dataclasses:
`ToneSourceConfig`, `PulsarByNameConfig`, `PulsarByParamsConfig`,
`NoiseConfig`. The two pulsar variants are deliberately separate types
(not one dataclass with optional fields), turning the old runtime
dict-shape check into a structural property of which type a caller
constructs. The mutual-exclusion/kind-dispatch validation moved to
`simulator.build_source_cfg`, a standalone function that turns one raw JSON
`source_cfgs` entry plus its already-resolved `DelayFeed` into the right
dataclass — a deliberate "validate at the boundary" move, since
`StartScan`'s JSON argument is the actual untyped-data entry point into
this system.

### Pulsar catalog: pre-generated templates loaded by name

`build_pulsar_template`'s one-time construction cost doesn't have to be
paid at every scan start. `pulsar_catalog.py` adds a small catalog of
pre-generated pulsar templates, built once offline by
`generate_pulsar_catalog.py` and loaded by name at
`DirectSynthesisStreamer` construction instead — essentially free startup
cost (an `np.load` plus a slice), no FFT. Both configuration styles
(catalog lookup by name, or building a custom template from
period/width/DM at construction) are supported side by side, deliberately:
fast, fixed-parameter setup for routine tests, while keeping the ability to
dial in an arbitrary period/DM for a test that specifically needs one.

Every catalog entry is generated at the full 384-channel band width,
starting at channel 64 — a station simulating a narrower sub-band just
slices the columns it needs out of the same array, so one `.npy` per named
pulsar serves every valid station configuration.

Stored as complex64 on disk (halves size vs. complex128, no meaningful
fidelity loss since content is quantized to int8 well downstream anyway),
upcast back to complex128 on load. `catalog.json` records the exact
`channel_width_hz`/`channel_output_rate`/`num_channels`/`base_freq_hz` each
entry was generated under, checked against current constants on load — a
catalog baked under since-corrected constants (see the ICD corrections
below — this project has been burned once already by a wrong
`channel_output_rate` assumption) fails loudly instead of silently
misapplying stale data.

A real, previously-unmeasured cost was discovered generating the default
catalog: three illustrative entries (10ms/89.3ms/300ms periods) came to
28MB/254MB/853MB respectively — ~1.1GB total for just three examples,
scaling roughly linearly with period. Deliberately not committed to git
(`pulsar_catalog_data/` is gitignored) — binary blobs this large would
permanently bloat repository history for every future clone. The
recommended flow is to run `generate_pulsar_catalog.py` as an OCI image
build step, not to bundle pre-generated `.npy` files into a wheel/sdist.

## SPS-CBF ICD corrections

Several channelization assumptions this codebase had been running on
turned out to be wrong once checked against the real ICD text (not a
screenshot, not carried forward from an earlier guess). All three
corrections are now reflected in `common.py`'s constants,
`direct_synthesis.py`'s validation, and `simulator.py`'s device properties
— see `README.md` for the current, correct values. The corrections
themselves:

1. **Maximum channel count is 384 (300MHz), not 448 (350MHz).** The ICD:
   "The bandwidth is channelized as 384 equispaced coarse channels" and
   "the number of frequency channels assigned to the beams is configurable
   from 8 to 384 in steps of 8." 448 channels was used throughout earlier
   sessions as "the full SKA-Low band" and benchmarked extensively at that
   size — that configuration was never valid on real hardware. Every "448
   channels" figure elsewhere in this history document is left as a record
   of what was actually measured at the time, not retroactively rewritten,
   but should not be read as a currently-valid configuration.
2. **The lowest channel is global ID 64, not 65.** The ICD: "the lowest
   frequency channel is channel 64, centre frequency 50MHz." 64 ×
   `CHANNEL_WIDTH_HZ` (781,250 Hz) = 50,000,000 Hz exactly, confirming
   `BASE_FREQ_HZ` is meant as a channel *centre* frequency (matching how
   it's used everywhere in this codebase, e.g. `channel_center =
   base_freq_hz + channel_idx*channel_width_hz`). The previous value
   (65/50.78125MHz) was off by one channel. The band's actual lower *edge*
   (distinct from channel 64's centre) is `50e6 - CHANNEL_WIDTH_HZ/2` =
   49,609,375 Hz, matching the ICD's stated "lower edge of this channel is
   49.61MHz" — nothing in this codebase needs that edge value directly,
   since every per-channel frequency calculation here is already expressed
   in channel centres.
3. **The real per-channel sample rate is oversampled by 32/27, not
   critically sampled.** The ICD: "Each channel of data from SPS to Low
   CBF has a sampling period of 1080 ns (1.25 ns per sample (ADC sample
   rate) x 1024 samples x 27/32, wherein 32/27 is the oversampling factor
   of the filterbank)." This codebase had been assuming
   `channel_output_rate = CHANNEL_WIDTH_HZ` (a 1280ns sample period —
   what a critically sampled channelizer would give), missing the
   oversampling factor entirely. `CHANNEL_WIDTH_HZ` itself (781.25kHz)
   remains correct as the frequency-domain channel *spacing* — what was
   wrong was treating that same number as the time-domain per-channel
   sample *rate* too. This is a numerically significant fix, not just a
   naming correction: it shrinks `BLOCK_DURATION_S` (the per-tick
   real-time budget every benchmark number is measured against) from
   2.621ms to 2.21184ms — about 15.6% tighter than what this codebase had
   been budgeting against. Confirmed compatible with the ICD's separate
   packet-size constraint ("the number of samples per packet shall be a
   multiple of the numerator of the oversampling ratio," i.e. a multiple
   of 32): `HEAP_LEN` (2048) / 32 = 64 exactly.

A related ICD passage was reasoned through but never independently
verified against real hardware: "The oversampled filterbank data shall be
derotated as well as ensuring the first sample of every SPS SPEAD packet
has zero phase." This describes a raw-PFB-output correction step this
codebase never needs, because `synth_tone_channel`/`add_pulsar_tick`
synthesize the already-clean, already-derotated baseband result directly
rather than generating the raw oversampled artifact and correcting it
afterward. The "zero phase at packet start" convention is read as a
normalization reference point (for a hypothetical zero-residual-frequency
signal), which this module's phase formula already satisfies trivially —
not as a literal per-heap phase reset for every source regardless of
residual frequency, which would destroy the cross-heap phase continuity
CBF's own delay-tracking and coherent beamforming need to recover. Revisit
if a delay-tracking test ever shows a phase discontinuity at heap
boundaries.

### Heap structure and the `channel_info`/`antenna_info` bit layout

One heap = one channel = 2048 consecutive time-domain samples, both
polarisations interleaved per sample (Vreal, Vimag, Hreal, Himag, each
int8). `pkt_len = 0x2000` (8192 bytes) confirms this: 2048 × 4 bytes.

The bit-field layout for `channel_info`/`antenna_info` used to be read off
a screenshot of the ICD diagram, flagged for a long time as the project's
top unverified risk. It was later confirmed against the real ICD text
directly: six items total, each a SPEAD-64-48 item pointer (48-bit value
field) — see `README.md` for the current table. No 7th "payload" item
pointer; CBF firmware reads the payload at a fixed byte offset rather
than doing a generic SPEAD parse. `pack_channel_info`/`pack_antenna_info`'s
bit widths matched this exactly, bit for bit, once confirmed.

(Every item is IMMEDIATE mode except `0x3300 payload_offset`, which is
ADDRESS mode — this was initially gotten wrong on the Go side too, see
"Known bugs (fixed)" #20 below for how a real CNIC reference capture
caught it.)

`BASE_FREQ_HZ` itself went through two corrections in sequence: first from
an initial `0.0` placeholder to 50.78125MHz (coarse channel 65) — this
fixed a real, previously-latent gap, since `simulator.py`'s `StartScan`
never overrode `base_freq_hz`, so any pulsed source configured through it
would have hit `DirectSynthesisStreamer`'s `base_freq_hz > 0` check and
failed outright while it was still `0.0`. Then, once the channel-64-vs-65
error was found (see above), to the current 50.0 MHz exactly.

### Why `spead.SpsPacketizer` doesn't use spead2

A real architectural problem surfaced testing the SPEAD encoding path
end-to-end: every heap came out with 8 items instead of the ICD's 6.
Checked directly against spead2==4.4.1's C++ source
(`send_packet.cpp::packet_generator::next_packet`), not just its Python
API: spead2's packet encoder unconditionally writes 4 reserved item
pointers (`HEAP_CNT`, `HEAP_LENGTH`, `PAYLOAD_OFFSET`, `PAYLOAD_LENGTH`) at
the start of every packet it ever emits, with no flag, `StreamConfig`
option, or `Heap` method to suppress any of them. Since CBF's ICD heap has
only 6 items total — fewer than spead2's own mandatory minimum of 4
reserved + any real items — spead2 cannot produce a compliant packet no
matter how it's configured. A firmware receiver that addresses payload
bytes directly at fixed offsets (rather than doing a generic SPEAD parse)
has no way to skip items it doesn't expect, so this isn't cosmetic.

Fixed by replacing spead2 with a hand-rolled SPEAD-64-48 encoder scoped to
exactly the 6 ICD items — the wire format itself (8-byte header, N 8-byte
item pointers, then payload) matches real SPEAD, confirmed against
spead2's own source; what differs is only which items get written.
`spead2` was dropped from `pyproject.toml` entirely. Verified two ways: an
item-by-item pytest check that parses the encoded bytes back into (ID,
value) pairs independent of the encoder's own logic, and a manual
byte-by-byte decode of a real generated pcap's hex dump against the ICD
table (every field matched, including packed `beam_id`/`frequency_id` and
station fields).

## Known bugs (fixed)

Numbered as they were found, kept for the lessons even where the buggy
code itself has since been deleted.

1. **V/H noise sharing**: one `NoiseSource` reused for both pols gave
   numerically identical noise (a deterministic pure function of inputs).
   Fixed: separate seeds per pol.
2. **Absolute-epoch-time precision collapse (delay polynomial)**:
   evaluating a 5th-order polynomial against raw Unix-epoch-scale `t`
   (~1.8×10⁹) blows up float64 precision. Fixed: evaluate relative to
   `start_validity_sec`.
3. **Same bug, tone phase**: absolute `t` fed directly into phase
   computation collapsed float64 precision (~0.1-0.5 rad error). Fixed:
   all generation takes `t_rel = t - obs_time_ref`, bounded by scan
   duration. This is why every kernel in this codebase takes a
   small-magnitude relative time, never raw epoch time.
4. *(deleted-file history, legacy wideband path)* Ring buffer `np.roll` on
   the whole buffer every tick — O(capacity) regardless of new-sample
   count. Fixed there with a real O(1) circular buffer; moot now that
   direct synthesis never buffers.
5. *(deleted-file history, legacy wideband path)* Channelizer looping
   `np.fft.fft()` per output row — thousands of individual FFT calls/tick.
   Fixed there with `sliding_window_view` + one batched FFT call; moot now
   that per-tick FFT channelization doesn't happen anywhere in this
   codebase (the pulsar's one FFT call is a one-time, offline construction
   step).
6. **`ThreadPoolExecutor`-chunked generation plateaus at ~4-8 workers**,
   confirmed on 3 different CPU architectures (Apple M5, Intel Xeon Silver
   4410T, AMD EPYC 9254/7443) regardless of core count — per-task
   Python/GIL overhead, not compute/bandwidth. Fixed: moved to
   `numba`+`prange`, near-linear scaling well past that plateau. General
   lesson: per-task Python overhead, not compute, caps naive threading.
7. **Numba's `nopython` mode doesn't support `numpy.random.Philox`**
   (confirmed by direct test). Replaced with a from-scratch splitmix64
   hash + Box-Muller. Statistically correct, deterministic, seekable, but
   not bit-identical to Philox. The splitmix64 hash was deliberately kept
   as a hand-rolled `@njit` function even after `Generator(Philox(...))`'s
   `counter=` constructor was found to support the same "pure function of
   an arbitrary index" property this needs: benchmarked at ~12.7us/call
   for a fresh Philox+Generator construction vs. ~0.13us/call for the
   numba hash (~100x slower — object-construction overhead). The `@njit`
   itself measured ~25x faster than the equivalent plain-Python/numpy-
   scalar version (0.13us vs. 3.3us/call).
8. *(deleted-file history, legacy wideband path)* `fftshift` was a
   full-array reorder every tick for no numerical reason — replaced there
   with natural FFT bin order + a fixed permutation applied only to small
   `channel_id` integers. The successor of that permutation logic is used
   only in the pulsar template's one-time, offline construction.
9. *(deleted-file history, legacy wideband path)* `OMP_NUM_THREADS=1` cap
   — originally needed to stop numpy/BLAS oversubscribing against
   `ThreadPoolExecutor`-chunked generation; confirmed
   no-longer-necessary-but-harmless once that path moved to numba. Never
   needed in `direct_synthesis.py`.
10. *(deleted-file history, legacy wideband path)* `numpy.fft.fft` has no
    multi-threading at any array size — the legacy channelizer switched to
    `scipy.fft.fft` (`workers=`) to work around this. Moot after that file
    was deleted along with `scipy` (later reinstated for the pulsar's own,
    unrelated reason).
11. **Benchmark-script bug** (not production code): a sweep loop left a
    tuning parameter at its last-tried value instead of the sweep's best.
    General lesson: don't trust a single-pass sweep result on this class
    of hardware without a 5+ repeat repeatability check.
12. *(deleted-file history, legacy wideband path)* Receiver noise
    incorrectly delay-corrected (`shared` and `noise` summed before the
    delay/channelization pipeline). `DirectSynthesisStreamer` was already
    immune by construction.
13. **`DirectSynthesisStreamer.generate_next_tick` allocated a fresh
    14.7MB complex128 array plus another fresh array from noise
    generation plus a separate full-array `+=`, twice per tick** — this
    allocation/copy churn, not Box-Muller math, was the dominant per-tick
    cost at high channel counts (~30ms of a ~33ms tick at 448 channels).
    Misdiagnosed at first as a compute/thread-dispatch problem — see the
    "Target server results" investigation below. Fixed: noise is written
    directly into a persistent, reused per-pol buffer; tone then adds on
    top. Verified bit-identical to the old path for the same seed/index.
    Safe to mutate in place because `ScanRunner._run` calls
    `generate_next_tick` synchronously and `HeapAccumulator.add` copies
    the data out before the next tick runs.
14. **Pulsar dispersion used `np.fft.rfft`/`irfft` instead of full complex
    `fft`/`ifft`** — see the pulsar section above.
15. **The pulsar's per-tick geometric-delay phase correction called
    `cos`/`sin` once per (sample, channel) pair** — see the pulsar section
    above.
16. **`fill_noise_bank`'s first version returned one freshly allocated
    array per worker thread and `np.concatenate`'d them** — peaked at
    several times the bank's own memory footprint and OOM-killed the
    process building a realistically sized bank (n_tiles=1024, 448
    channels). On a shared node, this took other tenants' pods down with
    it (`oom.group` kill), not just the simulator's own process. Caught
    only by actually running it at realistic size under
    `/usr/bin/time -v`, not by reasoning about the code. Fixed by having
    each worker write directly into its slice of one preallocated bank
    array, one tile at a time. A first fix attempt tried
    `numpy.random.Generator`'s `out=` parameter directly against
    `bank[start:end].real`/`.imag`, which fails outright (`out=` requires
    a C-contiguous target, and `.real`/`.imag` views of a complex array
    are strided) — caught immediately as a clear error, not a silent one.
    Plain assignment (`bank[i].real = arr`) does accept a strided target.
    Verified: peak RSS matches the bank's own size almost exactly.
17. **`SpsPacketizer.send_channel_heap`'s `heap_counter` formula multiplied
    `unix_to_tai2000_seconds(...)` by `CHANNEL_WIDTH_HZ` (the sample rate)
    instead of dividing by `BLOCK_DURATION_S` (the correct per-heap
    rate)** — inflated the value by `HEAP_LEN` (2048x). For any
    current-era timestamp this overflows the ICD's 40-bit `heap_counter`
    field, meaning every real `send_channel_heap()` call would have failed
    outright. This had silently never been caught because nothing
    exercised `send_channel_heap()` end-to-end before
    `generate_test_pcap.py` tried to actually send a heap through it for
    the first time. Fixed: `heap_counter = unix_to_tai2000_seconds(t) /
    BLOCK_DURATION_S`, comfortably within 40 bits until roughly year 2091
    — `encode_channel_heap` now raises `ValueError` outright if a future
    change ever pushes it out of range again, rather than silently
    truncating.
18. **spead2 cannot produce CBF's real heap format at all** — see "Why
    `spead.SpsPacketizer` doesn't use spead2" above.
19. **Three channelization assumptions were wrong** — see "SPS-CBF ICD
    corrections" above.
20. **`writeSpeadItemPointer` (Go) hardcoded every one of the 6 item
    pointers as IMMEDIATE mode**, including `0x3300 payload_offset`. Found
    by building `cmd/pcap-dump` (a small CLI that writes generated heaps
    straight to a pcap file, no network socket needed — see `README.md`'s
    "Inspecting SPEAD structure") and comparing its output byte-for-byte
    against a real CNIC reference capture (`source.pcap`, captured
    separately, unrelated content/parameters — only header/framing
    structure was compared): every one of 34,257 real `0x3300` item
    pointers in the reference capture had its mode bit clear (ADDRESS),
    while all of ours had it set (IMMEDIATE) — 100% systematic, not noise.
    `payload_offset` addresses the payload rather than carrying a scalar
    value of its own, so ADDRESS mode is the correct choice regardless of
    the fact that its value is always `0x0` (the payload always starts
    immediately after the last item pointer — see "Why `spead.SpsPacketizer`
    doesn't use spead2" above for the layout this depends on). Every other
    item matched the reference capture exactly already, including header
    shape (magic/version/id_bytes/addr_bytes), item order, and
    `0x3001`'s full-width bit packing. Fixed: `writeSpeadItemPointer` now
    takes an `immediate bool`; only `0x3300` passes `false`.

## Python/Tango benchmarking history

Per-tick budget was originally computed as `HEAP_LEN / CHANNEL_WIDTH_HZ` =
2048 / 781250 Hz ≈ 2.621ms (`common.BLOCK_DURATION_S`); this was later
corrected to 2.21184ms once the real oversampled `channel_output_rate` was
confirmed (see "SPS-CBF ICD corrections"). Work scales with channel count;
the budget doesn't. The project's own comfort bar was ~80% of budget;
95-110% was treated as not good enough as a baseline.

### Historical laptop results (tone+noise only, pre-convergence)

Machine: 10-core Apple Silicon (arm64), a single laptop-class machine, not
yet run on target server/Kubernetes hardware.

- 96 channels: ~1.3-1.4ms/tick at numba_threads=8-10 — ~50-55% of budget.
  Repeatability-checked (5 repeats, stdev ~0.02ms).
- 448 channels (since known invalid, see ICD corrections): ~5.2-5.4ms/tick
  even fully parallelized — ~200-210% of budget. The 96→448 scaling ratio
  was ~3.7-4.4x (close to linear), a real improvement over the legacy
  path's worse-than-linear ~4.67x+, but it did not close the gap on this
  hardware. Per-channel noise synthesis (not tone, which is O(1) and
  negligible) was the dominant, inherently O(channels) cost.

The legacy wideband `StationStreamer` (file since deleted) benchmarked on
the same machine at 96 channels: ~2.4-2.6ms/tick at numba_threads=8-10 with
FFT_WORKERS=2-8 — right around budget, not comfortably under it. It was
never re-benchmarked at 448 channels given its known worse-than-linear
scaling, and is moot now that it's deleted.

### Target server results — 2-socket AMD EPYC 9254

Real target-class K8s node hardware (2× EPYC 9254, 48c/96t total, 2 NUMA
nodes of 48 logical CPUs each). An ad hoc sweep script (channel count ×
thread count × NUMA pinning, 3 repeats/config, 15s clock-ramp burn-in
before each sweep — this machine idles at 1.5GHz and needs sustained load
before `schedutil` ramps to boost clocks) was used but not committed to the
repo.

**Initial diagnosis (wrong, kept for the record since the investigation
trail matters):** the first pass found 448 channels still not viable (best
≈33.0ms/tick, NUMA-pinned, 32 threads — ~12.6x over budget), throughput
saturating at ~24-32 threads and regressing beyond that, and concluded the
bottleneck was Box-Muller's per-sample log/sqrt/sin/cos combined with
numba parallel-dispatch fork-join overhead.

**Actual root cause: pure memory-allocation overhead, not compute** (bug
#13 above), found by isolating raw kernel cost from the full
`generate_next_tick` call and finding a ~14ms unexplained gap. After the
fix (persistent reused output buffer, `synth_noise_all_channels_into`
writing directly into it), re-swept end-to-end:

| channels | best (single NUMA node, 48 threads) | best (unpinned, 96 threads) |
|---|---|---|
| 96 | 0.54ms (20%) | 0.45ms (17%) |
| 160 | 0.75ms (29%) | 0.65ms (25%) |
| 224 | 0.97ms (37%) | 0.72ms (27%) |
| 288 | 1.19ms (45%) | 0.78ms (30%) |
| 352 | 1.42ms (54%) | 0.89ms (34%) |
| 416 | 1.69ms (65%) | 1.04ms (40%) |
| 448 | 1.84ms (70%) | 1.10ms (42%) |

(% is of the then-current 2.621ms budget, since superseded — see below.)
Every channel count cleared the 80% comfort bar with just one socket's
worth of threads (48), down from 33.0ms/1260% before the fix. This flipped
the earlier "more threads stop helping past 24-32" finding, too: with
allocation overhead gone, more threads helped monotonically again (up to
96), because per-dispatch work was now large enough relative to fork-join
overhead to benefit from more parallelism.

**Lesson for future benchmarking**: always isolate a suspiciously
expensive method into its component kernel calls vs. its full Python-level
body before trusting a "this kernel is the bottleneck" conclusion drawn
only from thread-count/channel-count sweeps. This sweep methodology was
rigorous (burn-in, repeats, NUMA control) and still pointed at the wrong
root cause, because it only ever measured the whole method, never
separated allocation from compute.

Operational implication drawn at the time: give each station pod one NUMA
node's worth of threads (48, pinned) for a comfortable margin at 448
channels, fitting 2 station pods per 96-thread physical node.

### Combined tone + tiled-noise + pulsar (after backend convergence)

The three source types were benchmarked individually elsewhere but never
together until the convergence into a single `DirectSynthesisStreamer`
made that the actual thing to benchmark. NUMA-pinned to one node (48
logical CPUs), tone + `n_tiles=256` noise bank + a 10ms/DM=2 pulsar all
active together:

| channels | best config | best mean | 8 threads |
|---|---|---|---|
| 96 | 16 threads | 0.588ms (22.4%) | 0.655ms (25.0%) |
| 448 | 24 threads | 1.884ms (71.9%) | 2.186ms (83.4%) |

(% is of the 2.621ms budget, since superseded.) 448 channels combined
cleared budget at every thread count tested from 8 threads up
(repeatability-checked, stdev 0.002ms at 448 channels). At exactly 8
threads it was just above the 80%-comfort bar (83.4%); 16+ threads got
back under 80% (74.1%). Noise tile-bank memory at these settings: 7.5GB
(both pols, `n_tiles=256`).

### Corrected combined benchmark: real 384-channel max + oversampled budget

The numbers above are superseded once the ICD corrections landed (384-
channel real maximum, `BLOCK_DURATION_S` = 2.21184ms). Same workload, same
NUMA pinning:

| channels | best config | best mean | 8 threads |
|---|---|---|---|
| 96 | 24 threads | 0.587ms (26.5%) | 0.689ms (31.2%) |
| 384 | 16 threads | 1.821ms (82.3%) | 2.034ms (92.0%) |

(% is of the corrected 2.212ms budget.) **The headline conclusion changed
for the worse.** At 384 channels — fewer channels than the old (invalid)
448-channel config — the best-case result was 82.3% of budget, outside the
80%-comfort bar (the old 448-channel measurement read 71.9% best-case,
comfortably under). This isn't a contradiction: work scales down only
~14% going from 448→384 channels, but the real budget is ~15.6% tighter
than what the old numbers were measured against — the two effects nearly
cancel, so fewer channels did not translate into more comfortable margin
once measured against the real budget. Repeatability-checked (5 repeats
at `numba_threads=16`): mean 1.863ms (84.2%), spread 1.769-2.074ms
(80.0%-93.8%) — stdev 0.130ms, meaningfully noisier than the 96-channel
case's 0.026ms, with the spread's upper end close enough to 100% that an
unlucky tick could plausibly miss budget. At exactly 8 threads (92.0%) this
is tight, not comfortable. Noise tile-bank memory at these settings:
6.44GB (both pols, `n_tiles=256`, 384 channels).

If a real deployment is hard-capped at 8 cores/pod, this configuration
needs either fewer channels, a smaller noise bank
(`n_tiles`/`tile_n_samples` tradeoff — effect size not swept), or accepting
a proportionally larger overrun-tolerance margin.

### One-time construction budget (target 10s, hard limit 30s)

Distinct from the per-tick budget above — the one-time cost of
`DirectSynthesisStreamer.__init__` before a scan can start. Measured in
isolation, NUMA-pinned, 448 channels (since superseded, see below):

| noise n_tiles | build (both pols) |
|---|---|
| 256 | 0.67s |
| 512 | 1.29s |
| 1024 | 2.54s |

| pulsar period | build, plain numpy.fft (pre-scipy) | build, scipy.fft workers=8 |
|---|---|---|
| 10ms | 0.48s | 0.32s |
| 100ms | 4.44s | 3.21s (well-factored `n_wide`) |
| 200ms | — | 6.53s (well-factored) |
| 300ms | — | 9.84s (well-factored) |
| 50ms | — | 4.73s (poorly-factored `n_wide`) |
| 1000ms | 46.17s — over the 30s hard limit | not re-measured after scipy |

Noise easily cleared the 10s target at every tested size. Run-to-run
variance on the shared host was real, not just factorization: a clean
isolated run measured 300ms/well-factored at 9.84s (just under the 10s
target); re-measured as part of the full benchmark sweep (same host, more
going on around it) at 12.5s (over it) — roughly 25% higher, plausibly
load/thermal/clock-state, not a code change.

### Corrected pulsar construction budget: real 384-channel max + oversampled rate

The table above is superseded, and the news is worse, not better.
`n_wide = num_channels * round(period_s * channel_output_rate)` depends on
both corrected quantities (384 not 448, and the real oversampled
`CHANNEL_OUTPUT_RATE_HZ` ≈ 925,925.93Hz not `CHANNEL_WIDTH_HZ` =
781,250Hz) — the resulting array lengths have noticeably worse
factorization properties across the board, not just for an occasional
unlucky period. Noise-bank fill numbers are unaffected (noise doesn't
depend on `channel_output_rate`) and still cleared the 10s target easily
at 384 channels: 0.55s/1.18s/2.22s at n_tiles=256/512/1024.

Re-swept `build_pulsar_template` at 384 channels, corrected
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
| 90ms | 31,999,872 | No | 11.21s | over 10s target |
| 100ms | 35,555,712 | No | 16.94s | over 10s target |
| 200ms | 71,111,040 | No | 10.38s | over 10s target (barely) |
| 300ms | 106,666,752 | No | 51.07s | over 30s hard limit |

Every single period tested came back poorly-factored — a direct,
structural consequence of the correction: `CHANNEL_OUTPUT_RATE_HZ` is
`CHANNEL_WIDTH_HZ * 32/27`, and that `/27` (`27 = 3^3`) means
`round(period_s * channel_output_rate)` lands on far fewer
small-prime-friendly integers than the old, cleaner `CHANNEL_WIDTH_HZ`
did. `scipy.fft.next_fast_len()`'s previously-deferred padding fix (see
the pulsar section above) became a much higher-value target after this
correction, since it would help essentially every period, not just
occasional unlucky ones.

Approximate safe limits after this correction: roughly ~80ms reliably
clears the 10s target (down from the old ~100-200ms figure — note the
non-monotonic 15ms→20ms jump, 0.71s to 3.26s, underscoring that
factorization noise dominates at these sizes too), and the 30s hard limit
is crossed somewhere between 200ms (10.38s) and 300ms (51.07s) — call it
~200ms as the practical ceiling until that range gets swept more finely.

### Historical hardware notes (legacy wideband path, superseded designs)

- **Apple M5** (4P+6E cores): `ThreadPoolExecutor` threading plateaus at
  ~2.6-2.7x speedup — the 4-P-core topology explained why `time_chunks=4`
  was the old design's sweet spot. Superseded — that design no longer
  exists; kept for the general lesson that this laptop's P-core count is
  the relevant number, not its total core count.
- **Intel Xeon Silver 4410T** (10c/20t, dual-socket, NUMA, AVX-512): same
  ~3-4x `ThreadPoolExecutor` ceiling. An AVX-512-throttling hypothesis was
  proposed but the diagnostic tool built to test it (since removed)
  averaged `/proc/cpuinfo` MHz across idle+busy cores, biasing
  low-thread-count readings down — inconclusive, never resolved either
  way.
- **AMD EPYC 9254/7443** (24-48c, NUMA): `numba`/`prange` scaled
  near-linearly well past where `ThreadPoolExecutor` plateaued — this is
  what confirmed the per-task-overhead diagnosis in bug #6.
- **NUMA pinning**: didn't change best-case mean throughput for the old
  `ThreadPoolExecutor` design, but meaningfully improved tail-latency
  stability for numba's thread pool (unpinned: stdev >10ms, spikes to
  50+ms on a ~3ms workload; pinned: no such spikes). Recommended
  regardless of mean-throughput reasoning, since tail latency is what
  matters for a per-tick real-time budget.

## Tango device → Go gRPC simulator migration

`simulator.py` used to construct a `DirectSynthesisStreamer` and
`ScanRunner` directly and run the whole producer/sender pipeline in-process
(SPEAD/UDP included). It was rewired to call a separate Go process
(`cmd/simulator`, `internal/server.Server`) over gRPC instead, using
generated stubs at
`tango/src/ska_low_station_beam_simulator/simulatorpb/`. This is the split
`api/simulator.proto`'s own doc comment already described before it was
actually wired up: a Tango device server owns Tango (device properties,
`AttributeProxy` subscriptions to CBF's delay-poly emulator), the Go
process owns signal generation and SPEAD/UDP sending, with no Tango access
of its own.

**Prototype scope at the time of migration, inherited from the Go side**:
the Go backend implemented tone + noise only, no pulsar.
`simulator.build_tone_source_request` raised `ValueError` immediately for
any `source_cfgs` entry with `kind != "tone"`, rather than silently
dropping it or attempting a local fallback — there was no local generation
path left in the device at all. `direct_synthesis.py`'s own pulsar support
remained untouched and still directly exercised by
`benchmark_direct_synthesis.py`, `generate_test_pcap.py`, and
`tests/test_direct_synthesis.py` — only the Tango-facing device stopped
driving it for a real scan. See `README.md` for whether pulsar support has
since been added to the Go side.

**`source_id` reuses `delay_attr_uri` verbatim.** The gRPC
`PushDelayUpdate` RPC routes an update to the right source by `source_id`,
a concept the old local-generation flow never needed. Since every source
already requires a unique `delay_attr_uri`, `simulator.py` reuses that URI
string as the source's `source_id` on both ends of the wire instead of
inventing a new required key in the `source_cfgs` JSON schema. The Go
server independently rejects a duplicate `source_id` within one scan,
which doubles as a duplicate-`delay_attr_uri` check.

**`StartScan`'s check-then-act race**: before touching any subscription,
`StartScan` calls `GetStatus` and raises if a scan is already reported
running — this runs before tearing down the previous scan's delay
subscriptions, specifically so a rejected `StartScan` never destroys the
actually-running scan's ability to receive delay updates. A concurrent
second `StartScan` call between that check and the real gRPC call could
still slip through (the check and the act aren't atomic); the real
backstop is the Go server's own `FailedPrecondition` rejection in
`StartScan` — the same non-atomicity the old local-generation flow already
tolerated.

**New Tango attributes added at the same time**: `drift_seconds` and
`tick_number`, both sourced from `ScanRunner` on the Go side
(`DriftSeconds()`/`TickNumber()` reading two atomics — `driftBits` via
`math.Float64bits`/`Float64frombits` since Go has no atomic float64 type,
and `tick`, an `atomic.Int64` — updated once per tick from the scan loop).
`drift_seconds` is wall-clock time minus that tick's target time at the
most recently produced tick; positive means the producer is running behind
its real-time pacing schedule. `simulator.py` exposes both as read-only
Tango attributes alongside `queue_depth`, each issuing its own `GetStatus`
RPC on read — three independent round-trips per polling cycle rather than
one cached call, deliberately kept simple; revisit if that's ever shown to
matter.

Verified against a live Go process (not just unit-level mocks): a locally
built `cmd/simulator` binary was driven directly with the generated Python
stub (`StartScan`, `PushDelayUpdate`, `GetStatus` mid-scan, `StopScan`) —
confirmed `drift_seconds`/`tick_number` populate with real, sane values
mid-scan and both return to their zero defaults once `StopScan` completes.

### Removal of the Python generation core

Once `simulator.py` no longer touched `direct_synthesis.py`/`common.py`'s
`ScanRunner`/`spead.py` at all (the gRPC migration above), that code —
along with the tools built on it (`benchmark_direct_synthesis.py`,
`generate_test_pcap.py`, the pulsar catalog builder/loader, and their
tests) — was deleted outright, including the pulsar generation support
described earlier in this document, even though pulsar had not been
ported to Go. `common.py` was cut down to just what `simulator.py` still
needed: logging setup and `parse_delay_polynomial_from_attr_value`/
`DelayPolynomial`. `astropy`, `numba`, and `scipy` were dropped from
`pyproject.toml` as a result (nothing left in the Python package used
them). This means, as of this point, pulsar/pulsed-source generation has
no implementation anywhere in the codebase — see `README.md`'s "Known
limitations" for the current state.

## Go simulator: real-hardware profiling and pacing investigation

This is a running log of the Go port's real-hardware pacing investigation.
A wrong turn that was tried and measured is recorded as a valuable finding
in its own right, not deleted once superseded.

**Hardware**: an SR-IOV VF on a Mellanox ConnectX-6 (100G NIC), MTU 9000,
on a real (non-laptop) target-class Linux box — a 2-socket AMD EPYC per
the Python-side target-server description above. `noise-stream
-cpuprofile <file>` plus `go tool pprof -top`/`-list`/`-peek <file>` was
the workflow that found every real cost below; a dev laptop (a 10-core
Apple M5) was useful for correctness and directional checks but is not
representative of the target machine's core count or memory-bandwidth
profile — every number below is from the real hardware unless explicitly
marked otherwise.

### Timeline

1. **Outbound send-queue saturation.** Initial testing found the send
   queue filling up and dropping packets, with receive-side throughput
   around 15MiB/s despite the 100G link. Root cause: one UDP socket
   sending one heap at a time. Fixed with batched sends
   (`golang.org/x/net/ipv4.PacketConn.WriteBatch`, which uses
   `sendmmsg(2)` on Linux) and N parallel sender sockets
   (`spead.SenderPool`, each its own source port — spreads outbound
   traffic across the NIC's/receiver's RSS flow hash instead of pinning
   everything to one queue), plus `SO_SNDBUF` sizing
   (`-udp-send-buffer-bytes`, default 8MiB vs. the OS's often-~208KB
   default). This resolved the send-side problem completely and was never
   revisited.

2. **Producer falling behind pacing.** With sending fixed, generation
   itself couldn't keep up — worse at 384 channels than 96, drift
   climbing without bound (e.g. 113s at tick 8499 in one early run). Six
   real fixes were needed, each found by profiling the actual bottleneck
   rather than assuming one, in this order:

   a. **`HeapAccumulator` allocation/access-pattern bug**: a transpose
      loop making 768 small allocations/tick at 384 channels, exceeding
      the entire per-tick budget on its own. Fixed with a layout change
      plus reusing a flat buffer instead of allocating per element.
   b. **Row-major → channel-major generation layout**: the streamer's
      output changed from sample-major to channel-major (one channel's
      samples contiguous), eliminating a per-tick transpose in
      `HeapAccumulator` entirely.
   c. **SPEAD encoding allocation elimination**: `spead.BatchSendLoop`
      reuses a per-goroutine pool of pre-allocated wire-size buffers
      instead of allocating a fresh buffer per heap — safe because a UDP
      send copies the buffer into the kernel synchronously before
      returning.
   d. **Producer parallelization**: profiling found `ScanRunner`'s single
      producer goroutine (generation + accumulation) running at ~99% duty
      cycle on one core for the whole scan, unlike sending (already
      parallelized). Both the streamer's noise fill and
      `HeapAccumulator.Add`/`PopReadyHeaps` were split across goroutines
      by channel range, with a `GOMAXPROCS`-based default worker count.
   e. **Negative result — raising the worker-count cap did not help**: a
      flat cap of 16 workers (copied from the noise bank's own one-time
      construction cap) was removed after a profile's average concurrency
      landed at ~15.18 — suspiciously exactly that cap. Re-profiling
      showed average concurrency rise to ~19.79 (+30%), total `memmove`
      CPU-seconds rise proportionally, and wall-clock drift stay
      completely unchanged (12.016s → 12.3s at the same tick). More
      threads were just doing more of the same redundant work in
      parallel — a bulk memory copy is bandwidth-bound, not
      thread-starved, matching the Python side's own noise-tile-bank
      finding ("regardless of core count, even 1 core is enough").
   f. **The real fix: eliminating redundant copies, not redistributing
      them.** Three layers, each found by re-profiling after the previous
      fix:
      - `HeapAccumulator.PopReadyHeaps` was copying every channel's
        buffer into a second, freshly-allocated flat buffer before
        handing it to the sender. Removed via a zero-copy reslice/handoff
        of the buffer `Add` already built.
      - Generation still wrote into its own scratch buffer, which
        `HeapAccumulator.Add`'s `append` then copied again into its own
        per-channel storage. Removed by changing the streamer interface
        so generation writes directly into `HeapAccumulator`-owned
        buffers, cutting the remaining copy in half again (confirmed:
        `runtime.memmove`'s share of total CPU dropped from ~57% to ~47%,
        then ~25% after the next fix below).
      - Once that copy was gone, `runtime.memclrNoHeapPointers` —
        `make()`'s mandatory zero-fill on every freshly-grown per-channel
        buffer, every tick — became the next-largest cost (~21% of all
        CPU time), even though that memory was about to be fully
        overwritten by the noise fill a moment later. Fixed with a
        `sync.Pool` of reusable per-channel buffers, released once a
        heap's samples are read for the last time.
   g. **Result**: 384 channels went from drift climbing without bound
      (10+ seconds over a ~15s window) to only occasional, self-recovering
      drift under 100ms.

3. **`-sender-goroutines`' flat default (4) was outgrown twice** during
   this investigation (4 → 8 → 16) as the producer stopped being the
   bottleneck and the send side had to absorb much higher sustained
   throughput. Replaced with a scaling default based on channel count
   (16 senders at the full 384-channel band, proportionally fewer for a
   narrower configuration).

### Margin was thin even after all of the above

A rough estimate from the final 384-channel profile (total CPU-seconds ÷
average concurrency) put the average per-tick cost at roughly 100-101% of
the 2.21184ms budget — consistent with the observed "occasional,
self-recovering drift under 100ms": that pattern is what running right at
the edge looks like, not a comfortable cushion.

Two dominant remaining costs, both scaling with `numChannels × HeapLen`,
not source count:
- **~45%**: the noise-tile-bank copy itself — an inherent, irreducible
  per-tick memcpy at that byte width.
- **~46%**: SPEAD quantize + encode + send, dominated by scalar
  `math.Round`/clamp work plus network syscalls.

**Tone's real cost, confirmed negligible for a single source.** Profiled
with one tone source at 384 channels: tone synthesis didn't even appear in
the profile's top ~95% of CPU time. Matches the analytical prediction:
tone injection is O(`nSamples`) per source, negligible next to the ~45%/
~46% costs above, which scale with `numChannels`. Caveat, not yet tested:
tone injection runs sequentially (unlike noise-fill/quantize, which use
every available core) — fine for one source, but if a real deployment
configures many tone sources, their costs land on the critical path and
add up linearly with no parallel speedup.

### Attacking the ~45%/~46% split

Re-reading the profile line by line (not just the two rollup percentages)
found three concrete costs, none requiring more threads (the "irreducible,
regardless of core count" lesson above is about a bulk copy's *thread
count*, not its *byte count*, which is a different lever):

- **`runtime.memmove` moved twice the bytes it needed to.** The whole
  pipeline stored samples as `complex128` (16 bytes) even though the wire
  format quantizes every component down to int8 in the end. Narrowed to
  `complex64` (8 bytes) — halves every byte moved by the noise-copy and by
  both quantize passes reading the same buffer straight afterward.
  Phase-sensitive computation (tone's NCO/delay-polynomial evaluation,
  noise's Box-Muller draw) still happens entirely in `float64` — only the
  final sample value narrows on store.
- **The quantization scale scan called `math.Sqrt` once per sample** just
  to find one per-channel max magnitude. Since sqrt is monotonic for
  non-negative inputs, tracking the max squared magnitude across the loop
  and taking one `math.Sqrt` at the end is exactly equivalent.
- **`math.Round` ran twice per sample** in the quantize/clamp path, paying
  for NaN/Inf/magnitude-≥2^52 handling that can never trigger on a sample
  already scaled into roughly [-127, 127]. Replaced with
  `v + math.Copysign(0.5, v)` then truncate-on-conversion to int8.

Confirmed real on target hardware (comparing CPU-seconds-per-wall-clock-
second between two profiling captures): `runtime.memmove` (noise-copy)
dropped ~47.7% (matching the theoretical complex64 halving almost
exactly), quantize+encode cumulative dropped ~30.4%, total average
concurrency dropped ~29.8%. `WriteBatch` (network syscalls alone) stayed
flat, confirming the reduction came from the encode side specifically, not
run-to-run variance.

**Still not enough to clear budget comfortably** at that point: occasional
drift of 1-10ms still occurred (down from "occasional drift under 100ms"
pre-fix — roughly a 10x reduction in overrun magnitude, but still present).

### Fixed vs. adaptive quantization scale

The next lever: replacing the adaptive per-heap scale (re-scanning every
heap's actual samples for their own max magnitude, every tick) with a
fixed scale computed once from the streamer's own configuration and reused
for every heap.

This changes only how already-generated noise gets digitized to fit int8
on the wire, not how it's generated (noise generation — independent
per-station/per-pol seeded Box-Muller draws, full float64 precision, no
delay-correction — is untouched). A complex sample's magnitude follows a
Rayleigh(std) distribution, whose tail is `P(magnitude > k·std) =
exp(-k²/2)`; a margin of 8 standard deviations was chosen so that even at
384 channels × 2 pols × ~10^8 ticks (a deliberately absurd
multi-year-continuous-scanning upper bound), the expected number of
samples that would ever exceed this bound across that lifetime is under
0.001. The tradeoff: the fixed scheme doesn't re-optimize per heap, so a
typical heap uses somewhat less of the full ±127 range than the adaptive
scheme's always-exactly-optimal fit — a pure-noise channel's
quantization-noise-to-signal-power ratio works out to roughly 1/3000
(fixed) vs. roughly 1/12700 (adaptive) — about 4x more quantization noise,
but both figures are negligible next to the noise's own power and
unrelated to anything a delay-tracking/correlation/beamforming test could
detect.

**Target-hardware result: worked exactly as designed at the function
level, but the aggregate effect was small and drift didn't meaningfully
improve.** The quantization-scale scan disappeared from the profile
entirely and the relevant function's cumulative cost dropped ~17.2%
between captures, but total average concurrency only dropped ~2.6% (far
short of the ~30% swing the complex64 change produced) — mostly because
the scale scan was already down to a minority cost before this fix, and
`runtime.memmove` (untouched this round) moved slightly the other way
between captures, almost certainly ordinary run-to-run variance on a
shared host. Largest drift measured 18ms — "similar performance" to
before per direct observation. The honest result of a fix that targeted an
already-minor cost.

### Pre-quantized noise tile bank

The next, bigger lever: noise-only channels (no tone source targets them)
now skip the complex64 dst-write/`HeapAccumulator` path entirely and get
complete, ready-to-send heaps built directly from a noise tile bank that's
pre-quantized to int8 once at construction, not adaptively
rescanned+rounded+clamped every tick. This is a real architectural change,
not a tuning knob.

A tone source's channel is fixed for the whole scan (it depends only on
the source's static configured frequency, never on per-tick delay), so
which channels need full-precision samples (to combine with tone before
quantizing) vs. which are noise-only is knowable once, at streamer
construction. The design added an optional, type-asserted "complex path
channel map" capability (so any streamer that doesn't implement it keeps
the original full-width behavior unchanged) and an optional "generate
quantized heaps directly" capability for everything else, bypassing
`HeapAccumulator`. `ChannelHeap` gained pre-quantized byte fields as an
alternative to complex sample fields per pol; the encoder branches on
which is set.

Verified: a dedicated test proves the pre-quantized bank produces bytes
that exactly equal the same fixed-scale quantization applied to the
original full-precision path's output, sample for sample — this doesn't
introduce a new precision tradeoff, only changes *when* the already-
decided fixed-scale quantization happens. A real end-to-end smoke test
(not just unit tests) confirmed the right channels carry tone vs.
noise-only content by decoding captured packets.

Dev-machine (Apple M5, directional only) benchmarks:
`BenchmarkProducerTick` at 384 channels: 212µs → 89.8µs (-57.7%); at 96
channels: 64.7µs → 21.3µs (-67.1%) (both already reflecting the earlier
complex64 halving; this is the additional drop from pre-quantizing).
Per-heap encode: adaptive 7865ns → fixed_scale 4506ns → quantized 2344ns.

**Target-hardware validation: `runtime.memmove` dropped sharply as
predicted** (down to 6.0% of total CPU in the next capture, from 33-37%
across the two prior captures). **But a new cost took its place**: the
plain byte-copy that replaced the old quantize/round/clamp work was itself
36.03% of all CPU — the #2 cost overall. Root cause: it ran independently
per pol, each call writing only 2 of 4 bytes per sample (a strided partial
write), meaning every 4-byte block in the shared payload buffer needed its
own read-for-ownership twice instead of once. Fixed with a combined pass
writing all 4 bytes of each sample together in one go — confirmed faster
via a same-machine before/after (-17% on the dev machine; the real-hardware
effect was expected to be larger, since the strided-vs-combined difference
is a cache/memory-bandwidth-pattern effect a warm-cache microbenchmark
understates relative to real, cold, high-throughput traffic). Proven
behavior-identical to the independent two-pass version via a dedicated
test.

A second, separate inefficiency found in the same capture: a test run
configured one tone source among 384 channels, meaning the full-precision
complex64 bank was built at the full 384-channel width (~3.2GB) even
though only that one channel's column was ever read. Fixed: the complex64
bank is now sized to just the tone-affected channel subset — for that
scenario, ~3.2GB down to ~8.4MB, dropping the Box-Muller work needed to
fill it by the same ~384x factor.

### Tick-clustering analysis: distinguishing warm-up from steady-state drift

Both fixes above were plausible contributors to "producer drifting,
especially at the start," but for different reasons — one a steady-state
per-tick cost (would show throughout the scan), the other a one-time
construction cost (wouldn't directly cause tick-to-tick drift, but its
page-fault/allocator settling could plausibly bleed into the first several
ticks).

Parsing every "falling behind pacing" log line's tick number into a
time-since-scan-start and bucketing into 5s windows (rather than just
eyeballing the log) found drift overwhelmingly clustered at the start:
2641 total events across a 90s scan, 76% of them in the first ~18s, with a
clean near-zero gap from 18-27s and only scattered events afterward.
Within that first-18s window, per-second event count wasn't monotonically
decaying either — a repeating sawtooth (bursts of drift, brief recovery,
another burst), settling out entirely by ~18-20s. This matched an
independently documented signature for this exact target machine (idles
at 1.5GHz, needs sustained load before `schedutil` ramps to boost clocks)
— not proof by itself (a GC/allocator-warmup explanation would also
plausibly show an early-and-settling pattern), but a strong prior toward
the CPU-frequency-scaling hypothesis, given the ~15-20s timescale match was
specific, not just "early vs. late."

Operationally more important than which hypothesis was right: real scans
don't run back-to-back — each station pod starts/stops per integration
test with multi-second gaps between successive scans, easily enough idle
time for `schedutil` to drop the CPU back down between scans. So whichever
mechanism this turned out to be, it was not a one-off benchmarking
artifact — it would recur on every single real scan.

### CPU governor test: `performance` mode + HPC BIOS profile

Directly tested the clock-ramp hypothesis by removing the variable it
depends on: CPU governor switched from `schedutil` to `performance` (fixed
max frequency, no ramp-up delay), plus a "HPC" BIOS power profile enabled
on the Dell R7525 test box.

**Result: drift events dropped from 2641 to 301 — an 8.8x reduction —
strongly confirming CPU clock ramp was a major real contributor.** But it
did not fully eliminate the pattern: 90% of the remaining events still
landed in the first 10s, tapering to a scattered handful through the rest
of a 78s window. Max single-event drift was 25ms here vs. 18ms in the
earlier capture — higher, not lower, on the single worst event, though
from 8.8x fewer samples (one data point, not yet repeatability-checked).

With the dominant confound (clock ramp) now controlled for, the
still-clustered-at-start residual became better evidence for a secondary
hypothesis (GC/allocator/page-fault settling right after scan start).

### GC ruled out

`GODEBUG=gctrace=1` (works on the compiled binary directly, no Go
toolchain needed on the target — set via `sudo env GODEBUG=gctrace=1
<binary> ...`, since plain `VAR=val sudo cmd` doesn't survive most
sudoers' `env_reset`) showed only 10 GC cycles across a 90s scan, every
one with a sub-2ms total stop-the-world pause, and after the first three
(heap ramping up to its steady-state size once) spaced evenly roughly
every 12s across the entire scan — not clustered early the way the drift
was. GC/allocator warm-up was eliminated as a hypothesis.

### Clock ramp also ruled out under `performance`/HPC

A per-core CPU frequency trace (0.2s resolution across the full 90s scan,
21,409 samples across 48 cores; a `/proc/cpuinfo` "cpu MHz" fallback was
needed since this EPYC's `amd_pstate` driver doesn't populate
`scaling_cur_freq` and `turbostat` wasn't installed) showed mean per-core
frequency essentially flat the entire scan, ~2860-2920MHz from t=0s
straight through t=90s, with no ramp-up shape at all — t=0's mean was if
anything slightly higher than several later buckets, the opposite of what
a clock-ramp hypothesis predicts. This ruled out CPU clock ramp too, under
`performance` governor + HPC. Both originally proposed hypotheses for the
residual first-5-10s drift clustering were now eliminated.

### Cold `sync.Pool`s at scan start: the actual remaining cause

With clock-ramp and GC both ruled out, the next candidate: the per-channel
buffer pools starting completely empty every time a scan begins — not
just on a fresh process, but in a real long-running device-server pod
too, since Go's runtime drops every `sync.Pool` entry on every GC cycle,
and real scans have multi-second-plus gaps between them (enough for at
least one GC cycle to land in the gap). Until enough buffers cycle through
release to refill the pools "for free," every allocation in that window
pays `make()`'s cost instead.

Fixed by warming both buffer pools with `numChannels*2` fresh buffers
before a scan's pacing loop can ever begin ticking (an oversized put on
the "wrong" pool is harmless — just a few extra entries that age out on
the next GC like anything else). This is a pure implementation-detail
change (same output, same wire content), so it was implemented directly
rather than proposed first, per this project's standing autonomy rule for
changes with no output impact.

**Target-hardware result: real, substantial improvement.** Total drift
events: 51, down from the 136-event baseline immediately prior (the full
chain across this investigation: 2641 → 301 → 231 → 136 → 51, roughly 52x
from where it started). Max single-event drift dropped to 7ms, the lowest
yet. More important than the count: the *shape* changed — every prior
capture had its very first drift event within the first handful of ticks;
this run's first drift event didn't happen until tick 1001, ~2.2s into the
scan, a genuinely different signature. The remaining events were also much
less front-loaded as a fraction (45% in 0-5s, vs. 72-91% in every earlier
capture).

Pool cold-start was a real, independent contributor alongside (not instead
of) clock ramp. Three real, independent causes were found and addressed in
total: clock ramp (via infra config), GC (ruled out entirely), and cold
pools (via this fix).

### Decision: pausing here

Confirmed, from the pacing loop itself rather than just from logs: the
scan loop's target wall-clock deadline for each tick is computed as the
scan's start time plus `tick * blockDuration` — an absolute per-tick
deadline, never derived from when the previous tick finished. A slow
tick's overrun is therefore isolated to that tick; the next tick's
deadline doesn't shift later to accommodate it, so as soon as one tick's
actual work drops back under budget, normal sleep-based pacing resumes
with no carried-over debt. This is also the structural reason drift showed
up as isolated bursts followed by clean gaps rather than a monotonically
growing overrun throughout this whole investigation — the one way this
guarantee would break is per-tick production becoming *systematically*
slower than budget for a sustained stretch, not just occasionally, which
nothing measured showed. `sim_time` is likewise computed from `obs_time +
tick*block_duration`, independent of wall-clock time — so a late tick's
content (heap timestamps, `heap_counter`, delay-poly evaluation) is still
exactly correct regardless of pacing drift; only when the heap goes out on
the wire is ever late, never what it says.

Given that guarantee plus the ~52x reduction in drift events achieved
across this investigation (2641 → 51 on the same 90s/384ch/one-tone test),
the decision was to stop chasing the remaining residual rather than pursue
a goroutine/OS-thread-spin-up hypothesis further. Revisit if a future
capture ever shows overrun growing tick-over-tick within a burst instead
of recovering, or if CBF's actual delay-tracking test tolerance turns out
to need tighter margins than this.

## Historical hardware notes

See "Historical hardware notes (legacy wideband path, superseded designs)"
under the Python benchmarking history above, and the EPYC/Mellanox
target-hardware descriptions throughout the Go pacing investigation above
— both are kept there rather than duplicated here since they're specific
to the investigations that used them.
