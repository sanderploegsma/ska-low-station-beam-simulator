# CBF Station-Beam Simulator

Software simulator generating SKA Low SPS station-beam data, replacing the
CNIC hardware/firmware tool currently used to test the CBF
correlator/beamformer's delay-tracking, correlation, and beamforming
logic. Independence from CBF matters here specifically: CNIC is built by
the same team that builds the CBF firmware being tested, which undermines
the value of the test. This simulator must generate "true" delay
independently of whatever CBF itself computes.

**Start with [`README.md`](README.md)** for the current architecture,
repository layout, and how to build/test/run both halves of this project
(a Python Tango device server under `tango/`, and a Go signal-generation
process at the repo root driven over gRPC). This file only covers what a
session needs on top of that: settled decisions not to relitigate, and
constraints that have already cost real debugging time once.

**For the reasoning behind any of the decisions below, past design
iterations that were tried and rejected, specific bugs that were found
and fixed, and benchmark/profiling investigations, see
[`docs/history.md`](docs/history.md).** That file is the archive; this
one is the summary. If something below seems surprising or arbitrary,
check there before changing it — it's very likely already been tried the
other way.

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
  architecturally required. **Open item**: the exact CSP LMC command for
  this is still not confirmed.
- **Deterministic, clock-independent `sim_time`.** Content generation
  never depends on any pod's own wall clock — only on `obs_time_ref`
  (given once, identically, to every station pod at scan start) plus a
  derived tick index / relative time. Every generation kernel (tone
  phase, noise sample index, delay polynomial evaluation) is a pure
  function of a small, `obs_time`-relative `t`, specifically so any pod
  can independently compute any tick without shared state, and so a slow
  tick never corrupts content — it only arrives late.
- **Each station is one Tango device server, one Kubernetes pod**,
  self-sufficient after receiving `obs_time` at scan start.
- **A Tango device server owns Tango; a separate Go process owns signal
  generation and SPEAD/UDP sending**, driven over gRPC
  (`api/simulator.proto`). The Go process has no Tango access of its own.
  Pulsar/pulsed-source generation is not currently implemented in either
  language (an earlier Python prototype was removed once tone/noise
  reached parity in Go — see `docs/history.md`).
- **TAI2000 is the SKA epoch** for `heap_counter`.

## Constraints that have already cost real debugging time — check `docs/history.md` before assuming otherwise

- **Every generation kernel takes a small-magnitude, `obs_time`-relative
  time, never raw epoch time.** Evaluating a delay polynomial or a tone's
  phase against raw Unix-epoch-scale `t` collapses float64 precision.
- **Noise must be seeded independently per (station, pol), never shared
  across stations or between V/H.** A shared seed makes every station
  emit byte-identical "noise" for a given tick, silently breaking any
  test that depends on receiver noise being uncorrelated across the
  array.
- **Noise never enters a delay pipeline.** Receiver noise originates
  locally per station, after any signal-path delay would apply —
  delay-correcting it is physically wrong, not just a style choice.
- **Every source (tone, and any future pulsar) requires its own real
  delay feed — there is no default/fallback delay.** A source with no
  real delay path would silently apply zero delay, producing content
  that's trivially "perfectly aligned" — exactly the kind of thing that
  could mask a real CBF delay-tracking bug, given this simulator's whole
  reason for existing.
- **spead2 cannot produce this ICD's heap format.** Its packet encoder
  unconditionally writes 4 reserved item pointers with no way to
  suppress any of them, but the ICD heap has only 6 items total. Both
  the Go (`internal/spead`) and Python (`tango/src/.../simulator.py`,
  historically) encoders are hand-rolled for exactly this reason — don't
  reach for spead2 or another general-purpose SPEAD library here.
- **The real per-channel sample rate is oversampled by 32/27** (1080ns
  per sample, not 1280ns) — this shrinks the real per-tick budget by
  ~15.6% versus what a critically-sampled assumption would give. See
  `README.md`'s "SPS-CBF ICD reference facts" for the current correct
  constants, and `docs/history.md` for how this (and two other
  channelization assumptions) were found wrong and corrected.
- **A bulk memory copy that's bandwidth-bound doesn't get faster with
  more threads/goroutines.** More than one investigation in this
  codebase's history mistook an allocation or copy bottleneck for a
  compute bottleneck and initially reached for more parallelism instead
  of eliminating the redundant copy — see `docs/history.md` for the
  specific cases (Python's per-tick noise buffer allocation; the Go
  port's `HeapAccumulator`/quantization pipeline).

## Quick reference

```
uv sync                 # Python deps (tango/ package, defined by root pyproject.toml)
uv run pytest           # Python tests
go build ./... && go vet ./... && go test ./...   # Go build/vet/tests
```

See `README.md`'s "Setup" section for proto regeneration (both
languages), running the simulator/`noise-stream` locally, and the
container image build.
