# Go station-beam simulator prototype

A Go port of this project's numeric core (tone + per-pol station noise)
and SPEAD-64-48 packetizer, plus a sample gRPC control-plane layer, to
evaluate replacing the Python simulator's signal-generation half with Go
while keeping (or later re-architecting) the Tango device server
separately. See the parent repo's `CLAUDE.md` for full background on the
simulator this ports from.

## Scope

**Ported**: tone synthesis, the per-pol noise tile bank, the hand-rolled
SPEAD-64-48 encoder, delay polynomial/`DelayFeed` handling, the heap
accumulator, and the real-time pacing scan loop (`ScanRunner`) — i.e.
everything needed for CNIC-replacement feature parity on tone/noise
sources.

**Deliberately NOT ported (yet)**: pulsar/pulsed sources. The Python
CLAUDE.md documents three design iterations, an external
(NANOGrav/PsrSigSim) physics cross-check, and a non-trivial one-time FFT
construction budget for that feature — a substantial follow-up, not
something to bolt on as an afterthought here. Tone/noise are what's
needed for CNIC feature parity first; pulsar support is a later
addition once this prototype's core numeric/architectural questions are
settled.

**Not addressed by this prototype**: the actual Tango-facing process
(device properties, `AttributeProxy` subscriptions to CBF's delay-poly
emulator) — see "Architecture" below for how that's expected to plug in
via gRPC, and `api/simulator.proto`'s doc comment for the split this
assumes.

## Why Go is a reasonable fit here (and where it isn't)

- **Per-tick kernels** (`synth.synthToneChannel`, the noise tile-bank
  fill/replay) map cleanly onto Go: no GIL to route around with numba's
  `prange`/fork-join model, `complex128`/`math/cmplx` are built in, and
  goroutines are a natural fit for the noise bank's independent-seed
  parallel fill (see `internal/synth/noise.go`).
- **What doesn't carry over directly**: `astropy`'s leap-second-aware
  TAI2000 conversion has no drop-in Go equivalent —
  `internal/common/tai2000.go` uses the same hardcoded-offset fallback
  the Python side already flags as unsafe for production (see that
  file's doc comment). A real deployment needs a proper leap-second
  source (or to share one with the Python side) before this matters.
- Pulsar's one-time dispersion-FFT construction step (not ported here)
  would need a Go FFT library (no direct `scipy.fft`-with-`workers=`
  equivalent) — flagged as a real open question for that follow-up work,
  not solved by this prototype.

## Architecture

```
Tango device server (Python, not in this prototype)
  - owns device properties, AttributeProxy subscriptions to CBF's
    delay-poly emulator
  - forwards delay polynomial updates via PushDelayUpdate
        |
        | gRPC (api/simulator.proto)
        v
This Go process (cmd/simulator)
  - StartScan/StopScan/GetStatus/PushDelayUpdate (internal/server)
  - numeric core: tone + noise tile bank (internal/synth)
  - SPEAD-64-48 encoding + UDP send (internal/spead)
  - real-time pacing loop (internal/common.ScanRunner)
        |
        | SPEAD/UDP
        v
      CBF
```

`internal/common` mirrors the Python `common.py`'s backend-agnostic
plumbing: `DelayPolynomial`/`DelayFeed`, `StationConfig`/`ChannelHeap`,
`HeapAccumulator`, and `ScanRunner` (driven through a small `Streamer`
interface, exactly like Python's `Streamer` `Protocol` — this package
never imports `internal/synth`, so it stays testable independent of the
generation strategy behind it).

Note one real difference from the Python original: Python's
`DelayFeed` documents that a plain reference swap is safe under the GIL
with no explicit lock. Go has no GIL, so `internal/common/delay.go` uses
`atomic.Pointer` for the polynomial itself and a mutex for the
"warned once" bookkeeping.

There are two entrypoints (`cmd/simulator`, the gRPC-served one above,
and `cmd/noise-stream`, a standalone noise (+ optional test-tone) CLI —
see "Building and running" below), sharing the plumbing that has
nothing gRPC-specific about it: `internal/common.HeapQueue` (a bounded,
non-blocking `HeapSender`), `internal/spead.BatchSendLoop` (drains a
queue into an `SpsPacketizer` via a pool of parallel sender goroutines —
see `spead.NewSenderPool` and the profiling history below for why this
is no longer the single-goroutine `SendLoop` it started as), and
`internal/netutil.InterfaceIPv4Addr` (resolves a named interface's
address for the `-spead-interface`/Multus flag both entrypoints
support). `internal/server` depends on all three; `cmd/noise-stream`
depends on `common`/`spead`/`netutil`/`synth` directly and never imports
`internal/server` or any gRPC package at all.

## Building and running

Requires Go 1.24+, and `protoc`/`protoc-gen-go`/`protoc-gen-go-grpc` only
if you need to regenerate `api/simulatorpb/` after editing
`api/simulator.proto`:

```
brew install protobuf
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

protoc --go_out=api/simulatorpb --go_opt=paths=source_relative \
       --go-grpc_out=api/simulatorpb --go-grpc_opt=paths=source_relative \
       --proto_path=api api/simulator.proto
```

Build, test, run:

```
go build ./...
go test ./...
go vet ./...

go run ./cmd/simulator -listen :50051 -station-id 1 -dest-ip 127.0.0.1 -dest-port 8000
```

`cmd/simulator`'s flags mirror `simulator.py`'s static device properties
(`station_id`/`substation_id`/`dest_ip`/`dest_port`); everything that
varies per scan (`subarray_id`, `beam_id`, tone sources, noise config) is
a `StartScan` gRPC request field instead, matching the Python
`StartScan` JSON argument's shape.

### `cmd/noise-stream`: a standalone noise (+ optional test-tone) CLI

For quickly exercising the numeric core + SPEAD packetizer end-to-end
(e.g. against a packet capture tool, or CBF's receive path) without
standing up the gRPC service or a Tango-facing counterpart process at
all:

```
go run ./cmd/noise-stream -dest-ip 127.0.0.1 -dest-port 8000 -scan-duration 10
```

No gRPC, no *real* delay sources — just the noise tile bank (and,
optionally, one tone source), driven by a `ScanRunner` exactly like a
real scan. `-station-id`/`-substation-id`/`-subarray-id`/`-beam-id`/
`-scan-id` all default to `1`, `-num-channels` defaults to `96`,
`-obs-time` defaults to `"now"` (or pass a fixed Unix epoch seconds
value for a reproducible run), `-scan-duration` defaults to `60`
(seconds). `-noise-std` defaults to `0.05` and `-noise-seed` defaults to
the same value as `-station-id` unless explicitly overridden — both
match `simulator.py`'s own `StartScan` noise default
(`NoiseConfig(std=0.05, seed=self.station_id)`). `-spead-interface`
works exactly as in `cmd/simulator`: pass a network interface name (e.g.
`net1` for a Multus-attached secondary NIC) to bind the outbound
SPEAD/UDP socket to that interface's IPv4 address instead of letting the
OS pick via its default route; leave unset to use the OS's default
selection. Stops on its own after `-scan-duration`, or immediately on
Ctrl-C/SIGTERM.

`-tone-freq-hz` (0 by default, meaning no tone) adds a single tone
source on top of the noise, at the given frequency in Hz, with
`-tone-amplitude` (default `1.0`). This exists **only** to measure
tone's computational cost on top of noise — see "Real-hardware
profiling" below for why that question came up. It's backed by a
STATIC delay feed (constructed once, updated once to a permanent
zero-delay polynomial, never touched again), not a real subscription —
every other source-config path in this codebase deliberately has no
default/fallback delay (see the Python CLAUDE.md's "Per-source delay"
section for why), and this flag doesn't change that principle for any
*real* scan. Don't use `-tone-freq-hz` to reason about delay-tracking
correctness, only about per-tick timing.

`-sender-goroutines` defaults to `0`, meaning "auto": it scales with
`-num-channels` via `spead.DefaultNumSendersForChannels` (16 sender
goroutines at the full 384-channel band, proportionally fewer for a
narrower configuration — see "Real-hardware profiling" below for the
real-hardware measurement this is based on). Pass an explicit value to
override.

## Container image

`Dockerfile` builds `cmd/simulator` into a minimal, non-root image
(`golang:1.25-alpine` build stage, `gcr.io/distroless/static-debian12:nonroot`
runtime — no shell, no package manager, matching this binary's actual
needs: a gRPC listen socket and a UDP socket to CBF). Build context is
this directory, not the parent repo:

```
docker build -t station-beam-simulator-go go-simulator
docker run --rm -p 50051:50051 station-beam-simulator-go \
    -listen :50051 -station-id 1 -dest-ip <cbf-host> -dest-port 8000
```

`.github/workflows/go-simulator-docker.yml` (repo root — GitHub only
looks for workflows there) runs `go build`/`go vet`/`go test -race` on
every push/PR touching this directory, then builds and publishes a
multi-arch (`linux/amd64`,`linux/arm64`) image to
`ghcr.io/<owner>/<repo>/go-simulator` on pushes to `main` (tag `latest`)
and on `go-simulator-v*` tags (semver tag) — pull requests build the
image (to catch a broken Dockerfile early) but never push it.

## Testing notes

`go test ./...` covers:

- **`internal/synth`**: tone lands in the expected channel with an exact
  (not first-order-approximated) delay-as-phase shift; the noise tile
  bank is deterministic given the same seed, statistically matches its
  configured std, and is independent across different (station, pol)
  seeds (guards against the Python codebase's bug #1: reusing one noise
  source for both pols); noise is never delay-corrected, matching the
  physical requirement that receiver noise originates after any
  signal-path delay.
- **`internal/spead`**: encoded heaps decode back to exactly the ICD's 6
  items (byte-level, independent of the encoder's own logic) with the
  correct bit-packed `channel_info`/`antenna_info`; a heap_counter
  overflow (the historical bug where the wrong formula inflated it by
  `HeapLen`=2048x) is rejected, not silently truncated.
- **`internal/common`**: `DelayFeed`'s zero-delay-until-first-update and
  keep-applying-stale-polynomial behaviours; `HeapAccumulator` framing
  and timestamping; `ScanRunner`'s pacing loop against a fake streamer.
- **`internal/server`**: the gRPC surface end-to-end over an in-memory
  `bufconn` listener (`StartScan`/`StopScan`/`GetStatus`/
  `PushDelayUpdate`, including the validation error codes); `Start()`
  actually binding to a real (loopback) interface, and failing for an
  unknown one.
- **`internal/netutil`**: resolving a real (loopback) interface's IPv4
  address, and erroring for an unknown interface name — portable across
  Linux/macOS (looked up by interface flag, not a hardcoded name like
  `"lo"`/`"lo0"`).

Not covered: a live end-to-end SPEAD capture decoded by an external
tool. Actual throughput/timing benchmarks against `common.BlockDurationS`
on target server hardware — the one gap this section used to flag as
entirely open — now has a substantial history; see "Real-hardware
profiling & pacing investigation" below.

## Real-hardware profiling & pacing investigation

This section is a running log of real-hardware findings for future
sessions to pick up from — matching the style (and, for the noise tile
bank/allocation-overhead lessons, the actual root causes) already
established in the parent Python simulator's own `CLAUDE.md`. Keep
adding to it rather than replacing it; a wrong turn that was tried and
measured is as valuable a record as a fix that worked, per that file's
own "measure, don't assume" precedent.

**Hardware**: an SR-IOV VF on a Mellanox ConnectX-6 (100G NIC), MTU
9000, on a real (non-laptop) target-class Linux box — a 2-socket AMD
EPYC per the parent `CLAUDE.md`'s own target-server description.
`noise-stream -cpuprofile <file>` plus `go tool pprof -top`/`-list`/
`-peek <file>` is the workflow that found every real cost below; a dev
laptop (this session used a 10-core Apple M5) is useful for correctness
and directional checks but is NOT representative of the target
machine's core count or memory-bandwidth profile — every number in this
section is from the real hardware unless explicitly marked otherwise.

### Timeline

1. **Outbound send-queue saturation.** Initial testing found the send
   queue filling up and dropping packets, with receive-side throughput
   around 15MiB/s despite the 100G link. Root cause: one UDP socket
   sending one heap at a time. Fixed with batched sends
   (`golang.org/x/net/ipv4.PacketConn.WriteBatch`, which uses
   `sendmmsg(2)` on Linux — confirmed directly from `x/net`'s source,
   not assumed) and N parallel sender sockets (`spead.SenderPool`, each
   its own source port — spreads outbound traffic across the NIC's/
   receiver's RSS flow hash instead of pinning everything to one queue),
   plus `SO_SNDBUF` sizing (`-udp-send-buffer-bytes`, default 8MiB vs.
   the OS's often-~208KB default). This resolved the send-side problem
   completely and was never revisited.

2. **Producer falling behind pacing.** With sending fixed, generation
   itself couldn't keep up — worse at 384 channels than 96, drift
   climbing without bound (e.g. 113s at tick 8499 in one early run). Six
   real fixes were needed, each found by profiling the ACTUAL bottleneck
   rather than assuming one, in this order:

   a. **`HeapAccumulator` allocation/access-pattern bug**: a transpose
      loop making 768 small allocations/tick at 384 channels, exceeding
      the ENTIRE per-tick budget on its own. Fixed with a layout change
      (see below) plus reusing a flat buffer instead of allocating per
      element.
   b. **Row-major → channel-major generation layout**: `Streamer.
      GenerateNextTick`'s output changed from sample-major to
      channel-major (one channel's samples contiguous), eliminating a
      per-tick transpose in `HeapAccumulator` entirely instead of paying
      for one every tick.
   c. **SPEAD encoding allocation elimination**: `spead.BatchSendLoop`
      now reuses a per-goroutine pool of pre-allocated wire-size buffers
      (`EncodeChannelHeapInto` writes in place) instead of allocating a
      fresh buffer per heap — safe because a UDP send copies the buffer
      into the kernel synchronously before returning.
   d. **Producer parallelization**: profiling found `ScanRunner`'s
      single producer goroutine (generation + accumulation) running at
      ~99% duty cycle on ONE core for the whole scan, unlike sending
      (already parallelized). Both `DirectSynthesisStreamer.
      GenerateNextTick`'s noise fill and `HeapAccumulator.Add`/
      `PopReadyHeaps` were split across goroutines by channel range
      (`common.forEachChannelRange`), with a `GOMAXPROCS`-based default
      worker count.
   e. **Negative result — raising the worker-count cap did NOT help**:
      `defaultParallelism` originally capped workers at a flat 16,
      copied from `fillNoiseBank`'s cap for its one-time construction
      cost. A real profile's average concurrency landed at ~15.18 —
      suspiciously exactly that cap — so it was removed. Re-profiling
      showed average concurrency rise to ~19.79 (+30%), total `memmove`
      CPU-seconds rise proportionally, and wall-clock drift stay
      completely unchanged (12.016s → 12.3s at the same tick). More
      threads were just doing more of the same redundant work in
      parallel — a bulk memory copy is bandwidth-bound, not
      thread-starved, exactly matching the Python `CLAUDE.md`'s own
      noise-tile-bank finding ("regardless of core count, even 1 core is
      enough"). Worth remembering before reaching for "add more
      goroutines" as a fix for a copy-dominated hot path again.
   f. **The real fix: eliminating redundant copies, not redistributing
      them.** Three layers, each found by re-profiling after the
      previous fix and confirming what actually changed:
      - `HeapAccumulator.PopReadyHeaps` was copying every channel's
        buffer into a second, freshly-allocated flat buffer before
        handing it to the sender. Removed via a zero-copy reslice/
        handoff of the buffer `Add` already built — safe because
        `TickNSamples() == HeapLen` always (by construction), so the
        "leftover tail" this handoff has to special-case is provably
        empty in production.
      - Generation still wrote into its own scratch buffer
        (`DirectSynthesisStreamer`'s old `outBufs`), which
        `HeapAccumulator.Add`'s `append` then copied AGAIN into its own
        per-channel storage. Removed by changing `common.Streamer`'s
        interface so `GenerateNextTick` writes directly into
        `HeapAccumulator`-owned buffers (`HeapAccumulator.
        PrepareWrite`), cutting the remaining copy in half again
        (confirmed: `runtime.memmove`'s share of total CPU dropped from
        ~57% to ~47%, then ~25% after the next fix below).
      - Once that copy was gone, `runtime.memclrNoHeapPointers` —
        `make()`'s mandatory zero-fill on every freshly-grown
        per-channel buffer, every tick — became the next-largest cost
        (~21% of ALL CPU time), even though that memory was about to be
        fully overwritten by the noise fill a moment later. Fixed with a
        `sync.Pool` of reusable per-channel buffers
        (`common.ReleaseSampleBuffers`, called from `spead.
        encodeHeapInto` once a heap's samples are read for the last
        time) — the same reuse pattern this codebase already used for
        SPEAD encode buffers, just applied one layer further upstream.
   g. **Result**: 384 channels went from drift climbing without bound
      (10+ seconds over a ~15s window) to only occasional, self-recovering
      drift under 100ms.

3. **`-sender-goroutines`' flat default (4) was outgrown twice** during
   this investigation (4 → 8 → 16) as the producer stopped being the
   bottleneck and the send side had to absorb much higher sustained
   throughput. Replaced with `spead.DefaultNumSendersForChannels`,
   scaling linearly from the confirmed real-hardware baseline (16 at the
   full 384-channel band) down to fewer senders for a narrower
   configuration — 96 channels lands on exactly 4, so existing
   narrow-band usage is unaffected. Not wired into `cmd/simulator`'s
   gRPC server: that `SenderPool` is created once at process `Start()`,
   before any scan's `num_channels` is known, so it still needs an
   explicit `-sender-goroutines` if a deployment needs something other
   than the flat default there.

### Current status and open items

**Margin is thin, not comfortable, even after all of the above.** A
rough estimate from the final 384-channel profile (total CPU-seconds ÷
average concurrency) puts the AVERAGE per-tick cost at roughly
100-101% of the 2.21184ms budget (`common.BlockDurationS`) — consistent
with the observed "occasional, self-recovering drift under 100ms": that
pattern is what running right at the edge looks like, not what a
comfortable cushion looks like. (Caveat: this estimate assumes perfectly
parallel work; the real pipeline overlaps generation and encoding across
ticks, so the true margin could be somewhat better or worse — trust a
direct measurement over this arithmetic if one becomes available.)

The two dominant remaining costs are roughly evenly split, and BOTH
scale with `numChannels × HeapLen`, not with source count:
- **~45%**: the noise-tile-bank copy itself (`fillNoiseRange`) — an
  inherent, now-irreducible per-tick memcpy, the same category of cost
  the Python side already documented as "regardless of core count, even
  1 core is enough."
- **~46%**: SPEAD quantize + encode + send (`spead.BatchSendLoop`),
  dominated by scalar `math.Round`/clamp work (the int8 quantization
  path) plus network syscalls, not allocation or copying anymore.

**Resolved: tone's real cost, confirmed negligible for a single source.**
Given how thin the margin above is, the natural next question was
whether adding tone sources would push 384 channels over budget.
Profiled on the real hardware with `-tone-freq-hz 250000000` (384
channels): `synthToneChannel` doesn't even appear in the profile's
top ~95% of CPU time — its total contribution across a 16.5s run fell
below the 2.09-CPU-second display cutoff, out of 418.18s total (under
~0.5%). The margin estimate came out at ~100.0%, statistically
indistinguishable from the noise-only run's ~101% — run-to-run
measurement noise on shared hardware is larger than tone's actual
contribution. This matches the analytical prediction: tone injection is
O(`nSamples`) per source (`synthToneChannel` computes one channel's
contribution in closed form, via the same NCO phase-accumulator trick
the Python pulsar work established, then adds `nSamples`=2048 values
into ONE channel) — negligible next to the ~45%/~46% costs above, which
scale with `numChannels`, not source count.

**Caveat, not yet tested: this was ONE tone source.** Tone injection
still runs SEQUENTIALLY (unlike noise-fill/quantize, which already use
every available core) — fine when one source's cost is unmeasurably
small, but if a real deployment configures many tone sources (tens, not
one or two), their costs land on the critical path and add up linearly
with no parallel speedup, unlike the rest of the pipeline. Not
parallelized because it hasn't needed to be yet; revisit (mirroring the
noise-fill/`HeapAccumulator` channel-range-split pattern) if a future
profile with many tone sources configured shows otherwise.

**Not yet tried**: whether `n_tiles`/`tile_n_samples` (the noise
tile-bank's own size/fidelity knobs, unchanged from their Go-port
defaults this whole session) trade meaningfully against the ~45% noise-
copy cost above; real multi-pod co-scheduling on one physical node
(every measurement above was one `noise-stream` process alone on the
target box).

### Follow-up: attacking the ~45%/~46% split directly (implemented, NOT yet target-hardware-validated)

Re-reading the `cpu-384ch.pprof` capture behind the ~45%/~46% split
above line by line (not just the two rollup percentages) found three
concrete, mechanically-verifiable costs inside them that hadn't been
addressed yet -- none of them require more threads (the "irreducible,
regardless of core count" lesson from item 2e above still holds for a
bulk copy's THREAD COUNT; it says nothing about the copy's total BYTE
COUNT, which is a different lever):

- **`runtime.memmove` (the noise-copy ~45%) moves twice the bytes it
  needs to.** The whole pipeline -- noise tile bank, `HeapAccumulator`
  storage, the SPEAD quantize passes that read it back out -- stored
  samples as `complex128` (16 bytes), even though the wire format
  quantizes every component down to int8 in the end (~2 decimal digits
  of real resolution). `complex64` (8 bytes) leaves several orders of
  magnitude more precision than that final step needs, while halving
  every byte moved by the noise-copy AND by both quantize passes reading
  the same buffer straight afterward. Phase-sensitive computation
  (tone's NCO/delay-polynomial evaluation, noise's Box-Muller draw)
  still happens entirely in `float64` -- only the FINAL sample value
  narrows on store, the same precision-vs-storage split this codebase
  already applies elsewhere. Changed throughout: `ChannelHeap`,
  `HeapAccumulator`'s buffers/pool, `Streamer.GenerateNextTick`'s `dst`
  type, the noise tile bank, `synthToneChannel`'s output, and the
  `spead.quantize*` functions' input type. `go test -race ./...` clean
  after.
- **`quantize8bitScale` (~10% of total CPU) called `math.Sqrt` once per
  SAMPLE** (e.g. ~1.57M calls/tick at 384 channels) just to find one
  per-channel max magnitude. Since sqrt is monotonic for non-negative
  inputs, tracking the max SQUARED magnitude across the loop and taking
  ONE `math.Sqrt` at the end is exactly equivalent -- cuts 2047 of every
  2048 sqrt calls at `HeapLen`=2048.
- **`math.Round` (~8% of total CPU) ran twice per sample** in
  `quantize8bitIntoPayload`, paying for NaN/Inf/magnitude-≥2^52 handling
  that can never trigger on a sample already scaled into roughly
  [-127, 127]. Replaced (`quantizeComponent`) with
  `v + math.Copysign(0.5, v)` then truncate-on-conversion to int8 --
  identical round-half-away-from-zero result for every value in that
  range, fused with the clamp in one pass instead of two separate calls.

**Not done, and deliberately left for a decision AFTER a real profile,
not before**: `defaultParallelism`'s scheduler-overhead question raised
while investigating this (a `findRunnable`/`schedule`/`stealWork`
cluster at roughly 13% of one capture) wasn't touched -- item 2e above
already found "throw more goroutines at a bandwidth-bound copy" to be a
dead end once, and the fixes above change how much data that copy
actually moves; re-chunking goroutine counts against a workload that's
about to look different would be reasoning ahead of measurement. Get a
fresh target-hardware profile with the changes above first.

**Existing correctness tests all pass unchanged** (same rounding/clamp
behavior verified bit-for-bit by `TestEncodeChannelHeap_
PayloadInterleavesVHRealImag`'s exact expected byte values, which
weren't touched), plus a new `BenchmarkEncodeChannelHeapInto` isolating
just the per-heap encode cost. Dev-machine-only numbers (Apple M5, NOT
the target box -- see this section's own opening caveat for why that
matters): `BenchmarkProducerTick/channels=384` (noise-copy path) at
212µs/tick, `BenchmarkEncodeChannelHeapInto` (quantize+encode path) at
8.0µs/heap, 0 allocs. Useful as a "did this obviously break something or
get slower" sanity check, NOT as a replacement for a real target-hardware
profile -- next step is exactly that: re-profile `noise-stream` on the
EPYC box the way the timeline above did, and confirm the ~45%/~46% split
actually shrunk, not just that it should have.

### Target-hardware validation of the above (`cpu-384ch.pprof`, 2026-09-08 13:25 CEST)

Confirmed real, not just theoretical. New profile, same 384-channel
noise-only workload, same box: 16.20s capture, 247.32s total samples
(1526.71% -- avg. concurrency 15.27 cores), vs. the profile the fixes
above were based on (60.59s, 1318.46s total, avg. concurrency 21.76
cores). Comparing CPU-seconds-per-wall-clock-second (the fair way to
compare two captures of different lengths):

| cost | before (core-rate) | after (core-rate) | change |
|---|---|---|---|
| `runtime.memmove` (noise-copy) | 9.73 | 5.09 | **-47.7%** -- matches the theoretical complex64 halving almost exactly |
| quantize+encode (`quantize8bitIntoPayload` cum) | 7.57 | 5.27 | **-30.4%** |
| `spead.BatchSendLoop` cum (encode+send together) | 10.00 | 7.73 | -22.6% |
| `WriteBatch` cum (network syscalls ALONE) | 2.19 | 2.24 | +2.3% (flat, as expected -- untouched code) |
| **total avg. concurrency** | **21.76** | **15.27** | **-29.8%** |

`WriteBatch` sitting flat while `BatchSendLoop`'s total dropped
confirms the reduction is coming from the encode side, exactly where
the fixes targeted it, not from some unrelated variance between runs.
Total CPU-rate needed to sustain the same 384-channel workload dropped
~30% -- a real, substantial reduction, not noise.

**Still not enough to clear budget comfortably.** Per the logs from this
same session: occasional drift of 1-10ms still occurs (down sharply from
the pre-fix "occasional, self-recovering drift under 100ms" in the
timeline above -- roughly a 10x reduction in overrun MAGNITUDE -- but
still present). That's the correct signal to trust here, not profile
arithmetic (see this section's own repeated "trust a direct measurement"
caveat) -- occasional drift, even small, means the pipeline is still
running close enough to the edge that it occasionally loses the race,
not comfortably underneath it.

**What's now dominant, ranked, from the fresh profile**:
1. `runtime.memmove` -- still #1 even after halving (33.32% flat, was
   44.70%). The noise-tile-bank copy remains the single largest line
   item; it's now byte-width-minimal (complex64) short of not storing
   full-precision samples for noise at all (see below).
2. `quantizeComponent` (round+clamp) itself -- 15.90% flat/18.87% cum.
   This is real per-sample work (two calls/sample, ~1.57M samples/tick
   at 384ch) that the sqrt/round fixes made cheaper per-call but didn't
   eliminate -- there's no further "wrong algorithm" fix left here, only
   "do it to fewer samples" (see below) or SIMD, which Go's stdlib
   doesn't expose.
3. `quantize8bitScale`'s own loop -- dropped from 9.66% to 6.38%, a
   real but smaller-than-hoped win: eliminating per-sample `math.Sqrt`
   only removed part of this function's cost -- the remaining per-sample
   multiply-add-compare loop (now also doing a `complex64`-to-`float64`
   promotion per component) is still real work.
4. Network send (`WriteBatch`/`sendmmsg`) -- ~14.2% cum, unchanged in
   absolute rate, now proportionally more visible simply because
   everything else got cheaper (Amdahl's law, not a regression).

**Caveat on this specific capture**: 16.20s is short enough that a
one-time cost (the noise tile bank's own construction --
`fillNoiseBank.func1`/`NormFloat64`/`PCG.Uint64` show up at a combined
~5.5% cum, which should be near-zero in true steady state) may be
inflating the numbers above somewhat. Doesn't change the ranking or the
core conclusion, but a longer capture (or one that starts profiling only
after the bank fill completes) would give a cleaner steady-state-only
read next time.

**Next lever, implemented after review**: `memmove` staying #1 even at
half the byte width pointed at the copy-then-separately-quantize round
trip as the next target. The FULL version of that idea -- pre-quantizing
the noise tile bank itself into int8 and skipping the complex64
intermediate entirely for noise-only channels -- would need a bigger
pipeline change (Streamer/HeapAccumulator/ChannelHeap all currently
promise complex64 samples, not pre-quantized bytes) and was NOT done
here; left as a still-open, larger follow-up if `memmove` remains the
top cost after everything else below.

What WAS implemented is the piece of that idea that doesn't require a
pipeline change: replacing the ADAPTIVE per-heap scale
(`quantize8bitScale` re-scanning every heap's actual samples for their
own max magnitude, every tick -- itself still 6.38% of total CPU per the
fresh profile above) with a FIXED scale computed ONCE from the streamer's
own configuration and reused for every heap, removing that scan from the
per-tick hot path entirely.

**The science, for anyone revisiting this**: this changes ONLY how the
already-generated noise gets digitized to fit int8 on the wire -- NOT
how it's generated. Noise generation (independent per-station/per-pol
seeded Box-Muller draws, full float64 precision, no delay-correction --
see the Noise section of the parent Python CLAUDE.md, which this port
follows) is completely untouched. The adaptive scheme finds each heap's
own actual peak sample and scales so that peak exactly fills ±127; the
fixed scheme instead reserves headroom based on the KNOWN statistics of
Gaussian noise plus the largest configured tone amplitude, once, instead
of re-measuring every heap. A complex sample's magnitude follows a
Rayleigh(std) distribution, whose tail is `P(magnitude > k·std) =
exp(-k²/2)` -- `synth.quantizeSigmaMargin = 8.0` was chosen so that even
at 384 channels × 2 pols × ~10^8 ticks (a deliberately absurd
multi-year-continuous-scanning upper bound, nowhere near real usage),
the expected number of samples that would EVER exceed this bound across
that whole lifetime is under 0.001 -- i.e. clipping risk is negligible,
not just "low." The tradeoff that DOES exist: the fixed scheme doesn't
re-optimize per heap, so a typical heap uses somewhat less of the full
±127 range than the adaptive scheme's always-exactly-optimal fit --
concretely, a pure-noise channel's quantization-noise-to-signal-power
ratio works out to roughly 1/3000 (fixed, `quantizeSigmaMargin=8`) vs.
roughly 1/12700 (adaptive, using a typical heap's actual ~3.9σ observed
max over 2048 samples) -- about 4x more quantization noise with the
fixed scheme, but both figures are utterly negligible next to the
noise's own power (σ²) and unrelated to anything a delay-tracking/
correlation/beamforming test could detect.
`DirectSynthesisStreamer.QuantizeScale()` computes this bound as the SUM
of every configured tone source's amplitude (worst case: all of them
land in the same channel and add exactly in phase) plus
`quantizeSigmaMargin` standard deviations of noise; 0 (meaning "use the
original adaptive scale") if neither noise nor tone is configured.

**Plumbing**: `spead.SpsPacketizer` gained `SetQuantizeScale`/an internal
atomic scale field (0 = adaptive, the untouched default -- every existing
test's exact expected byte values still pass unchanged with no call to
`SetQuantizeScale` at all). `spead.SenderPool.SetQuantizeScale` forwards
to its one shared packetizer -- needed because the gRPC-served path's
`SenderPool` is created once at process `Start()`, before any scan's
noise/tone config is known, and can outlive many scans with DIFFERENT
configs (see `server.Server.Start`'s doc comment), so this has to be a
live, thread-safe update (`StartScan` calls it), not a construction-time
value. `cmd/noise-stream` calls it once right after constructing its
streamer, since its config is known upfront from CLI flags.

**Measured (dev machine, Apple M5, NOT the target box)**:
`BenchmarkEncodeChannelHeapInto/adaptive` at 8058 ns/op vs.
`/fixed_scale` at 4951 ns/op -- a 38.6% reduction in per-heap encode
cost, isolating exactly the scan this removes. Same caveat as every
other dev-machine number in this file: directional only, next real
confirmation is another target-hardware profile.

### Target-hardware validation of the fixed-scale fix (`cpu-384ch.pprof`, 2026-09-08 13:44 CEST)

**Worked exactly as designed at the function level, but the aggregate
effect was small and the observable outcome (drift) didn't meaningfully
improve.** Both results matter and are recorded here, not just the
first one.

`quantize8bitScale` is now COMPLETELY ABSENT from the profile's top
nodes -- direct confirmation the scan is gone. `EncodeChannelHeapInto`'s
own cumulative cost dropped 17.2% (core-rate 5.31->4.40) between the two
target-hardware captures (16.20s/247.32 samples -> 15.81s/235.04
samples). Exactly what the fixed-scale change was supposed to do, and it
did it.

**But total avg. concurrency only dropped 15.27->14.87 cores (-2.6%)**,
far short of the ~30% swing the complex64 change produced. Why: (a)
`quantize8bitScale` was already down to a minority cost (6.38%, ~0.97
core-rate) before this fix -- there was only ever a small amount left to
remove here, unlike memmove; (b) `runtime.memmove` (untouched this
round) moved core-rate 5.09->5.44 (+6.9%) between the two captures --
almost certainly ordinary run-to-run variance on a shared host, not a
regression (nothing in this change touches noise generation/copying),
but it happened to offset a real chunk of the quantize win in this
specific pair of captures.

**Per the logs from this same test: largest drift 0.018s (18ms) --
"similar performance" to before, per direct user observation.** Not
better in any way that shows up in the metric that actually matters. This
is the honest, expected result of a fix that targeted an already-minor
cost: it was real, it's confirmed in the profile, and it still wasn't
enough to move the needle on pacing. `memmove` remains the dominant,
still-unaddressed cost (36.59% in this capture, arguably now MORE
clearly the top target since the smaller win around it has been
captured) -- the earlier-flagged "pre-quantize the tile bank into int8,
skip the complex64 round-trip for noise-only channels" pipeline change
remains the next real lever, and after this result it looks like the
one actually worth the bigger engineering investment, not an optional
nice-to-have alongside smaller wins.

### Pre-quantized noise tile bank -- implemented (2026-09-08 session)

The lever flagged above IS now implemented: noise-only channels (no
tone source targets them) skip the complex64 dst-write/HeapAccumulator
path entirely and get complete, ready-to-send heaps built DIRECTLY from
a noise tile bank that's pre-quantized to int8 ONCE at construction,
not adaptively rescanned+rounded+clamped every tick. This is a real
architectural change, not a tuning knob -- approved explicitly on the
basis that "different implementation, same-enough output" doesn't need
a stop-and-ask, only "would this change what comes out" does (see this
session's own discussion of the fixed-scale tradeoff's science for the
precedent).

**Design**: a tone source's channel is FIXED for the whole scan (it
depends only on the source's static configured FreqHz, never on
per-tick delay) -- so which channels need full-precision samples
(to combine with tone before quantizing) vs. which are noise-only is
knowable ONCE, at `NewDirectSynthesisStreamer` construction, not
per-tick:
- `DirectSynthesisStreamer.ComplexPathChannelIDMap()` (new,
  `common.ComplexPathChannelIDMapper`): the small, tone-affected
  channel subset. `common.ScanRunner` sizes its `HeapAccumulator` (and
  therefore `GenerateNextTick`'s `dst`) to just this subset now,
  instead of every channel -- a genuinely OPTIONAL Streamer capability
  (type-asserted, not a new required interface method), so any
  Streamer that doesn't implement it keeps the original full-width
  behavior unchanged.
- `DirectSynthesisStreamer.GenerateQuantizedHeaps(t)` (new,
  `common.QuantizedHeapProducer`, also optional/type-asserted): builds
  complete heaps for every OTHER (noise-only) channel directly,
  bypassing `HeapAccumulator`. `common.ScanRunner.run` calls this
  alongside (not instead of) the original `PrepareWrite`/
  `GenerateNextTick`/`PopReadyHeaps` sequence each tick.
- `common.ChannelHeap` gained `VQuantized`/`HQuantized []byte` fields --
  an ALTERNATIVE to `VSamples`/`HSamples`, not an addition, per pol.
  `spead.EncodeChannelHeapInto` branches on which is set: pre-quantized
  bytes get a straight, strided byte copy into the wire payload (no
  scale, no rounding, no clamping -- `copyQuantizedIntoPayload`); the
  original complex path is completely unchanged for tone-affected
  channels.
- `synth.fillQuantizedNoiseBank` (new) mirrors `fillNoiseBank`'s exact
  tile layout and per-worker seeding, but quantizes each draw (via the
  now-exported `spead.QuantizeComponent`) to an int8 pair on the way
  into the bank -- 2 bytes/sample instead of complex64's 8 (a further
  4x reduction on top of the earlier complex128->complex64 halving,
  8x total from where this session started). The full-precision
  complex64 bank is now skipped ENTIRELY when no tone is configured at
  all (the common case for `cmd/noise-stream`'s defaults) -- nothing
  would ever read it.

**The science, restated for this specific change**: this does NOT
introduce any NEW precision tradeoff beyond the fixed-scale one already
approved and explained earlier in this file -- pre-quantizing at
construction time instead of adaptively per heap only changes WHEN the
already-decided fixed-scale quantization happens, never WHAT it
computes.
`TestGenerateQuantizedHeaps_MatchesComplexPathQuantizedWithSameFixedScale`
proves this directly: `fillQuantizedNoiseBank` given the same seed as
`fillNoiseBank` produces bytes that exactly equal
`spead.QuantizeComponent` applied to `fillNoiseBank`'s own output at the
same fixed scale, sample for sample.

**Verified**: `go build`/`go vet`/`go test -race ./...` all clean.
Existing tests that called `GenerateNextTick` directly against a
noise-only (no-tone) streamer were rewritten to call
`GenerateQuantizedHeaps` instead (that's the whole point -- noise-only
channels no longer reach `GenerateNextTick` at all), same assertions
(determinism, cross-station independence, V/H independence, delay-
independence) preserved. `BenchmarkProducerTick` (mirrors
`ScanRunner.run`'s real per-tick sequence) rewritten to size
`HeapAccumulator` via `ComplexPathChannelIDMap()` and call
`GenerateQuantizedHeaps` too, matching what production actually runs
now (previously it would have silently emitted STALE/garbage heaps for
noise-only channels once the sizing changed, without this fix).

Also did a REAL end-to-end smoke test on this dev machine (not just unit
tests): `noise-stream` against a local UDP listener, both with a tone
configured and without. Decoding real captured packets confirmed: all
96 configured channel_ids present (64..159, matching `ChannelStart`),
no duplicates, the tone's channel (computed independently: `round((55MHz
-50MHz)/781.25kHz)` internal index 6 -> external channel_id 70) showed
a distinctly higher magnitude (~106) than its noise-only neighbors
(~14-16) -- confirming the complex path and the new quantized path are
both correctly wired to the right channels, not just "doesn't crash."
Without any tone configured, noise-only magnitudes were correspondingly
LARGER (~50-58, since `QuantizeScale()` no longer reserves headroom for
a tone that isn't there) -- confirms `QuantizeScale()`'s per-scan
(not per-channel) computation is doing the right thing.

**Dev-machine (Apple M5, NOT the target box) benchmarks, directional
only**:
- `BenchmarkProducerTick` (the real per-tick sequence): 384 channels
  212µs -> **89.8µs (-57.7%)**, 96 channels 64.7µs -> **21.3µs (-67.1%)**
  -- both ALREADY reflected the complex64 halving from earlier this
  session; this is the ADDITIONAL drop from pre-quantizing.
- `BenchmarkEncodeChannelHeapInto`: adaptive 7865ns -> fixed_scale
  4506ns -> **quantized 2344ns (-70.2% vs. adaptive, -48.0% vs.
  fixed_scale)**.

Not yet confirmed on target hardware -- next profile should show
`runtime.memmove` (still #1 at 36.6% in the last real capture) drop
sharply, since noise-only channels (the vast majority in any config
without many tone sources) no longer touch the complex64 bank/dst path
that memmove was measuring at all.

### Target-hardware validation: memmove fixed, but a new cost took its place (`cpu-384ch.pprof`, 2026-09-08 14:13 CEST, 90s scan, 384ch, one tone)

`runtime.memmove` DID drop sharply, exactly as predicted: 6.0% of total
CPU in this capture (core-rate 0.41), down from 33-37% across the two
prior captures. The pre-quantized-bank change is doing exactly what it
was built to do.

**But `spead.copyQuantizedIntoPayload` -- the plain byte-copy that
replaced the old quantize/round/clamp work -- was itself 36.03% of ALL
CPU (core-rate 2.48), the #2 cost overall.** Not the near-free operation
it should have been. Root cause, found by reading the function rather
than assuming a byte copy is always cheap: it ran independently per pol,
each call writing only 2 of dst's 4 bytes per sample (a strided partial
write) -- meaning every 4-byte block in the shared payload buffer needed
its own read-for-ownership TWICE (once per pol) instead of once. Fixed
with `copyQuantizedVHIntoPayload`, a combined pass writing all 4 bytes
of each sample together in one go (the actual common case: a noise-only
channel always sets both `VQuantized` and `HQuantized`) -- confirmed
faster via a same-machine before/after (not assumed): -17% just from
combining passes on this dev machine (Apple M5); the real-hardware
effect should be larger, since the strided-vs-combined difference is a
cache/memory-bandwidth-pattern effect that a warm-cache microbenchmark
loop understates relative to real, cold, high-throughput traffic.
Proven behavior-identical to the independent two-pass version via
`TestCopyQuantizedVHIntoPayload_MatchesTwoIndependentPasses`.

**A second, separate inefficiency found investigating the same
capture**: this run configured ONE tone source among 384 channels --
meaning `NewDirectSynthesisStreamer` built the full-precision complex64
bank at the FULL 384-channel width (~3.2GB, `DefaultNTiles`=256), even
though only that ONE channel's column was ever going to be read. Fixed:
the complex64 bank is now sized to `len(ComplexPathChannelIDMap())`
(the tone-affected subset, here: 1), not `numChannels` -- for this exact
scenario, ~3.2GB down to ~8.4MB, and the Box-Muller work needed to fill
it drops by the same ~384x factor. This wasn't visible in the
noise-only-no-tone captures earlier in this file (the complex64 bank
was already skipped ENTIRELY when zero channels have a tone -- this
regression only shows up in a MIXED config, exactly what this 90s test
happened to use).

**On "still saw the producer drifting, especially at the start"**: both
fixes above are plausible, genuine contributors, but for different
reasons, and a single aggregate 90-second profile can't distinguish
"worse at the start vs. steady throughout" on its own (it has no
per-time-bucket resolution) -- so this is reasoned diagnosis, not
confirmed:
- `copyQuantizedIntoPayload`'s inefficiency is a STEADY-STATE per-tick
  cost -- it would contribute to drift throughout the whole scan, not
  specifically at the start. Now fixed regardless.
- The oversized complex64 bank is a ONE-TIME CONSTRUCTION cost, paid
  entirely before `ScanRunner.Start()` begins ticking -- so it wouldn't
  directly cause tick-to-tick drift by itself. But it's exactly the kind
  of thing that COULD explain "worse specifically at the start": ~3.2GB
  of freshly `make()`'d memory has to be either zeroed or first-touched
  by the OS as the parallel fill workers write to it, and Linux commits
  those pages lazily -- if any of that settling (page faults, TLB
  pressure, allocator/GC catching up on a multi-GB allocation burst)
  bleeds into the first several ticks after `Start()`, it would show up
  as exactly "elevated drift early, self-corrects once the pages/caches
  are warm" -- without directly measuring it, this is a plausible
  mechanism, not a confirmed one. Also plausible and NOT mutually
  exclusive: this same target machine is independently documented (see
  the parent Python `CLAUDE.md`) as idling at 1.5GHz and needing
  sustained load before `schedutil` ramps clocks to boost -- the exact
  same "worse at the start, self-corrects" signature, for a completely
  unrelated reason.
- **What would actually distinguish these** (not done yet): the
  producer's own "falling behind pacing by Xs at tick N" log line
  already includes the tick number -- checking whether drift events
  cluster at LOW tick numbers specifically (vs. spread evenly across
  the whole 90s) would directly confirm or rule out a
  construction/warm-up-related cause vs. a steady-state one. Worth
  doing before assuming either explanation over the other.

### Tick-clustering analysis (`noise-stream.log`, 2026-09-08 13:44-ish CEST run, same config as above) -- distinguishes the two hypotheses

Did the check proposed above by parsing every "falling behind pacing"
line's tick number into a time-since-scan-start (`tick *
BlockDurationS`) and bucketing into 5s windows, instead of just eyeballing
the log. Answer: **overwhelmingly clustered at the start, not spread
evenly** -- 2641 total events across the 90s scan, 76% of them (≈2010)
land in the first ~18s, with a clean, near-zero gap from 18-27s and only
scattered, much rarer events afterward. Within that first-18s window the
per-second event count isn't monotonically decaying either -- it's a
repeating sawtooth (bursts of drift, brief recovery, another burst),
settling out entirely by ~18-20s.

This is the same "elevated for the first ~15-20s, then settles" signature
independently documented for this exact target machine in the parent
Python `CLAUDE.md` (idles at 1.5GHz, needs sustained load before
`schedutil` ramps to boost clocks -- that project's own benchmark sweep
script carries a 15s clock-ramp burn-in for exactly this reason). Not
proof by itself (a GC/allocator-warmup explanation would also plausibly
show an early-and-settling pattern), but a strong prior toward the
CPU-frequency-scaling hypothesis over the oversized-bank/GC hypothesis,
given the ~15-20s timescale match is specific, not just "early vs. late."

**Operationally more important than which hypothesis is right**: real
scans do not run back-to-back. Each station pod starts/stops per
integration test with multi-second gaps between successive scans (control
software overhead), and the full integration suite runs only a few times
a day -- easily enough idle time for `schedutil` to drop the CPU back
down between scans. So whichever mechanism this turns out to be, it is
NOT a one-off benchmarking artifact that only shows up on a cold process's
very first scan ever -- it will recur on every single real scan.

### CPU governor test: `performance` mode + Dell "HPC" BIOS profile (`cpu-384ch`/`cpu-384ch.pprof`/`noise-stream.log`, 2026-09-08 15:03-15:05 CEST, 90s scan, 384ch, one tone)

Directly tests the clock-ramp hypothesis above by removing the variable
it depends on: CPU governor switched from `schedutil` to `performance`
(fixed max frequency, no ramp-up delay) at the OS level, plus the Dell
R7525's "HPC" BIOS power profile enabled, then the same 90s/384-channel/
one-tone test re-run.

**Result: drift events dropped from 2641 to 301 -- an 8.8x reduction --
strongly confirming CPU clock ramp was a major real contributor, not a
red herring.** But it did NOT fully eliminate the pattern: of the 301
remaining events, 270 (90%) still land in the first 10s (237 in 0-5s, 33
in 5-10s), tapering to a scattered handful (1-4 per 5s window) through
the rest of the scan out to 78s. Max single-event drift was 25ms here vs.
18ms in the earlier (non-`performance`-governor) capture -- higher, not
lower, on the single worst event, though from 8.8x fewer samples, so this
is one data point, not yet repeatability-checked per this project's own
"don't trust a single-pass result" rule.

**Reading this**: `performance` mode/HPC removed most, but evidently not
all, of the early-scan drift -- with CPU frequency ramp now controlled
for, the STILL-clustered-at-start residual (90% of what's left, in the
same 0-10s window) is now better evidence for the secondary hypothesis
(GC/allocator/page-fault settling right after `Start()`) than it was
before, since the dominant confound has been removed. `Duration:
90.39s, Total samples = 307.21s (339.87%)` in this capture's profile is
an aggregate over the whole scan (no per-time-bucket resolution, same
limitation as the 14:13 capture above) and so can't itself confirm
this -- the profile's top costs
(`internal/runtime/syscall/linux.Syscall6` 42.20% flat,
`spead.copyQuantizedVHIntoPayload` 29.29% flat, `runtime.memmove` 9.42%
flat) are steady-state send/copy work, consistent with earlier captures,
not something new at 90s duration.

**Next step to actually settle the residual, not yet done**: `GODEBUG=
gctrace=1` is a Go runtime environment variable read at process startup
-- it works on the already-compiled binary with no Go toolchain needed on
the target host (just prepend it when invoking, e.g. `sudo GODEBUG=
gctrace=1 ./noise-stream-linux-amd64 ...` or `sudo env GODEBUG=gctrace=1
...` if plain `sudo` strips the environment). Since `performance` mode
now controls for clock ramp, a short (~15-20s is enough to cover the
warm-up window) capture with `gctrace=1` would give a much cleaner signal
than before on whether GC cycles specifically cluster in the first 5-10s
to match the remaining drift.

**This also reframes the fix target, not just the diagnosis.** Given
scans genuinely cold-start every time in production (see above), "accept
it as a warm-up characteristic" isn't sufficient even now that it's
smaller. Two real fix directions, not mutually exclusive, deliberately
not implemented yet pending a decision:
1. **Infrastructure-level (this session's `performance`/HPC change is a
   first step here)**: keep pinning CPU governor/power profile at the
   node level rather than working around it in application code -- the
   8.8x reduction already measured (2641 events down to 301) suggests
   this is the higher-leverage fix if it can be made a standard part of
   how these nodes are provisioned. Outside this codebase's control:
   needs whoever owns the actual K8s node fleet, not just this one
   manually-configured test box.
2. **Application-level**: an explicit CPU/memory warm-up burn-in
   immediately before `ScanRunner.Start()` begins real pacing (mirroring
   the 15s burn-in this project's own Python benchmark sweep script
   already uses). Deliberately NOT implemented: this delays real heap
   output by however long the burn-in runs, which is a genuine
   operational-timing change (multi-station scan synchronization, CSP
   LMC's expectations of how quickly `StartScan` produces real data) that
   needs sign-off, not just an internal implementation swap -- unlike
   this project's usual "different implementation, same output" bar for
   unilateral changes.

### `GODEBUG=gctrace=1` result: GC ruled out (`noise-stream.log`, 2026-09-08 13:11 CEST run, `performance`/HPC still on)

Ran with `GODEBUG=gctrace=1` (works on the compiled binary directly, no Go
toolchain needed on the target -- confirmed: set it as part of the
invocation, e.g. `sudo env GODEBUG=gctrace=1 ./noise-stream-linux-amd64
...`, since plain `VAR=val sudo cmd` doesn't survive most sudoers'
`env_reset`). **GC is not the cause.** Only 10 GC cycles across the whole
90s scan, every one with a sub-2ms total STW pause (e.g. `gc 4 @12.899s:
0.34+1.6+0.059 ms clock`), and after the first three (heap ramping up to
its steady-state ~787MB live size once, not per-tick) they're spaced
evenly roughly every 12s across the ENTIRE scan -- not clustered early the
way the drift is. That run's drift was still 91% concentrated in the
first 5s (210 of 231 events), with no GC cycle anywhere near large enough
to plausibly cause it. A second, un-gctraced repeat (231 -> 136 events,
continuing the run-to-run improvement trend) showed the identical
first-5s-dominant shape. GC/allocator warm-up is eliminated as a
hypothesis.

### CPU frequency trace: clock ramp ALSO ruled out under `performance`/HPC (`cpu_freq.log`, 2026-09-08 15:24-15:28 CEST, 384ch, one tone, 90s scan)

Built `scripts/capture_cpu_freq.sh` (not committed) to log per-core
frequency at 0.2s resolution to a file instead of requiring a live-watched
terminal -- first attempt silently produced an empty file (this exact
EPYC's `amd_pstate` driver doesn't populate `scaling_cur_freq`, and
`turbostat` wasn't installed on the target, so the fallback path had
nothing to poll); fixed by adding a `/proc/cpuinfo` "cpu MHz" fallback
(near-universal on x86 Linux) plus explicit diagnostics so a future empty
file fails loudly instead of silently.

Re-ran and got 21,409 real samples across 48 cores over the full 90s.
Bucketed into 1s windows and compared against the matching
`noise-stream.log`'s own timestamp (epoch-aligned via the ~2h UTC/CEST
offset between the two machines' clocks): **mean per-core frequency is
essentially flat the entire scan, ~2860-2920MHz from t=0s straight
through t=90s, with no ramp-up shape at all** -- t=0's mean (2950MHz) is
if anything slightly HIGHER than several later buckets (e.g. t=50s:
2858MHz), the opposite of what a clock-ramp hypothesis predicts. Per-core
min/max within each 1s bucket varies widely (roughly 1700-3900MHz), but
that's ordinary boost/idle variation across 48 cores under a workload
that doesn't peg every core at 100% simultaneously, not a systematic
early-vs-late difference.

**This rules out CPU clock ramp too, under `performance` governor +
HPC.** Between this and the GC result above, BOTH originally proposed
hypotheses for the residual first-5-10s drift clustering are now
eliminated -- the governor/HPC change's real 8.8x-and-climbing event
reduction across repeated runs was real, but whatever's left isn't either
of the two things that reduction was originally attributed to controlling
for.

### New hypothesis, implemented as a candidate fix: cold `sync.Pool`s at scan start

With clock-ramp and GC both ruled out, the next candidate is
`sampleBufferPool`/`quantizedBufferPool` (`internal/common/
heap_accumulator.go`) starting completely empty every time a scan begins
-- not just on a fresh process (which is all `noise-stream`, a one-shot
CLI, ever exercises), but in the real long-running device-server pod too:
Go's runtime drops every `sync.Pool` entry on EVERY GC cycle, and given
real scans have multi-second-plus gaps between them (control-software
overhead, per the parent Python `CLAUDE.md`'s documented cadence), it's
close to certain at least one GC cycle lands in that gap -- so the pools
are just as cold at the start of scan N+1 as they were for scan 1. Until
enough buffers have cycled through `ReleaseSampleBuffers` to refill the
pools "for free," every `Get()` in that window pays `make()`'s cost
instead -- a plausible, previously un-considered mechanism that fits the
observed "elevated only for the first several hundred/thousand ticks,
then settles and never recurs mid-scan" shape exactly, without touching
CPU frequency or GC at all.

**Fixed** (this is a pure implementation-detail change -- same output,
same wire content, so implemented directly per this project's standing
autonomy rule rather than asking first): added `common.WarmBufferPools
(numChannels int)`, called once from `NewScanRunner` before `Start()` can
ever begin ticking. Puts `numChannels*2` fresh buffers into EACH pool
(covering V+H for every channel regardless of which pool a given
channel's path actually draws from -- an oversized Put on the "wrong"
pool is harmless, just a few extra entries that age out on the next GC
like anything else already in the pool). Verified: `go build`/`go vet`/
`go test ./...`/`go test -race ./internal/common/... ./internal/synth/...`
all clean.

**Not yet target-hardware-validated** -- this is a candidate fix for the
now-GC-and-clock-ramp-eliminated residual, not a confirmed one. Next real
step: re-run the same 90s/384ch/one-tone test on the target server with
this change and compare drift-event count/clustering against the 136-event
baseline above. If the first-5s cluster shrinks substantially, that
confirms cold pools as the (or a) real mechanism; if it doesn't move,
something else is still at play (goroutine/OS-thread pool spin-up to fill
48 `P`s is the next candidate, since it wouldn't show up in either GC
trace or CPU frequency either).

### Target-hardware validation: `WarmBufferPools` -- real, substantial improvement (`noise-stream.log`, 2026-09-08 13:34 CEST, 90s scan, 384ch, one tone, `performance`/HPC still on)

**Confirms cold `sync.Pool`s were a real, meaningful contributor, not a
dead end.** Total drift events: 51, down from the 136-event baseline
immediately prior (a further 2.7x reduction on top of everything else
this session -- the full chain across this session's fixes is now 2641 ->
301 -> 231 -> 136 -> **51**, roughly 52x from where it started). Max
single-event drift also dropped to 7ms, the lowest yet.

**More important than the count: the SHAPE changed, not just the
total.** Every prior capture (pre- and post-`performance`-governor alike)
had its very first drift event within the first handful of ticks (tick 3,
tick 357, etc. -- essentially immediately). This run's first drift event
doesn't happen until **tick 1001, ~2.2s into the scan** -- a genuinely
different signature, not just a scaled-down version of the same one. The
remaining 51 events are also much less front-loaded as a fraction: 23 in
0-5s (45%, vs. 72-91% in every earlier capture), with the largest single
5s bucket now at 30-35s (17 events) rather than 0-5s -- checked the
matching `cpu_freq.log` for that window and the whole capture stays flat
at the same ~2860-2950MHz mean seen everywhere else with no anomaly
around 30-35s either (note: this run's derived clock-offset between the
two logs came out ~433s different from the prior run's -- worth treating
with a little less certainty than the earlier alignment, but doesn't
change the conclusion, since NOTHING in the entire frequency trace stands
out anywhere, regardless of exactly which window maps to which).

**Reading this**: pool cold-start was a real, independent contributor
alongside (not instead of) clock ramp -- fixing it didn't just reduce
drift, it eliminated the immediate-first-tick burst specifically,
consistent with the "pays make() until the pool fills" mechanism this fix
targets. There's still a smaller residual (roughly half the remaining
events in 0-10s, plus that one unexplained 30-35s bump), so this isn't
fully solved -- but three real, independent causes have now been found
and addressed (clock ramp via infra config, GC ruled out entirely, cold
pools via this fix), each confirmed by actually measuring before/after
rather than assumed. Whether to keep chasing the residual (goroutine/
OS-thread spin-up remains the next candidate) or treat ~51 events/90s
scan (mostly sub-10ms) as acceptable for this project's actual test
tolerance is a call for whoever owns the CBF-side acceptance criteria,
not something to keep optimizing blind.
