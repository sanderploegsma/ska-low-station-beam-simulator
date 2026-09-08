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
