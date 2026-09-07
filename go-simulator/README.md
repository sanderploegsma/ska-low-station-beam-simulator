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
  `PushDelayUpdate`, including the validation error codes).

Not covered (left for real hardware/integration testing, same as the
Python project's own stated gaps): actual throughput/timing benchmarks
against `common.BlockDurationS` on target server hardware, and a live
end-to-end SPEAD capture decoded by an external tool.
