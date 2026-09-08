# SKA Low station-beam simulator

A software simulator generating SKA Low SPS station-beam data, replacing
the CNIC hardware/firmware tool currently used to test the CBF
correlator/beamformer's delay-tracking, correlation, and beamforming
logic. Independence from CBF matters here specifically: CNIC is built by
the same team that builds the CBF firmware being tested, which undermines
the value of the test. This simulator generates "true" delay
independently of whatever CBF itself computes.

Deployment target: one Tango device server per station, one Kubernetes
pod per device, sending real SPEAD/UDP heaps to CBF per the SPS-CBF ICD.

For the reasoning behind the decisions below, design iterations that were
tried and rejected, past bugs, and benchmark investigations, see
[`docs/history.md`](docs/history.md). For repo-specific guidance aimed at
an AI coding assistant working in this codebase, see
[`CLAUDE.md`](CLAUDE.md).

## Architecture

```
Tango device server (Python, tango/)
  - owns Tango: device properties, StartScan/StopScan command handling
  - subscribes to CBF's delay-poly emulator's Tango attributes
    (AttributeProxy + CHANGE_EVENT)
  - forwards every pushed delay update over gRPC (PushDelayUpdate)
        |
        | gRPC (api/simulator.proto)
        v
Go simulator process (cmd/simulator)
  - StartScan/StopScan/GetStatus/PushDelayUpdate (internal/server)
  - signal generation: tone + per-pol station noise (internal/synth)
  - SPEAD-64-48 heap encoding + UDP send (internal/spead)
  - real-time pacing loop (internal/common.ScanRunner)
        |
        | SPEAD/UDP
        v
      CBF
```

A Tango device server owns everything Tango-facing; a separate Go process
owns signal generation and SPEAD/UDP sending and has no Tango access of
its own. This split exists so the numerically-heavy generation/encoding
path isn't constrained by Python, while Tango integration (device
properties, attribute subscriptions) stays in the language SKA's Tango
tooling targets.

Settled architectural decisions (independent delay generation, CSP LMC
driving the scan rather than TMC, deterministic `sim_time`, one device
server per station/one pod per device, TAI2000 as the epoch for
`heap_counter`) are listed in `CLAUDE.md`; their background and rationale
are in `docs/history.md`.

**Current scope**: the Go simulator implements tone sources and per-pol
station (receiver) noise. Pulsar/pulsed-source generation is not
currently implemented in either language — an earlier Python prototype
existed (see `docs/history.md`) but was removed once tone/noise reached
parity in Go and nothing in the Tango device used it any longer;
reimplementing pulsar support (if still needed) should target the Go
side directly rather than reviving the old Python path.

## Repository layout

```
api/
  simulator.proto              gRPC service definition (Tango device <-> Go simulator)
  simulatorpb/                 generated Go stubs (protoc-gen-go/protoc-gen-go-grpc)

cmd/
  simulator/                   gRPC-served entrypoint: StartScan/StopScan/GetStatus/PushDelayUpdate
  noise-stream/                standalone CLI: noise (+ optional single tone), no gRPC, no Tango

internal/
  common/                      backend-agnostic plumbing: DelayPolynomial/DelayFeed, StationConfig/
                                ChannelHeap, HeapAccumulator, ScanRunner (driven through a small
                                Streamer interface -- this package never imports internal/synth)
  synth/                       DirectSynthesisStreamer: tone + noise-tile-bank generation kernels
  spead/                       SPEAD-64-48 heap encoding (SpsPacketizer) + batched UDP sending
  netutil/                     resolves a named network interface's IPv4 address (Multus support)
  server/                      the gRPC service itself, wiring common/synth/spead/netutil together

tango/
  src/ska_low_station_beam_simulator/
    common.py                  logging setup + delay-polynomial parsing, used by simulator.py
    simulator.py                the Tango device server (StationSimulatorDevice) -- a gRPC CLIENT
                                of the Go simulator, not a local signal generator
    simulatorpb/                generated Python gRPC stubs (grpc_tools.protoc), from api/simulator.proto
  tests/                        pytest suite for simulator.py's Tango-facing wiring (no live Tango
                                context or live gRPC server; fakes/monkeypatches instead)

docs/
  history.md                   design history, past bugs, benchmark/profiling investigations

go.mod / go.sum / Dockerfile    Go module + container image, at the repo root
pyproject.toml / uv.lock        Python package, also at the repo root (see "Setup" below for why)
```

`go.mod`/`go.sum` and `pyproject.toml`/`uv.lock` intentionally sit
side by side at the repo root — this is one repository containing both
halves of the split described above, not two separate projects.

## SPS-CBF ICD reference facts

Per the real ICD text (not a screenshot, not carried forward from an
earlier guess — see `docs/history.md` for the corrections this required):

- The band is channelized as **384 equispaced coarse channels**,
  configurable from **8 to 384 in steps of 8**.
- The lowest channel is **global ID 64**, centre frequency **50.0 MHz**
  exactly. Channel spacing (`CHANNEL_WIDTH_HZ`) is 781.25 kHz.
- Each channel's actual per-sample period is **1080 ns**, not the 1280 ns
  a critically-sampled channelizer would give — the polyphase filterbank
  oversamples by a factor of 32/27 (`CHANNEL_OUTPUT_RATE_HZ` ≈
  925,925.93 Hz). This sets the real per-tick real-time budget
  (`BLOCK_DURATION_S` = `HEAP_LEN` / `CHANNEL_OUTPUT_RATE_HZ` ≈
  2.21184 ms), not the channel spacing.
- One heap = one channel = 2048 consecutive time-domain samples
  (`HEAP_LEN`), both polarisations interleaved per sample (Vreal, Vimag,
  Hreal, Himag, each int8). `packet_payload_length` is fixed at `0x2000`
  (8192 bytes = 2048 × 4 bytes).
- Six SPEAD-64-48 immediate item pointers per heap, no more:

  | item ID | bit layout |
  |---|---|
  | `0x0001` | 8 bits reserved \| 40 bits `heap_counter` |
  | `0x0004` | 48 bits `packet_payload_length` (fixed: `0x2000`) |
  | `0x3010` | 48 bits `scan_id` |
  | `0x3000` | 16 bits reserved \| 16 bits `beam_id` \| 16 bits `frequency_id` |
  | `0x3001` | 8 bits `substation_id` \| 8 bits `subarray_id` \| 16 bits `station_id` \| 16 bits reserved |
  | `0x3300` | 48 bits `payload_offset` (fixed: `0x0` — heaps are always exactly one packet) |

  ...followed immediately by the 8192-byte interleaved V/H I/Q payload at
  a fixed byte offset (56 bytes in) — no 7th "payload" item pointer; CBF
  firmware reads the payload directly rather than doing a generic SPEAD
  parse. This is why `internal/spead`/`tango/src/.../simulator.py`'s
  encoding path is hand-rolled rather than built on a general-purpose
  SPEAD library — see `docs/history.md` for why spead2 specifically
  cannot produce this format.

## Setup

### Go (signal generation + gRPC service)

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

Build, vet, test:

```
go build ./...
go vet ./...
go test ./...
```

Run the gRPC-served simulator:

```
go run ./cmd/simulator -listen :50051 -station-id 1 -dest-ip 127.0.0.1 -dest-port 8000
```

`cmd/simulator`'s flags mirror `simulator.py`'s static device properties
(`station_id`/`substation_id`/`dest_ip`/`dest_port`); everything that
varies per scan (`subarray_id`, `beam_id`, tone sources, noise config) is
a `StartScan` gRPC request field instead, matching the Python
`StartScan` JSON argument's shape.

For quickly exercising the numeric core + SPEAD packetizer end-to-end
(e.g. against a packet capture tool, or CBF's receive path) without
standing up the gRPC service or a Tango-facing counterpart process at
all, use `cmd/noise-stream`:

```
go run ./cmd/noise-stream -dest-ip 127.0.0.1 -dest-port 8000 -scan-duration 10
```

No gRPC, no real delay sources — just the noise tile bank (and,
optionally, one tone source via `-tone-freq-hz`), driven by a
`ScanRunner` exactly like a real scan. Run `go run ./cmd/noise-stream -h`
for the full flag list (station/subarray/beam/scan IDs, channel count,
obs time, scan duration, noise std/seed, sender-goroutine count,
`-spead-interface` for binding to a named network interface such as a
Multus-attached secondary NIC).

### Python (Tango device server)

Python 3.10 (`.python-version`), dependency management via `uv`
(`pyproject.toml`/`uv.lock`). One private index is configured
(`artefact.skao.int`, for `ska-tango-base`) — confirm network access to
it from wherever you're running this before `uv sync`.

```
uv sync
uv run pytest
```

`pytango` isn't required to run the above — `simulator.py` degrades to
stub Tango classes if `pytango` isn't installed (importable, not
deployable).

Regenerating the Python gRPC stubs, after editing `api/simulator.proto`:

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
# from inside the simulatorpb package:
sed -i '' 's/^import simulator_pb2 as simulator__pb2$/from . import simulator_pb2 as simulator__pb2/' \
    tango/src/ska_low_station_beam_simulator/simulatorpb/simulator_pb2_grpc.py
```

`grpcio-tools` (the `dev` dependency group) provides `grpc_tools.protoc` —
no separate `protoc`/plugin binaries to install beyond what `uv sync`
already pulls in, unlike the Go side's regeneration command.

### Container image

`Dockerfile` builds `cmd/simulator` into a minimal, non-root image
(`golang:1.25-alpine` build stage, `gcr.io/distroless/static-debian12:nonroot`
runtime — no shell, no package manager, matching this binary's actual
needs: a gRPC listen socket and a UDP socket to CBF). Build context is
the repo root (`.dockerignore` excludes the unrelated Python simulator
under `tango/`):

```
docker build -t station-beam-simulator-go .
docker run --rm -p 50051:50051 station-beam-simulator-go \
    -listen :50051 -station-id 1 -dest-ip <cbf-host> -dest-port 8000
```

`.github/workflows/go-simulator-docker.yml` runs `go build`/`go vet`/
`go test -race` on every push/PR touching the Go sources at the repo
root, then builds and publishes a multi-arch (`linux/amd64`,
`linux/arm64`) image to `ghcr.io/<owner>/<repo>/go-simulator` on pushes to
`main` (tag `latest`) and on `go-simulator-v*` tags (semver tag) — pull
requests build the image (to catch a broken Dockerfile early) but never
push it. There is no equivalent image/CI for the Python side yet.

## Testing notes

`go test ./...` covers:

- **`internal/synth`**: tone lands in the expected channel with an exact
  (not first-order-approximated) delay-as-phase shift; the noise tile
  bank is deterministic given the same seed, statistically matches its
  configured std, and is independent across different (station, pol)
  seeds; noise is never delay-corrected, matching the physical
  requirement that receiver noise originates after any signal-path delay.
- **`internal/spead`**: encoded heaps decode back to exactly the ICD's 6
  items (byte-level, independent of the encoder's own logic) with the
  correct bit-packed `channel_info`/`antenna_info`; a `heap_counter`
  overflow is rejected, not silently truncated.
- **`internal/common`**: `DelayFeed`'s zero-delay-until-first-update and
  keep-applying-stale-polynomial behaviours; `HeapAccumulator` framing
  and timestamping; `ScanRunner`'s pacing loop against a fake streamer.
- **`internal/server`**: the gRPC surface end-to-end over an in-memory
  `bufconn` listener (`StartScan`/`StopScan`/`GetStatus`/
  `PushDelayUpdate`, including validation error codes); `Start()`
  actually binding to a real (loopback) interface, and failing for an
  unknown one.
- **`internal/netutil`**: resolving a real (loopback) interface's IPv4
  address, and erroring for an unknown interface name — portable across
  Linux/macOS (looked up by interface flag, not a hardcoded name like
  `"lo"`/`"lo0"`).

Not covered: a live end-to-end SPEAD capture decoded by an external tool;
real multi-pod co-scheduling on one physical node (every real-hardware
measurement so far has been one process alone on the target box — see
`docs/history.md`).

`uv run pytest` covers `simulator.py`'s Tango-facing wiring only —
`build_tone_source_request`'s JSON-boundary validation (rejects
non-`'tone'` kinds, missing `delay_attr_uri`) and the delay-poly
attribute subscription/forwarding wiring (against fake
`AttributeProxy`/gRPC-stub objects, not a live Tango context or a live Go
process). This codebase doesn't unit test the Tango device server layer
itself anywhere.

## Known limitations / open items

- **Pulsar/pulsed-source generation isn't implemented** in either
  language right now (see "Current scope" above).
- **The exact CSP LMC command for pushing a delay model without going
  through TMC is still unconfirmed.**
- **The delay-poly attribute's exact wire payload shape, and whether
  `AttributeProxy` delivers an immediate `CHANGE_EVENT` with the
  attribute's current value on subscribe (vs. only on the next actual
  change), are both unverified** against a real delay-poly emulator —
  `common.parse_delay_polynomial_from_attr_value` assumes a JSON
  string/mapping matching `DelayPolynomial`'s fields.
- **Real multi-pod co-scheduling hasn't been validated** — the Go
  pacing/profiling investigation (see `docs/history.md`) was done with
  one `noise-stream`/`simulator` process alone on the target box.
  Residual pacing drift after that investigation was small enough
  (occasional sub-10ms events on a 90s/384-channel/one-tone test) and
  recoverable (a slow tick's overrun never compounds into the next
  tick — see `docs/history.md`'s "Decision: pausing here") that further
  optimization was paused rather than pursued further; revisit if a real
  co-scheduled test shows worse behaviour or if CBF's actual
  delay-tracking test tolerance turns out to need tighter margins.
- **The "zero phase at first sample of every SPS SPEAD packet" ICD
  passage** has been reasoned through as a normalization convention (see
  `internal/synth/tone.go` and `docs/history.md`) but not independently
  verified against real hardware/firmware.
