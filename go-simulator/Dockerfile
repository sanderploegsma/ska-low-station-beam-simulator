# Builds cmd/simulator (the gRPC-served numeric core -- see README.md)
# into a minimal, non-root runtime image. Build context is THIS
# directory (go-simulator/), not the parent repo -- this module is a
# standalone Go module with its own go.mod/go.sum.
#
#   docker build -t station-beam-simulator-go .
#   docker run --rm -p 50051:50051 station-beam-simulator-go \
#       -listen :50051 -station-id 1 -dest-ip <cbf-host> -dest-port 8000

# syntax=docker/dockerfile:1

FROM golang:1.25-alpine AS build
WORKDIR /src

# Cache module downloads separately from source changes.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/simulator ./cmd/simulator

# distroless static + nonroot: no shell, no package manager, minimal
# attack surface for a statically-linked Go binary that only needs to
# open a gRPC listen socket and a UDP socket to CBF.
FROM gcr.io/distroless/static-debian12:nonroot AS runtime

COPY --from=build /out/simulator /usr/local/bin/simulator

# gRPC control-plane port (see cmd/simulator's -listen flag default).
EXPOSE 50051

ENTRYPOINT ["/usr/local/bin/simulator"]
