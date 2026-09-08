# Builds the Tango device server (tango/src/ska_low_station_beam_simulator,
# a gRPC CLIENT of the Go simulator -- see README.md) on top of SKAO's
# standard ska-tango-images Python base. Build context is the repo root
# -- pyproject.toml/uv.lock live there too, alongside the unrelated Go
# simulator (excluded via the Dockerfile.dockerignore next to this
# file). Adapted from
# https://gitlab.com/ska-telescope/ska-low-csp-testware/-/blob/main/Dockerfile,
# with COPY paths adjusted for this repo's tango/src layout instead of a
# top-level src/.
#
#   docker build -f images/tango-device-server/Dockerfile -t station-beam-simulator-tango .

# syntax=docker/dockerfile:1

ARG BUILD_IMAGE="artefact.skao.int/ska-build-python:0.3.3"
ARG BASE_IMAGE="artefact.skao.int/ska-tango-images-tango-python:0.4.1"

FROM $BUILD_IMAGE AS build

# Copy uv binaries
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/

ENV UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install runtime dependencies from uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy project code. pyproject.toml declares its readme and wheel
# package as "tango/README.md" and "tango/src/..." (both resolved
# relative to pyproject.toml itself), so that layout has to be
# preserved under /app rather than flattened -- a multi-source COPY
# into a directory copies each source in by basename, which would drop
# the "tango/" prefix.
COPY pyproject.toml uv.lock /app/
COPY tango/README.md /app/tango/README.md
COPY tango/src /app/tango/src

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

FROM $BASE_IMAGE

USER root

# Copy application from the build image
COPY --from=build /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Use `/app` as the working directory
WORKDIR /app
