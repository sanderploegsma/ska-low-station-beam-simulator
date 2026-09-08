"""
Shared plumbing for the Tango-facing device server (``simulator.py``):
logging setup and delay-polynomial parsing.

Signal generation, heap accumulation, and SPEAD/UDP sending all live in
the Go process this device drives over gRPC (see the repo root's
``cmd/server``) — nothing in this module generates content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cbf_sim")


@dataclass
class DelayPolynomial:
    """A ``ska-low-csp-delaymodel/1.0`` delay polynomial for one source."""

    station_id: int
    start_validity_sec: float
    validity_period_sec: float
    xypol_coeffs_ns: list[float]
    ypol_offset_ns: float


def parse_delay_polynomial_from_attr_value(value, station_id: int) -> DelayPolynomial:
    """Parses one delay-poly attribute push into a ``DelayPolynomial``.

    UNVERIFIED WIRE FORMAT: ``ska-low-csp-delaymodel/1.0`` is a documented
    schema (ADR-88 in ``ska-telmodel``) but the exact payload a real
    delay-poly Tango attribute pushes hasn't been checked against it here.
    Assumes ``value`` is a JSON string (or an already-parsed mapping) with
    keys matching ``DelayPolynomial``'s fields. Confirm against the real
    schema and the real CBF delay-poly emulator before deploying.

    :param value: a JSON string or already-parsed mapping with keys
        matching ``DelayPolynomial``'s fields.
    :param station_id: the station this polynomial applies to.
    :returns: the parsed ``DelayPolynomial``.
    """
    import json

    data = json.loads(value) if isinstance(value, str) else value
    return DelayPolynomial(
        station_id=station_id,
        start_validity_sec=float(data["start_validity_sec"]),
        validity_period_sec=float(data["validity_period_sec"]),
        xypol_coeffs_ns=[float(c) for c in data["xypol_coeffs_ns"]],
        ypol_offset_ns=float(data["ypol_offset_ns"]),
    )
