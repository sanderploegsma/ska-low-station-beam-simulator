"""
Shared plumbing for the Tango-facing device server (``simulator.py``):
logging setup and delay-polynomial parsing.

Signal generation, heap accumulation, and SPEAD/UDP sending all live in
the Go process this device drives over gRPC (see the repo root's
``cmd/server``) — nothing in this module generates content.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DELAYMODEL_SCHEMA_URL = "https://schema.skao.int/ska-low-csp-delaymodel/1.0"


@dataclass
class DelayPolynomial:
    """A ``ska-low-csp-delaymodel/1.0`` delay polynomial for one source."""

    station_id: int
    start_validity_sec: float
    validity_period_sec: float
    xypol_coeffs_ns: list[float]
    ypol_offset_ns: float


def parse_delay_polynomial_from_attr_value(
    value: str,
    station_id: int,
) -> DelayPolynomial | None:
    """Parses one delay-poly attribute push into a ``DelayPolynomial``.

    :param value: a JSON string matching the ``ska-low-csp-delaymodel/1.0`` schema.
    :param station_id: the station this polynomial applies to.
    :returns: the parsed ``DelayPolynomial`` or ``None`` if the station ID is not found.
    """
    logger.debug(
        "Parsing delay polynomial for station %d from value: %s", station_id, value
    )
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Failed to parse delay polynomial JSON")
        return None

    if data["interface"] != DELAYMODEL_SCHEMA_URL:
        logger.warning("Unsupported delay polynomial interface: %s", data["interface"])
        return None

    for delay in data.get("station_beam_delays", []):
        if delay["station_id"] == station_id:
            return DelayPolynomial(
                station_id=station_id,
                start_validity_sec=float(data["start_validity_sec"]),
                validity_period_sec=float(data["validity_period_sec"]),
                xypol_coeffs_ns=[float(c) for c in delay["xypol_coeffs_ns"]],
                ypol_offset_ns=float(delay["ypol_offset_ns"]),
            )

    logger.warning("No delay polynomial found for station ID %d", station_id)
    return None
