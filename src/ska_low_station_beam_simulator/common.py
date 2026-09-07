"""
Shared plumbing used by ``direct_synthesis.py``'s ``DirectSynthesisStreamer``
(the sole signal-generation backend): config constants, delay polynomial
handling, station/heap data structures, and the producer/sender plumbing
driven by ``ScanRunner``. SPEAD-64-48 heap encoding itself lives in
``spead.py`` (``SpsPacketizer`` and friends), not here. This module's
development history (deleted legacy paths, past incorrect assumptions
and how they were found and fixed) lives in CLAUDE.md, not here — this
docstring describes only the design as it stands.

Deliberately backend-agnostic: this module never imports
``direct_synthesis.py``. The streamer class exposes a small uniform
surface (``channel_id_map``, ``num_channels``, ``tick_n_samples()``,
``generate_next_tick()``) that ``ScanRunner`` relies on instead of
importing/isinstance-checking a concrete class — see ``ScanRunner``
below. Kept this way so ``common.py`` stays testable and reusable
independent of which generation strategy is behind it.

ALSO UNVERIFIED:

    - ``HeapAccumulator``: buffers per-channel samples until 2048/channel
      are available, then emits one heap per channel.
    - TAI2000 heap_counter conversion (astropy-based, with a heavily
      caveated non-astropy fallback).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from ska_low_station_beam_simulator.spead import SpsPacketizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cbf_sim")


# ============================================================
# CONFIG — genuinely ICD-fixed constants shared by every source type
# direct_synthesis.py generates.
# ============================================================

CHANNEL_WIDTH_HZ = 781_250.0  # per SPS-CBF ICD coarse channel spacing — CONFIRMED (channel centre spacing)
CHANNEL_START = 64  # CONFIRMED against the real ICD text: the lowest frequency channel is channel 64, centre frequency 50MHz

# CONFIRMED against the real ICD text (not a screenshot, not an
# assumption): the band is channelized as 384 equispaced coarse channels
# spanning 50-350MHz sky frequency, configurable in steps of 8 from 8 to
# 384 channels. Previously assumed 448 channels (350MHz) as "the full
# band" -- WRONG, 384*CHANNEL_WIDTH_HZ = 300MHz is the actual maximum;
# 448 channels was never a real configuration this hardware supports.
# See CLAUDE.md's SPS-CBF ICD section for the correction and its
# knock-on effect on every "448 channels" benchmark number from earlier
# sessions (left in CLAUDE.md as historical record of what was actually
# measured then, not retroactively rewritten).
MIN_NUM_CHANNELS = 8
MAX_NUM_CHANNELS = 384
NUM_CHANNELS_STEP = 8

# CONFIRMED against the real ICD text: "the lowest frequency channel is
# channel 64, centre frequency 50MHz" -- 64 * CHANNEL_WIDTH_HZ =
# 50,000,000 Hz exactly, confirming BASE_FREQ_HZ is meant as a CHANNEL
# CENTRE frequency (matching how it's used everywhere in this codebase,
# e.g. direct_synthesis.synth_tone_channel's
# `channel_center = base_freq_hz + channel_idx*channel_width_hz`), not a
# band edge. Previously assumed channel 65 / 50.78125MHz -- WRONG, off by
# one channel; channel 64 is the true lowest channel. (The band's actual
# lower EDGE -- distinct from channel 64's CENTRE -- is
# 50e6 - CHANNEL_WIDTH_HZ/2 = 49,609,375 Hz, matching the ICD's stated
# "lower edge of this channel is 49.61MHz"; nothing in this codebase
# needs that edge value directly, since every per-channel frequency
# calculation here is expressed in terms of channel CENTRES.)
# StationConfig.first_channel_id (64 by default, see below) is what maps
# local channel 0 onto the correct GLOBAL channel ID 64 in the
# wire-packed channel_info.
BASE_FREQ_HZ = 50.0e6

HEAP_LEN = 2048  # time samples per heap per channel, per ICD

# CONFIRMED against the real ICD text: each channel's actual per-sample
# period is 1080ns (1.25ns ADC sample period * 1024 * 27/32), NOT the
# naive 1/CHANNEL_WIDTH_HZ (1280ns) a CRITICALLY sampled channelizer
# would give. The polyphase filterbank deliberately OVERSAMPLES by a
# factor of 32/27 (more output samples per unit time than critical
# sampling, hence a SHORTER sample period by 27/32) -- this is a real
# hardware property of the SPS filterbank, not a simulator design choice,
# and was previously missed entirely (channel_output_rate was assumed
# equal to channel_width_hz -- see direct_synthesis.py's
# DirectSynthesisStreamer, now fixed). CHANNEL_OUTPUT_RATE_HZ is the
# per-channel TIME-DOMAIN sample rate; CHANNEL_WIDTH_HZ remains correct
# as-is for FREQUENCY-DOMAIN channel spacing (unaffected by oversampling
# -- channel centres are still exactly CHANNEL_WIDTH_HZ apart). Verify:
# 1 / (CHANNEL_WIDTH_HZ * 32/27) = 1.08e-6 s = 1080ns exactly.
OVERSAMPLING_NUMERATOR = 32
OVERSAMPLING_DENOMINATOR = 27
CHANNEL_OUTPUT_RATE_HZ = (
    CHANNEL_WIDTH_HZ * OVERSAMPLING_NUMERATOR / OVERSAMPLING_DENOMINATOR
)

# Per-tick time budget, FIXED regardless of channel count — this is the
# real-time constraint generation is racing against. Equal to
# HEAP_LEN / CHANNEL_OUTPUT_RATE_HZ (the real, OVERSAMPLED per-channel
# sample rate -- NOT HEAP_LEN/CHANNEL_WIDTH_HZ, which would assume
# critical sampling and give a budget ~15.6% too generous: 2.621ms
# instead of the real 2.212ms). Also confirms HEAP_LEN satisfies the
# ICD's separate requirement that "the number of samples per packet
# shall be a multiple of the numerator of the oversampling ratio" (32):
# 2048 / 32 = 64 exactly.
BLOCK_DURATION_S = HEAP_LEN / CHANNEL_OUTPUT_RATE_HZ

OVERRUN_TOLERANCE = 2.0
QUEUE_MAXSIZE = 4096  # heaps are much smaller units than before — more of them
QUEUE_PUT_TIMEOUT_S = 0.5

# TAI2000 epoch: 2000-01-01T00:00:00 TAI. Confirmed by you as the SKA epoch.
TAI2000_EPOCH_ISO = "2000-01-01T00:00:00"


# ============================================================
# TIME CONVERSION — heap_counter = packet count since SKA (TAI2000) epoch
# ============================================================


def unix_to_tai2000_seconds(unix_time: float) -> float:
    """Seconds since 2000-01-01T00:00:00 TAI, given a Unix (UTC-based) time.

    Strongly prefer the astropy path — it handles leap seconds correctly.
    The fallback is a hardcoded leap-second count and WILL silently
    produce wrong results after the next leap second is inserted (or if
    run against historical dates before the hardcoded count applied).
    Do not deploy the fallback path without replacing it.

    :param unix_time: a Unix (UTC-based) timestamp, in seconds.
    :returns: seconds since the TAI2000 epoch.
    """
    try:
        from astropy.time import Time

        t = Time(unix_time, format="unix", scale="utc")
        epoch = Time(TAI2000_EPOCH_ISO, scale="tai")
        return (t.tai - epoch).sec
    except ImportError:
        log.warning(
            "astropy not installed — using an APPROXIMATE fixed TAI-UTC "
            "leap-second offset (37s, valid as of ~2017-2026 with no new "
            "leap seconds inserted). This is not safe for production. "
            "Install astropy."
        )
        TAI_UTC_OFFSET_S = 37.0  # verify current value before trusting this path at all
        UNIX_TIME_AT_TAI2000_EPOCH = 946684800.0 - 32.0  # approx, see docstring
        return (unix_time + TAI_UTC_OFFSET_S) - UNIX_TIME_AT_TAI2000_EPOCH


# ============================================================
# DELAY POLYNOMIAL (per ska-low-csp-delaymodel/1.0 schema)
# ============================================================


@dataclass
class DelayPolynomial:
    station_id: int
    start_validity_sec: float
    validity_period_sec: float
    xypol_coeffs_ns: list[float]
    ypol_offset_ns: float

    @property
    def valid_until(self) -> float:
        return self.start_validity_sec + self.validity_period_sec

    def eval_delay_seconds(self, t: float, pol: str) -> float:
        """IMPORTANT: evaluated relative to ``start_validity_sec``, NOT
        raw absolute epoch time — a 5th-order poly blows up otherwise
        (this was a real bug, caught by a smoke test, present in earlier
        iterations of this design too). ``t_rel`` should stay within
        roughly [0, validity_period_sec].

        :param t: absolute epoch time, in seconds.
        :param pol: ``"V"`` or ``"H"`` — ``"H"`` adds ``ypol_offset_ns``.
        :returns: delay, in seconds.
        """
        t_rel = t - self.start_validity_sec
        tau_x_ns = sum(c * t_rel**i for i, c in enumerate(self.xypol_coeffs_ns))
        tau_ns = tau_x_ns if pol == "V" else tau_x_ns + self.ypol_offset_ns
        return tau_ns * 1e-9


# ============================================================
# PER-SOURCE DELAY FEEDS
#
# Every sky source (a tone, a pulsar) simulated by DirectSynthesisStreamer
# needs its OWN delay polynomial — two sources at different directions
# genuinely have different geometric delay, and treating them identically
# would implicitly put every source at the same point in the sky. There is
# deliberately NO default/fallback feed: a source with no real delay path
# would silently produce content that's trivially "perfectly aligned" —
# exactly the kind of thing that could mask a real CBF delay-tracking bug
# rather than exercise it, given this simulator's whole reason for
# existing is generating true delay independently of CBF. Every tone/
# pulsed source_cfg MUST supply a `delay_feed`; DirectSynthesisStreamer
# raises at construction time otherwise (see its __init__).
# ============================================================

_ZERO_DELAY_COEFFS_NS = [0.0]


class DelayFeed:
    """Answers "what delay polynomial applies at time t" for one source —
    fed by a Tango CHANGE_EVENT subscription on one of CBF's delay-poly
    emulator's per-direction attributes (RA/Dec, Az/El, or static — see
    ``simulator.py``), though nothing here is Tango-specific: ``update()``
    just needs calling from whatever thread learns of a new polynomial
    (tests call it directly). ``get()`` is called from the generation
    thread. A plain reference swap is safe across threads under the GIL
    without an explicit lock — no field of the swapped-in
    ``DelayPolynomial`` is ever mutated in place, only the ``_poly``
    reference itself is replaced.

    Two deliberate behaviours, not oversights:

      - No polynomial received yet -> zero delay, warned ONCE (not every
        tick). A reasonable default for "hasn't started publishing yet"
        rather than blocking scan start on an external device being up.
      - Polynomial expired (t >= valid_until) with no replacement having
        arrived -> keep applying it as-is, warned once per staleness
        episode. Detecting or recovering from a stalled upstream
        publisher is explicitly NOT this simulator's job — it applies
        whatever delay it was actually given and logs when that delay is
        known to be stale, so the discrepancy is visible to whoever is
        debugging a test failure (see CLAUDE.md's Observability section:
        the same "surface it, don't paper over it" principle as the
        dropped-heap/pacing counters there).
    """

    def __init__(self, name: str):
        self.name = name
        self._poly: DelayPolynomial | None = None
        self._warned_no_poly = False
        self._warned_stale_valid_until: float | None = None

    def update(self, poly: DelayPolynomial) -> None:
        self._poly = poly
        self._warned_stale_valid_until = None

    def get(self, t: float) -> DelayPolynomial:
        if self._poly is None:
            if not self._warned_no_poly:
                log.warning(
                    "delay source %r has not received a polynomial yet — "
                    "applying zero delay until one arrives",
                    self.name,
                )
                self._warned_no_poly = True
            return DelayPolynomial(
                station_id=-1,
                start_validity_sec=t,
                validity_period_sec=float("inf"),
                xypol_coeffs_ns=_ZERO_DELAY_COEFFS_NS,
                ypol_offset_ns=0.0,
            )
        if (
            t >= self._poly.valid_until
            and self._warned_stale_valid_until != self._poly.valid_until
        ):
            log.warning(
                "delay source %r polynomial expired at t=%.3f "
                "(valid_until=%.3f) with no replacement received yet — "
                "continuing to apply the expired coefficients",
                self.name,
                t,
                self._poly.valid_until,
            )
            self._warned_stale_valid_until = self._poly.valid_until
        return self._poly


def parse_delay_polynomial_from_attr_value(value, station_id: int) -> DelayPolynomial:
    """UNVERIFIED WIRE FORMAT — same category of risk as this module's
    ICD bit-packing functions below. ``ska-low-csp-delaymodel/1.0`` is a
    documented schema (ADR-88 in ``ska-telmodel``) but the exact payload
    a real delay-poly Tango attribute pushes hasn't been checked against
    it here. Assumes ``value`` is a JSON string (or an already-parsed
    mapping) with keys matching ``DelayPolynomial``'s fields. Confirm
    against the real schema and the real CBF delay-poly emulator before
    deploying.

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


# ============================================================
# STATION CONFIG / CHANNEL HEAP — shared data structures, backend-agnostic
# ============================================================


@dataclass
class StationConfig:
    station_id: int
    substation_id: int
    subarray_id: int
    beam_id: int
    scan_id: int = 0


@dataclass
class ChannelHeap:
    channel_id: int
    v_samples: np.ndarray  # complex, shape (HEAP_LEN,)
    h_samples: np.ndarray  # complex, shape (HEAP_LEN,)
    heap_start_time: float  # sim_time (obs_time-relative seconds) of first sample


class HeapAccumulator:
    """Buffers per-channel samples (whatever backend produced them — a
    channelized FFT wave or direct synthesis, both look identical from
    here: a ``(n_new, num_channels)`` complex array per pol) until
    ``HEAP_LEN`` are available per channel, then emits one
    ``ChannelHeap`` per channel."""

    def __init__(
        self,
        num_channels: int,
        obs_time: float,
        sample_rate_per_channel: float,
        channel_id_map: np.ndarray | None = None,
    ):
        self.num_channels = num_channels
        self.obs_time = obs_time
        self.sample_rate_per_channel = sample_rate_per_channel
        self._buffers = {
            "V": np.zeros((0, num_channels), dtype=np.complex128),
            "H": np.zeros((0, num_channels), dtype=np.complex128),
        }
        self._samples_consumed = 0  # total samples already popped, for timestamping
        # Maps a column index in the arriving (n_new, num_channels) chunks
        # to the external channel_id to label it with. Identity for
        # DirectSynthesisStreamer, which always produces already-external-
        # order columns. Kept as an explicit, overridable map (rather than
        # assuming identity) so a future generation strategy with its own
        # internal channel ordering (e.g. one built on an FFT-bin-order
        # intermediate) wouldn't require changing this class.
        self._channel_id_map = (
            channel_id_map if channel_id_map is not None else np.arange(num_channels)
        )

    def add(self, pol: str, chunk: np.ndarray):
        self._buffers[pol] = np.concatenate([self._buffers[pol], chunk], axis=0)

    def pop_ready_heaps(self) -> list[ChannelHeap]:
        heaps = []
        while (
            len(self._buffers["V"]) >= HEAP_LEN and len(self._buffers["H"]) >= HEAP_LEN
        ):
            v = self._buffers["V"][:HEAP_LEN]
            h = self._buffers["H"][:HEAP_LEN]
            self._buffers["V"] = self._buffers["V"][HEAP_LEN:]
            self._buffers["H"] = self._buffers["H"][HEAP_LEN:]

            heap_start_time = (
                self.obs_time + self._samples_consumed / self.sample_rate_per_channel
            )
            self._samples_consumed += HEAP_LEN

            for ch in range(self.num_channels):
                heaps.append(
                    ChannelHeap(
                        channel_id=int(self._channel_id_map[ch]),
                        v_samples=v[:, ch],
                        h_samples=h[:, ch],
                        heap_start_time=heap_start_time,
                    )
                )
        return heaps


# ============================================================
# PRODUCER / SENDER — drives the streamer via a small structural
# protocol rather than importing a concrete class.
#
# Backend-agnostic on purpose: a Streamer only needs to provide
# `channel_id_map`, `num_channels`, `tick_n_samples()`, and
# `generate_next_tick(t, n)` — this module never imports
# direct_synthesis.DirectSynthesisStreamer, the sole implementation.
# Kept as a Protocol (not a direct import) so this module stays testable
# independent of the generation strategy, and so a future alternative
# implementation wouldn't require changing this file.
# ============================================================


class Streamer(Protocol):
    @property
    def channel_id_map(self) -> np.ndarray: ...

    @property
    def num_channels(self) -> int: ...

    def tick_n_samples(self) -> int:
        """How many per-channel output samples ``generate_next_tick``'s
        second argument should be for one tick — sized so one tick
        produces close to exactly one heap's worth of per-channel
        samples (``BLOCK_DURATION_S`` is defined for exactly this).

        :returns: per-channel samples for one tick.
        """
        ...

    # Positional-only (``/``) so implementations can use their own, more
    # descriptive parameter names without tripping Protocol structural
    # matching on name. Mapping (not dict) for the return type since it's
    # covariant in the value type, allowing a future implementation to
    # return None for a channel with no data yet without breaking this
    # protocol.
    def generate_next_tick(
        self, t: float, n: int, /
    ) -> Mapping[str, np.ndarray | None]: ...


class ScanRunner:
    def __init__(
        self,
        streamer: Streamer,
        send_queue: queue.Queue[ChannelHeap],
        obs_time: float,
        scan_duration_s: float,
    ):
        self.streamer = streamer
        self.send_queue = send_queue
        self.obs_time = obs_time
        self.scan_duration_s = scan_duration_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        # Fixed by the ICD (the OVERSAMPLED per-channel sample rate --
        # see CHANNEL_OUTPUT_RATE_HZ's definition above for why this is
        # NOT CHANNEL_WIDTH_HZ), not by whichever backend is in use.
        self.channel_output_rate = CHANNEL_OUTPUT_RATE_HZ
        self.n_samples_per_tick = streamer.tick_n_samples()

        self.accumulator = HeapAccumulator(
            streamer.num_channels,
            obs_time,
            self.channel_output_rate,
            channel_id_map=streamer.channel_id_map,
        )

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 5.0):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def _run(self):
        wall_start = time.monotonic()
        n_ticks = int(self.scan_duration_s / BLOCK_DURATION_S)

        for tick in range(n_ticks):
            if self.stop_event.is_set():
                break

            sim_time = self.obs_time + tick * BLOCK_DURATION_S
            target_wall = wall_start + tick * BLOCK_DURATION_S
            now = time.monotonic()
            if now < target_wall:
                if self.stop_event.wait(timeout=target_wall - now):
                    break
            else:
                overrun = now - target_wall
                if overrun > BLOCK_DURATION_S * OVERRUN_TOLERANCE:
                    log.warning(
                        "producer falling behind pacing by %.3fs at tick %d",
                        overrun,
                        tick,
                    )

            raw_results = self.streamer.generate_next_tick(
                sim_time, self.n_samples_per_tick
            )
            for pol, chunk in raw_results.items():
                if chunk is not None:
                    self.accumulator.add(pol, chunk)

            for heap in self.accumulator.pop_ready_heaps():
                try:
                    self.send_queue.put(heap, timeout=QUEUE_PUT_TIMEOUT_S)
                except queue.Full:
                    log.warning(
                        "send queue full — dropping heap ch=%d t=%.4f",
                        heap.channel_id,
                        heap.heap_start_time,
                    )

        log.info("scan producer finished (stopped=%s)", self.stop_event.is_set())


def sender_loop(
    send_queue: queue.Queue[ChannelHeap],
    packetizer: SpsPacketizer,
    shutdown_event: threading.Event,
):
    while not shutdown_event.is_set():
        try:
            heap = send_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            packetizer.send_channel_heap(heap)
        except Exception:
            log.exception(
                "failed to send heap ch=%d t=%.4f",
                heap.channel_id,
                heap.heap_start_time,
            )
