"""
Shared plumbing used by direct_synthesis.py's DirectSynthesisStreamer
(the sole signal-generation backend — the legacy wideband+FFT
StationStreamer this simulator used to fall back to for pulsed sources
has been removed, see CLAUDE.md): config constants, delay polynomial
handling, station/heap data structures, SPEAD packetization, and the
producer/sender plumbing driven by ScanRunner.

Deliberately backend-agnostic: this module never imports
direct_synthesis.py. The streamer class exposes a small uniform surface
(channel_id_map, num_channels, tick_n_samples(), generate_next_tick())
that ScanRunner relies on instead of importing/isinstance-checking a
concrete class — see ScanRunner below. Kept this way (rather than
importing DirectSynthesisStreamer directly, now that it's the only
implementation) so common.py stays testable and reusable independent of
which generation strategy is behind it.

===========================================================================
SPEAD ENCODING IS HAND-ROLLED, NOT VIA spead2 — see SpsPacketizer below.
===========================================================================
spead2's own packet encoder (its C++ send_packet.cpp::packet_generator::
next_packet, checked directly against spead2==4.4.1's actual source, not
just its Python API) unconditionally writes 4 reserved item pointers
(HEAP_CNT, HEAP_LENGTH, PAYLOAD_OFFSET, PAYLOAD_LENGTH) at the start of
EVERY packet it emits, with no flag, StreamConfig option, or Heap method
to suppress any of them — it's hardcoded, not a missing config knob.
CBF's real ICD heap has exactly 6 items total (see SpsPacketizer's ITEM
LAYOUT), fewer than spead2's own mandatory minimum, so spead2 cannot
produce a compliant packet no matter how it's configured. This was
discovered the hard way: a real heap sent through spead2 came out with 8
items instead of 6, which a firmware receiver that addresses payload
bytes directly (rather than doing a generic SPEAD parse) would misread.
SpsPacketizer below replaces spead2 entirely for the send path with a
from-scratch SPEAD-64-48 item-pointer encoder.

`pack_channel_info`/`pack_antenna_info`'s bit-field boundaries were
originally read off a screenshot of the ICD diagram and flagged
unverified; the exact widths below are now CONFIRMED against the real
ICD (not the screenshot) and happen to match what was already there bit
for bit.

ALSO UNVERIFIED:
    - HeapAccumulator: buffers per-channel samples until 2048/channel are
      available, then emits one heap per channel.
    - TAI2000 heap_counter conversion (astropy-based, with a heavily
      caveated non-astropy fallback).
"""

from __future__ import annotations

import logging
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cbf_sim")


# ============================================================
# CONFIG — genuinely ICD-fixed constants shared by every source type
# direct_synthesis.py generates.
# ============================================================

CHANNEL_WIDTH_HZ = 781_250.0  # per SPS-CBF ICD coarse channel spacing — CONFIRM
NUM_CHANNELS = 96  # 96 * 781.25kHz ≈ 75 MHz, per your requirement

# CONFIRMED: the lowest valid frequency for the SKA-Low telescope is
# 50.78125 MHz, which is coarse channel ID 65 in the ICD's GLOBAL channel
# numbering (65 * CHANNEL_WIDTH_HZ = 50,781,250 Hz exactly — no rounding).
# This is the frequency of LOCAL channel 0 for a station simulating the
# band's bottom edge; StationConfig.first_channel_id (65 by default, see
# below) is what maps that local channel 0 onto the correct GLOBAL
# channel ID 65 in the wire-packed channel_info. No longer a placeholder
# -- was 0.0 pending this confirmation; that made any pulsed source
# configured through simulator.py's StartScan (which never overrode
# base_freq_hz) fail outright, since DirectSynthesisStreamer requires
# base_freq_hz > 0 for pulsed sources (dispersion diverges as f -> 0).
BASE_FREQ_HZ = 50.78125e6

HEAP_LEN = 2048  # time samples per heap per channel, per ICD

# Per-tick time budget, FIXED regardless of channel count — this is the
# real-time constraint generation is racing against. Equal to
# HEAP_LEN / channel_output_rate for a critically sampled channelizer —
# a property of the ICD's channel width, not of any particular
# generation strategy.
BLOCK_DURATION_S = HEAP_LEN / CHANNEL_WIDTH_HZ

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
        """IMPORTANT: evaluated relative to start_validity_sec, NOT raw
        absolute epoch time — a 5th-order poly blows up otherwise (this
        was a real bug, caught by a smoke test, present in earlier
        iterations of this design too). t_rel should stay within roughly
        [0, validity_period_sec]."""
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
    simulator.py), though nothing here is Tango-specific: update() just
    needs calling from whatever thread learns of a new polynomial (tests
    call it directly). get() is called from the generation thread. A
    plain reference swap is safe across threads under the GIL without an
    explicit lock — no field of the swapped-in DelayPolynomial is ever
    mutated in place, only the `_poly` reference itself is replaced.

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
        self._poly: Optional[DelayPolynomial] = None
        self._warned_no_poly = False
        self._warned_stale_valid_until: Optional[float] = None

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
    ICD bit-packing functions below. ska-low-csp-delaymodel/1.0 is a
    documented schema (ADR-88 in ska-telmodel) but the exact payload a
    real delay-poly Tango attribute pushes hasn't been checked against it
    here. Assumes `value` is a JSON string (or an already-parsed mapping)
    with keys matching DelayPolynomial's fields. Confirm against the real
    schema and the real CBF delay-poly emulator before deploying."""
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
    # 65 = the confirmed GLOBAL coarse channel ID of BASE_FREQ_HZ
    # (50.78125 MHz, the lowest valid SKA-Low frequency) — the right
    # default for a station simulating the band's bottom edge, whose
    # local channel 0 corresponds to that global channel. Override for a
    # station covering a different sub-band.
    first_channel_id: int = 65
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
    here: a (n_new, num_channels) complex array per pol) until HEAP_LEN
    are available per channel, then emits one ChannelHeap per channel."""

    def __init__(
        self,
        num_channels: int,
        obs_time: float,
        sample_rate_per_channel: float,
        channel_id_map: Optional[np.ndarray] = None,
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
# PACKING — channel_info / antenna_info bit layout
# CONFIRMED against the real ICD (see SpsPacketizer's ITEM LAYOUT below)
# ============================================================


def pack_channel_info(beam_id: int, frequency_id: int) -> int:
    """0x3000 item value: 16 bits reserved | 16 bits beam_id | 16 bits
    frequency_id, within the low 48 bits of the SPEAD item pointer."""
    return ((beam_id & 0xFFFF) << 16) | (frequency_id & 0xFFFF)


def pack_antenna_info(substation_id: int, subarray_id: int, station_id: int) -> int:
    """0x3001 item value: 8 bits substation_id | 8 bits subarray_id |
    16 bits station_id | 16 bits reserved, within the low 48 bits of the
    SPEAD item pointer."""
    return (
        ((substation_id & 0xFF) << 40)
        | ((subarray_id & 0xFF) << 32)
        | ((station_id & 0xFFFF) << 16)
    )


def build_heap_payload_bytes(
    v_i8: np.ndarray, v_q8: np.ndarray, h_i8: np.ndarray, h_q8: np.ndarray
) -> bytes:
    """Interleave as Vreal, Vimag, Hreal, Himag per sample, per diagram."""
    interleaved = np.empty((HEAP_LEN, 4), dtype=np.int8)
    interleaved[:, 0] = v_i8
    interleaved[:, 1] = v_q8
    interleaved[:, 2] = h_i8
    interleaved[:, 3] = h_q8
    payload = interleaved.tobytes()
    assert len(payload) == 0x2000, f"payload size {len(payload)} != expected 8192 bytes"
    return payload


def quantize_8bit(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-call independent scaling. Returns (real_i8, imag_i8)."""
    scale = 127.0 / (np.max(np.abs(samples)) + 1e-12)
    scaled = samples * scale
    real = np.clip(np.round(scaled.real), -128, 127).astype(np.int8)
    imag = np.clip(np.round(scaled.imag), -128, 127).astype(np.int8)
    return real, imag


# ============================================================
# SPEAD ENCODING — hand-rolled SPEAD-64-48, NOT spead2
#
# spead2 cannot produce CBF's real heap format: its packet encoder always
# writes 4 mandatory reserved item pointers spead2 itself picks, and
# CBF's ICD heap has only 6 items total, fewer than that mandatory
# minimum — see this module's docstring for the full finding (checked
# directly against spead2==4.4.1's C++ source, not assumed). The encoder
# below is a minimal from-scratch replacement, scoped to exactly the 6
# items CBF's firmware expects and nothing else.
# ============================================================

SPEAD_VERSION = 4
SPEAD_ITEM_POINTER_BITS = 64
SPEAD_HEAP_ADDRESS_BITS = 48  # SPEAD-64-48, confirmed against the real ICD
_SPEAD_ID_BITS = SPEAD_ITEM_POINTER_BITS - 1 - SPEAD_HEAP_ADDRESS_BITS  # 15
_SPEAD_ID_MASK = (1 << _SPEAD_ID_BITS) - 1
_SPEAD_VALUE_MASK = (1 << SPEAD_HEAP_ADDRESS_BITS) - 1
_SPEAD_HEAP_COUNTER_BITS = 40  # top 8 bits of the 0x0001 item are reserved
_SPEAD_HEAP_COUNTER_MASK = (1 << _SPEAD_HEAP_COUNTER_BITS) - 1

PAYLOAD_LENGTH_BYTES = 0x2000  # HEAP_LEN * 4 bytes/sample, fixed by the ICD


def _spead_header_bytes(n_items: int) -> bytes:
    """The 8-byte SPEAD packet header: magic (0x53) + version (0x04),
    then the two field-width bytes spead2's own encoder also writes (item
    ID field width in bytes, heap-address field width in bytes — fixed
    here at SPEAD-64-48's 2 and 6), then the item-pointer count. Layout
    matches the real SPEAD wire format (checked directly against
    spead2==4.4.1's send_packet.cpp, not reverse-engineered from output);
    what differs from spead2 is only WHICH items follow this header."""
    heap_address_bytes = SPEAD_HEAP_ADDRESS_BITS // 8
    id_bytes = (SPEAD_ITEM_POINTER_BITS // 8) - heap_address_bytes
    word = (
        (0x5300 | SPEAD_VERSION) << 48
        | (id_bytes << 40)
        | (heap_address_bytes << 32)
        | n_items
    )
    return struct.pack(">Q", word)


def _spead_item_pointer(item_id: int, value: int) -> bytes:
    """One IMMEDIATE SPEAD-64-48 item pointer (mode bit set, value
    embedded directly in the pointer's low 48 bits). CBF's 6-item heap
    never needs an ADDRESS-mode pointer: even the fixed 0x2000 payload
    length and the always-zero payload offset are immediate values here,
    not pointers into the payload. The actual sample payload follows the
    item pointers as raw bytes with no item pointer of its own — CBF
    firmware reads it at a fixed byte offset (56 bytes in: 8-byte header
    + 6*8-byte item pointers), not through generic SPEAD item addressing."""
    if not (0 <= item_id <= _SPEAD_ID_MASK):
        raise ValueError(f"item id {item_id:#x} does not fit in {_SPEAD_ID_BITS} bits")
    if not (0 <= value <= _SPEAD_VALUE_MASK):
        raise ValueError(
            f"value {value:#x} for item {item_id:#x} does not fit in "
            f"{SPEAD_HEAP_ADDRESS_BITS} bits"
        )
    pointer = (1 << 63) | (item_id << SPEAD_HEAP_ADDRESS_BITS) | value
    return struct.pack(">Q", pointer)


class SpsPacketizer:
    """Encodes and sends one ChannelHeap per SPS-CBF ICD, via a hand-rolled
    SPEAD-64-48 encoder — NOT spead2 (see this module's docstring for why
    spead2 cannot produce this format at all).

    ITEM LAYOUT (confirmed against the real ICD, six items total):
        0x0001  8 bits reserved | 40 bits heap_counter
        0x0004  48 bits packet_payload_length (fixed: 0x2000)
        0x3010  48 bits scan_id
        0x3000  16 bits reserved | 16 bits beam_id | 16 bits frequency_id
        0x3001  8 bits substation_id | 8 bits subarray_id |
                16 bits station_id | 16 bits reserved
        0x3300  48 bits payload_offset (fixed: 0x0 — heaps are always
                exactly one packet, never fragmented, so this is never
                anything else)
    ...followed immediately by the 8192-byte interleaved V/H I/Q payload.
    There is no 7th "payload" item pointer — CBF firmware addresses the
    payload bytes directly at a fixed offset after the 6 item pointers,
    rather than doing a generic SPEAD parse (this is also exactly why
    spead2's own extra reserved items are a real problem, not cosmetic:
    a fixed-offset reader has no way to skip items it doesn't expect).
    """

    def __init__(
        self,
        station: StationConfig,
        dest_ip: Optional[str] = None,
        dest_port: Optional[int] = None,
        sock=None,
    ):
        """`sock`, if given (anything with a `.sendto(bytes, addr)`
        method), is used directly instead of constructing a live UDP
        socket from dest_ip/dest_port — lets a caller capture raw SPEAD
        bytes without actually sending them (see generate_test_pcap.py),
        or a test inject a fake socket to inspect what would be sent."""
        self.station = station
        self.dest_addr = (dest_ip, dest_port) if dest_ip is not None else None
        if sock is not None:
            self._sock = sock
        elif dest_ip is not None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self._sock = None

    def encode_channel_heap(self, heap: ChannelHeap) -> bytes:
        """Builds the raw SPEAD-64-48 heap bytes for one channel — the
        full on-wire payload of one UDP packet, per this class's ITEM
        LAYOUT. Split out from send_channel_heap so callers that only
        need the encoded bytes (generate_test_pcap.py, tests) don't need
        a real or fake socket at all."""
        v_i8, v_q8 = quantize_8bit(heap.v_samples)
        h_i8, h_q8 = quantize_8bit(heap.h_samples)
        payload = build_heap_payload_bytes(v_i8, v_q8, h_i8, h_q8)

        # "packet count since SKA epoch" -- count of HEAP_LEN-sample
        # BLOCKS (BLOCK_DURATION_S each) since epoch, NOT a count of
        # individual samples. A real, previously-undetected bug lived
        # here: multiplying by CHANNEL_WIDTH_HZ (the SAMPLE rate) instead
        # of dividing by BLOCK_DURATION_S (the correct HEAP rate) inflated
        # this by a factor of HEAP_LEN (2048x) -- for any current-era
        # timestamp that would overflow the ICD's 40-bit heap_counter
        # field. Caught only by actually exercising this path end-to-end
        # for the first time (generate_test_pcap.py) -- this had never
        # been exercised before. Fixed formula keeps heap_counter
        # comfortably within 40 bits until roughly year 2091.
        heap_counter = int(
            round(unix_to_tai2000_seconds(heap.heap_start_time) / BLOCK_DURATION_S)
        )
        if not (0 <= heap_counter <= _SPEAD_HEAP_COUNTER_MASK):
            raise ValueError(
                f"heap_counter {heap_counter} does not fit in the ICD's "
                f"{_SPEAD_HEAP_COUNTER_BITS}-bit field (top 8 bits of the "
                f"0x0001 item are reserved)"
            )

        items = (
            (0x0001, heap_counter),
            (0x0004, PAYLOAD_LENGTH_BYTES),
            (0x3010, self.station.scan_id),
            (
                0x3000,
                pack_channel_info(
                    self.station.beam_id,
                    self.station.first_channel_id + heap.channel_id,
                ),
            ),
            (
                0x3001,
                pack_antenna_info(
                    self.station.substation_id,
                    self.station.subarray_id,
                    self.station.station_id,
                ),
            ),
            (0x3300, 0x0),
        )
        pointers = b"".join(_spead_item_pointer(item_id, value) for item_id, value in items)
        return _spead_header_bytes(len(items)) + pointers + payload

    def send_channel_heap(self, heap: ChannelHeap):
        if self._sock is None or self.dest_addr is None:
            raise RuntimeError(
                "SpsPacketizer has no destination -- pass dest_ip/dest_port "
                "(a sock alone, with no dest_ip, only supports "
                "encode_channel_heap, not send_channel_heap)"
            )
        self._sock.sendto(self.encode_channel_heap(heap), self.dest_addr)


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
        """How many per-channel output samples generate_next_tick's
        second argument should be for one tick — sized so one tick
        produces close to exactly one heap's worth of per-channel
        samples (BLOCK_DURATION_S is defined for exactly this)."""
        ...

    # Positional-only (`/`) so implementations can use their own, more
    # descriptive parameter names without tripping Protocol structural
    # matching on name. Mapping (not dict) for the return type since it's
    # covariant in the value type, allowing a future implementation to
    # return None for a channel with no data yet without breaking this
    # protocol.
    def generate_next_tick(
        self, t: float, n: int, /
    ) -> Mapping[str, Optional[np.ndarray]]: ...


class ScanRunner:
    def __init__(
        self,
        streamer: Streamer,
        send_queue: "queue.Queue[ChannelHeap]",
        obs_time: float,
        scan_duration_s: float,
    ):
        self.streamer = streamer
        self.send_queue = send_queue
        self.obs_time = obs_time
        self.scan_duration_s = scan_duration_s
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        # Fixed by the ICD (critically sampled channel spacing), not by
        # whichever backend is in use.
        self.channel_output_rate = CHANNEL_WIDTH_HZ
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
    send_queue: "queue.Queue[ChannelHeap]",
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
