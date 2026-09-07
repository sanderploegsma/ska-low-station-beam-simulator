"""SPEAD-64-48 heap encoding for the SPS-CBF ICD wire format —
hand-rolled, NOT via spead2.

===========================================================================
SPEAD ENCODING IS HAND-ROLLED, NOT VIA spead2.
===========================================================================
spead2's own packet encoder (its C++ ``send_packet.cpp::packet_generator::
next_packet``, checked directly against spead2==4.4.1's actual source,
not just its Python API) unconditionally writes 4 reserved item pointers
(HEAP_CNT, HEAP_LENGTH, PAYLOAD_OFFSET, PAYLOAD_LENGTH) at the start of
EVERY packet it emits, with no flag, ``StreamConfig`` option, or ``Heap``
method to suppress any of them — it's hardcoded, not a missing config
knob. CBF's real ICD heap has exactly 6 items total (see
``SpsPacketizer``'s ITEM LAYOUT), fewer than spead2's own mandatory
minimum, so spead2 cannot produce a compliant packet no matter how it's
configured — a firmware receiver that addresses payload bytes directly
(rather than doing a generic SPEAD parse) would misread the extra items.
``SpsPacketizer`` below replaces spead2 entirely for the send path with
a from-scratch SPEAD-64-48 item-pointer encoder.

``pack_channel_info``/``pack_antenna_info``'s bit-field boundaries are
CONFIRMED against the real ICD.
"""

from __future__ import annotations

import socket
import struct

import numpy as np

from ska_low_station_beam_simulator.common import (
    BLOCK_DURATION_S,
    CHANNEL_START,
    HEAP_LEN,
    ChannelHeap,
    StationConfig,
    unix_to_tai2000_seconds,
)

# ============================================================
# PACKING — channel_info / antenna_info bit layout
# CONFIRMED against the real ICD (see SpsPacketizer's ITEM LAYOUT below)
# ============================================================


def pack_channel_info(beam_id: int, frequency_id: int) -> int:
    """0x3000 item value: 16 bits reserved | 16 bits beam_id | 16 bits
    frequency_id, within the low 48 bits of the SPEAD item pointer.

    :param beam_id: the beam ID.
    :param frequency_id: the GLOBAL coarse channel ID.
    :returns: the packed 0x3000 item value.
    """
    return ((beam_id & 0xFFFF) << 16) | (frequency_id & 0xFFFF)


def pack_antenna_info(substation_id: int, subarray_id: int, station_id: int) -> int:
    """0x3001 item value: 8 bits substation_id | 8 bits subarray_id |
    16 bits station_id | 16 bits reserved, within the low 48 bits of the
    SPEAD item pointer.

    :param substation_id: the substation ID.
    :param subarray_id: the subarray ID.
    :param station_id: the station ID.
    :returns: the packed 0x3001 item value.
    """
    return (
        ((substation_id & 0xFF) << 40)
        | ((subarray_id & 0xFF) << 32)
        | ((station_id & 0xFFFF) << 16)
    )


def build_heap_payload_bytes(
    v_i8: np.ndarray, v_q8: np.ndarray, h_i8: np.ndarray, h_q8: np.ndarray
) -> bytes:
    """Interleave as Vreal, Vimag, Hreal, Himag per sample, per diagram.

    :param v_i8: V polarisation real component, int8, shape (HEAP_LEN,).
    :param v_q8: V polarisation imaginary component, int8, shape (HEAP_LEN,).
    :param h_i8: H polarisation real component, int8, shape (HEAP_LEN,).
    :param h_q8: H polarisation imaginary component, int8, shape (HEAP_LEN,).
    :returns: the interleaved 8192-byte heap payload.
    """
    interleaved = np.empty((HEAP_LEN, 4), dtype=np.int8)
    interleaved[:, 0] = v_i8
    interleaved[:, 1] = v_q8
    interleaved[:, 2] = h_i8
    interleaved[:, 3] = h_q8
    payload = interleaved.tobytes()
    assert len(payload) == 0x2000, f"payload size {len(payload)} != expected 8192 bytes"
    return payload


def quantize_8bit(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-call independent scaling.

    :param samples: complex samples to quantize.
    :returns: a ``(real_i8, imag_i8)`` tuple of int8 arrays.
    """
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
    what differs from spead2 is only WHICH items follow this header.

    :param n_items: number of item pointers following this header.
    :returns: the 8-byte header.
    """
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
    + 6*8-byte item pointers), not through generic SPEAD item addressing.

    :param item_id: the SPEAD item ID (must fit in 15 bits).
    :param value: the immediate value (must fit in 48 bits).
    :returns: the 8-byte item pointer.
    :raises ValueError: if ``item_id`` or ``value`` doesn't fit in its
        allotted field width.
    """
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
    """Encodes and sends one ``ChannelHeap`` per SPS-CBF ICD, via a
    hand-rolled SPEAD-64-48 encoder — NOT spead2 (see this module's
    docstring for why spead2 cannot produce this format at all).

    ITEM LAYOUT (confirmed against the real ICD, six items total)::

        0x0001  8 bits reserved | 40 bits heap_counter
        0x0004  48 bits packet_payload_length (fixed: 0x2000)
        0x3010  48 bits scan_id
        0x3000  16 bits reserved | 16 bits beam_id | 16 bits frequency_id
        0x3001  8 bits substation_id | 8 bits subarray_id |
                16 bits station_id | 16 bits reserved
        0x3300  48 bits payload_offset (fixed: 0x0 -- heaps are always
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
        dest_ip: str | None = None,
        dest_port: int | None = None,
        sock=None,
    ):
        """
        :param station: identifies this packetizer's station in every
            encoded heap.
        :param dest_ip: destination IPv4 address for ``send_channel_heap``
            — required for sending, not for ``encode_channel_heap`` alone.
        :param dest_port: destination UDP port, paired with ``dest_ip``.
        :param sock: anything with a ``.sendto(bytes, addr)`` method,
            used directly instead of constructing a live UDP socket from
            ``dest_ip``/``dest_port`` — lets a caller capture raw SPEAD
            bytes without actually sending them (see
            ``generate_test_pcap.py``), or a test inject a fake socket to
            inspect what would be sent.
        """
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
        LAYOUT. Split out from ``send_channel_heap`` so callers that only
        need the encoded bytes (``generate_test_pcap.py``, tests) don't
        need a real or fake socket at all.

        :param heap: the channel heap to encode.
        :returns: the complete SPEAD-64-48 heap bytes (header + 6 item
            pointers + 8192-byte payload).
        :raises ValueError: if ``heap``'s ``heap_start_time`` yields a
            ``heap_counter`` that doesn't fit in the ICD's 40-bit field.
        """
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
        heap_counter = round(
            unix_to_tai2000_seconds(heap.heap_start_time) / BLOCK_DURATION_S
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
                    CHANNEL_START + heap.channel_id,
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
        pointers = b"".join(
            _spead_item_pointer(item_id, value) for item_id, value in items
        )
        return _spead_header_bytes(len(items)) + pointers + payload

    def send_channel_heap(self, heap: ChannelHeap):
        if self._sock is None or self.dest_addr is None:
            raise RuntimeError(
                "SpsPacketizer has no destination -- pass dest_ip/dest_port "
                "(a sock alone, with no dest_ip, only supports "
                "encode_channel_heap, not send_channel_heap)"
            )
        self._sock.sendto(self.encode_channel_heap(heap), self.dest_addr)
