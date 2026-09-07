"""Tests for common.SpsPacketizer's hand-rolled SPEAD-64-48 encoding, and
for generate_test_pcap.py's pcap output.

SpsPacketizer does NOT use spead2 -- spead2's own packet encoder always
writes 4 reserved item pointers (HEAP_CNT, HEAP_LENGTH, PAYLOAD_OFFSET,
PAYLOAD_LENGTH) with no way to suppress any of them (confirmed directly
against spead2==4.4.1's C++ source), but CBF's real ICD heap has exactly
6 items total -- fewer than that mandatory minimum. So this module
hand-encodes the SPEAD-64-48 wire format itself, scoped to exactly the
6 items the ICD specifies. See common.py's module docstring and
SpsPacketizer's docstring for the full item layout.

Regression coverage for a real bug from an earlier (spead2-based)
version of this code: heap_counter used to be computed as
tai2000_seconds * CHANNEL_WIDTH_HZ (the SAMPLE rate) instead of
tai2000_seconds / BLOCK_DURATION_S (the correct per-HEAP rate),
inflating it by HEAP_LEN (2048x) -- enough to overflow the ICD's 40-bit
heap_counter field for any current-era timestamp. This was never caught
by this project's existing tests because nothing exercised
send_channel_heap()/encode_channel_heap() end-to-end before
generate_test_pcap.py did.
"""

import struct
import time

import numpy as np

from ska_low_station_beam_simulator import generate_test_pcap as gen_pcap
from ska_low_station_beam_simulator.common import (
    BLOCK_DURATION_S,
    CHANNEL_START,
    HEAP_LEN,
    PAYLOAD_LENGTH_BYTES,
    SPEAD_HEAP_ADDRESS_BITS,
    ChannelHeap,
    SpsPacketizer,
    StationConfig,
    build_heap_payload_bytes,
    pack_antenna_info,
    pack_channel_info,
    quantize_8bit,
    unix_to_tai2000_seconds,
)

SPEAD_HEAP_COUNTER_MAX = 2**40 - 1


def _make_heap(
    channel_id: int = 0, heap_start_time: float | None = None
) -> ChannelHeap:
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(HEAP_LEN) + 1j * rng.standard_normal(HEAP_LEN)) * 0.1
    return ChannelHeap(
        channel_id=channel_id,
        v_samples=samples,
        h_samples=samples,
        heap_start_time=heap_start_time if heap_start_time is not None else time.time(),
    )


def _station() -> StationConfig:
    return StationConfig(
        station_id=1, substation_id=2, subarray_id=3, beam_id=4, scan_id=99
    )


def _parse_item_pointers(raw: bytes, n_items: int) -> list[tuple[int, int]]:
    """Parses ``raw``'s SPEAD-64-48 header + item pointers back into
    ``(item_id, value)`` pairs, independent of ``SpsPacketizer``'s own
    encoding logic -- so a bug in the encoder can't also hide from its
    own test.

    :param raw: the encoded heap bytes.
    :param n_items: expected number of item pointers.
    :returns: a list of ``(item_id, value)`` pairs.
    """
    (header_word,) = struct.unpack(">Q", raw[:8])
    assert (header_word >> 48) == 0x5304  # magic 0x53, version 4
    assert (header_word >> 40) & 0xFF == 2  # item-ID field width in bytes
    assert (header_word >> 32) & 0xFF == 6  # heap-address field width in bytes
    assert (header_word & 0xFFFFFFFF) == n_items

    id_mask = (1 << 15) - 1
    value_mask = (1 << SPEAD_HEAP_ADDRESS_BITS) - 1
    pointers = []
    for i in range(n_items):
        (word,) = struct.unpack(">Q", raw[8 + i * 8 : 16 + i * 8])
        assert word >> 63 == 1, "expected every ICD item to be immediate-mode"
        item_id = (word >> SPEAD_HEAP_ADDRESS_BITS) & id_mask
        value = word & value_mask
        pointers.append((item_id, value))
    return pointers


def test_encode_channel_heap_item_pointers_match_icd_spec():
    """Checks that every one of the ICD's 6 heap items is encoded with
    the exact item ID and value the spec requires, decoded independently
    of SpsPacketizer's own encoding logic (see _parse_item_pointers) so a
    bug in the encoder can't also hide from its own test -- CLAUDE.md
    flags this bit-packing as the single highest-priority correctness gap
    in the whole codebase."""
    station = _station()
    packetizer = SpsPacketizer(station)
    heap = _make_heap(channel_id=7, heap_start_time=time.time())

    raw = packetizer.encode_channel_heap(heap)
    pointers = dict(_parse_item_pointers(raw, n_items=6))

    expected_counter = round(
        unix_to_tai2000_seconds(heap.heap_start_time) / BLOCK_DURATION_S
    )
    assert pointers[0x0001] == expected_counter
    assert pointers[0x0004] == PAYLOAD_LENGTH_BYTES
    assert pointers[0x3010] == station.scan_id
    assert pointers[0x3000] == pack_channel_info(
        station.beam_id, CHANNEL_START + heap.channel_id
    )
    assert pointers[0x3001] == pack_antenna_info(
        station.substation_id, station.subarray_id, station.station_id
    )
    assert pointers[0x3300] == 0x0


def test_encode_channel_heap_total_length_and_payload():
    """Confirms the encoded heap has exactly the ICD's expected byte
    length (header + 6 item pointers + payload, no more) and that the
    payload itself matches the quantized V/H samples -- a length or
    content mismatch here would produce a heap CBF can't parse."""
    station = _station()
    packetizer = SpsPacketizer(station)
    heap = _make_heap()

    raw = packetizer.encode_channel_heap(heap)

    # 8-byte header + 6*8-byte item pointers + 8192-byte payload, no more.
    assert len(raw) == 8 + 6 * 8 + PAYLOAD_LENGTH_BYTES

    v_i8, v_q8 = quantize_8bit(heap.v_samples)
    h_i8, h_q8 = quantize_8bit(heap.h_samples)
    expected_payload = build_heap_payload_bytes(v_i8, v_q8, h_i8, h_q8)
    assert raw[8 + 6 * 8 :] == expected_payload


def test_heap_counter_fits_within_icd_field_for_current_time():
    """Regression guard for bug #17: the old heap_counter formula
    inflated the value by HEAP_LEN (2048x) and would have overflowed the
    ICD's 40-bit field for any present-day timestamp -- pins the fixed
    formula to actually fit for 'now'."""
    heap_counter = round(unix_to_tai2000_seconds(time.time()) / BLOCK_DURATION_S)
    assert 0 <= heap_counter <= SPEAD_HEAP_COUNTER_MAX


def test_heap_counter_still_fits_decades_from_now():
    """Headroom check: the fixed formula should comfortably outlive this
    codebase, not just barely fit today."""
    thirty_years_from_now = time.time() + 30 * 365.25 * 86400
    heap_counter = round(
        unix_to_tai2000_seconds(thirty_years_from_now) / BLOCK_DURATION_S
    )
    assert 0 <= heap_counter <= SPEAD_HEAP_COUNTER_MAX


def test_encode_channel_heap_rejects_out_of_range_heap_counter():
    """The ICD reserves the top 8 bits of the 0x0001 item -- a heap_counter
    that doesn't fit in the remaining 40 bits must be rejected outright,
    not silently truncated (silent truncation would misencode a real
    timestamp instead of failing loudly)."""
    station = _station()
    packetizer = SpsPacketizer(station)
    # heap_start_time far enough in the future that tai2000/BLOCK_DURATION_S
    # exceeds 2**40 - 1 (~year 2091, per common.py) without going so far
    # into the future that astropy's Time itself starts to complain.
    heap = _make_heap(heap_start_time=time.time() + 100 * 365.25 * 86400)
    try:
        packetizer.encode_channel_heap(heap)
        assert False, "expected ValueError for an out-of-range heap_counter"
    except ValueError:
        pass


def test_send_channel_heap_sends_encoded_bytes_to_dest_addr():
    """Exercises SpsPacketizer.send_channel_heap end-to-end against an
    injected fake socket, without needing a real network or spead2."""

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def sendto(self, data, addr):
            self.sent.append((data, addr))

    station = _station()
    fake_sock = FakeSocket()
    packetizer = SpsPacketizer(
        station, dest_ip="10.0.0.5", dest_port=8001, sock=fake_sock
    )
    heap = _make_heap()

    expected = packetizer.encode_channel_heap(heap)
    packetizer.send_channel_heap(heap)

    assert len(fake_sock.sent) == 1
    data, addr = fake_sock.sent[0]
    assert data == expected
    assert addr == ("10.0.0.5", 8001)


def test_send_channel_heap_without_destination_raises():
    """A packetizer with no destination configured has nowhere to
    actually send a heap -- must raise clearly rather than silently
    dropping it or failing with an unrelated error."""
    station = _station()
    packetizer = SpsPacketizer(station)  # no dest_ip, no sock
    try:
        packetizer.send_channel_heap(_make_heap())
        assert False, "expected RuntimeError with no destination configured"
    except RuntimeError:
        pass


# ============================================================
# generate_test_pcap.py
# ============================================================


def test_generate_test_pcap_writes_well_formed_pcap(tmp_path):
    """Checks the pcap file's own header and per-record framing (magic
    number, link type, record count, no truncated/trailing bytes) are
    well-formed, independent of what's inside each record --
    generate_test_pcap.py exists specifically because spead2 has no pcap
    writer of its own (see module docstring), so this format has to be
    hand-verified rather than trusted from a library."""
    output = tmp_path / "test.pcap"
    gen_pcap.generate_test_pcap(str(output), n_heaps=3)

    data = output.read_bytes()
    magic, _ver_major, _ver_minor, _thiszone, _sigfigs, _snaplen, network = (
        struct.unpack("<IHHiIII", data[:24])
    )
    assert magic == gen_pcap.PCAP_MAGIC_MICROSECONDS
    assert network == gen_pcap.LINKTYPE_ETHERNET

    offset = 24
    n_records = 0
    while offset < len(data):
        _ts_sec, _ts_usec, incl_len, orig_len = struct.unpack(
            "<IIII", data[offset : offset + 16]
        )
        assert incl_len == orig_len
        offset += 16 + incl_len
        n_records += 1
    assert n_records == 3
    assert offset == len(data)  # no trailing garbage, no truncated final record


def test_generate_test_pcap_frames_are_valid_udp_over_ipv4():
    """Checks the actual Ethernet/IPv4/UDP framing this module hand-rolls
    (spead2 has no pcap writer -- see module docstring), including that
    the IPv4 header checksum is correct, not just present."""
    payload = b"x" * 20
    frame = gen_pcap._wrap_udp_frame(payload, "10.0.0.1", "10.0.0.2", 8000, 8000)

    eth_type = struct.unpack("!H", frame[12:14])[0]
    assert eth_type == 0x0800  # IPv4

    ip_header = frame[14:34]
    assert (
        gen_pcap._ipv4_checksum(ip_header) == 0
    )  # a correct checksum sums to 0 over the whole header
    version_ihl = ip_header[0]
    assert version_ihl == 0x45  # IPv4, 20-byte header (no options)

    udp_header = frame[34:42]
    src_port, dst_port, udp_len, _ = struct.unpack("!HHHH", udp_header)
    assert (src_port, dst_port) == (8000, 8000)
    assert udp_len == 8 + len(payload)
    assert frame[42:] == payload
