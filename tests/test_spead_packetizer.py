"""Tests for common.SpsPacketizer's SPEAD encoding, and for
generate_test_pcap.py's pcap output.

Regression coverage for a real bug this session: heap_counter used to be
computed as tai2000_seconds * CHANNEL_WIDTH_HZ (the SAMPLE rate) instead
of tai2000_seconds / BLOCK_DURATION_S (the correct per-HEAP rate),
inflating it by HEAP_LEN (2048x) -- enough to overflow spead2's actual
40-bit cnt limit for any current-era timestamp and make every real
send_channel_heap() call fail outright with OSError. This was never
caught by this project's existing tests because spead2 usage lived only
in SpsPacketizer/the actual device server, neither of which was
exercised end-to-end before generate_test_pcap.py.
"""

import struct
import time

import numpy as np
import spead2
import spead2.send

from ska_low_station_beam_simulator import generate_test_pcap as gen_pcap
from ska_low_station_beam_simulator.common import (
    BLOCK_DURATION_S,
    ChannelHeap,
    HEAP_LEN,
    SpsPacketizer,
    StationConfig,
    unix_to_tai2000_seconds,
)

SPEAD2_MAX_CNT = 2**40 - 1


def _make_heap(channel_id: int = 0, heap_start_time: float | None = None) -> ChannelHeap:
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(HEAP_LEN) + 1j * rng.standard_normal(HEAP_LEN)) * 0.1
    return ChannelHeap(
        channel_id=channel_id,
        v_samples=samples,
        h_samples=samples,
        heap_start_time=heap_start_time if heap_start_time is not None else time.time(),
    )


def _station() -> StationConfig:
    return StationConfig(station_id=1, substation_id=0, subarray_id=1, beam_id=1, scan_id=1)


def test_heap_counter_fits_within_spead2_cnt_limit_for_current_time():
    """The actual regression check: a realistic, present-day timestamp
    must not overflow spead2's 40-bit cnt field (confirmed against the
    installed spead2==4.4.1 by direct probing, not assumed)."""
    heap_counter = int(round(unix_to_tai2000_seconds(time.time()) / BLOCK_DURATION_S))
    assert 0 <= heap_counter <= SPEAD2_MAX_CNT


def test_heap_counter_still_fits_decades_from_now():
    """Headroom check: the fixed formula should comfortably outlive this
    codebase, not just barely fit today."""
    thirty_years_from_now = time.time() + 30 * 365.25 * 86400
    heap_counter = int(round(unix_to_tai2000_seconds(thirty_years_from_now) / BLOCK_DURATION_S))
    assert 0 <= heap_counter <= SPEAD2_MAX_CNT


def test_send_channel_heap_succeeds_via_bytes_stream():
    """Actually exercises SpsPacketizer.send_channel_heap end-to-end
    (not just the heap_counter arithmetic in isolation) -- this call
    raised OSError unconditionally before the heap_counter fix, for any
    current-era heap_start_time."""
    stream = spead2.send.BytesStream(spead2.ThreadPool())
    packetizer = SpsPacketizer(_station(), stream=stream)
    packetizer.send_channel_heap(_make_heap())
    raw = stream.getvalue()
    assert len(raw) > 8192  # at least the raw payload, plus SPEAD framing overhead


def test_sps_packetizer_accepts_injected_stream_without_dest_ip_port():
    """SpsPacketizer(stream=...) shouldn't need dest_ip/dest_port at all
    -- this is what lets generate_test_pcap.py capture raw bytes instead
    of needing a live UdpStream."""
    stream = spead2.send.BytesStream(spead2.ThreadPool())
    packetizer = SpsPacketizer(_station(), stream=stream)
    assert packetizer.stream is stream


# ============================================================
# generate_test_pcap.py
# ============================================================


def test_generate_test_pcap_writes_well_formed_pcap(tmp_path):
    output = tmp_path / "test.pcap"
    gen_pcap.generate_test_pcap(str(output), n_heaps=3)

    data = output.read_bytes()
    magic, ver_major, ver_minor, thiszone, sigfigs, snaplen, network = struct.unpack(
        "<IHHiIII", data[:24]
    )
    assert magic == gen_pcap.PCAP_MAGIC_MICROSECONDS
    assert network == gen_pcap.LINKTYPE_ETHERNET

    offset = 24
    n_records = 0
    while offset < len(data):
        ts_sec, ts_usec, incl_len, orig_len = struct.unpack("<IIII", data[offset : offset + 16])
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
    assert gen_pcap._ipv4_checksum(ip_header) == 0  # a correct checksum sums to 0 over the whole header
    version_ihl = ip_header[0]
    assert version_ihl == 0x45  # IPv4, 20-byte header (no options)

    udp_header = frame[34:42]
    src_port, dst_port, udp_len, _ = struct.unpack("!HHHH", udp_header)
    assert (src_port, dst_port) == (8000, 8000)
    assert udp_len == 8 + len(payload)
    assert frame[42:] == payload
