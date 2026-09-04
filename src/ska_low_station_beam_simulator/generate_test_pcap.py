"""
Generates a small .pcap file containing a handful of heaps produced by
DirectSynthesisStreamer's noise floor (content doesn't matter for testing
SPEAD encoding — see below), for feeding into an external SPEAD unpacker
or spead2.recv's own pcap-file reader.

spead2 has NO pcap WRITER — confirmed by inspecting the installed
spead2==4.4.1 API directly, not assumed: spead2.send.BytesStream
captures only the raw SPEAD-protocol bytes (getvalue() -> bytes), with
no Ethernet/IP/UDP framing and no pcap record headers at all. It is NOT
a pcap file on its own. spead2 DOES have a pcap file READER on the
receive side (spead2.recv.Stream.add_udp_pcap_file_reader, backed by the
bundled libpcap) — confirmation that a real pcap (real network framing)
is the expected input format for testing against spead2 or an external
unpacker. So this module does the Ethernet/IPv4/UDP + pcap-record
wrapping itself: capture each heap's raw SPEAD bytes via
SpsPacketizer(stream=BytesStream(...)) (see common.SpsPacketizer's
`stream` parameter, added for exactly this), then wrap.

Only the noise floor is generated (no tone/pulsar, no delay_feed
needed) — the point of this file is exercising the SPEAD encoding path
(item packing, heap_counter, payload framing), not signal content.

Building this actually caught a real, previously-undetected production
bug (common.SpsPacketizer.send_channel_heap's heap_counter formula
multiplied by CHANNEL_WIDTH_HZ, the SAMPLE rate, instead of dividing by
BLOCK_DURATION_S, the correct per-HEAP rate — inflating it by 2048x,
enough to overflow spead2's actual 40-bit cnt limit for any current-era
timestamp and make every real send_channel_heap() call fail outright —
see the fix and its comment in common.py). This had never been caught
before because spead2 is explicitly not required to run this project's
test suite (see CLAUDE.md's Setup section) — this module is the first
thing that actually exercises send_channel_heap() end-to-end.

The output IS a correct, well-formed pcap — verified independently via
`tcpdump -r` (not just by this module's own logic) — and a SMALL
synthetic heap round-trips successfully through spead2.recv.Stream's
OWN `add_udp_pcap_file_reader`, confirming the encoding/framing is sound
at the protocol level. However, a FULL heap (the real ICD's 8192-byte
payload, ~9.2KB once SPEAD-encoded and UDP/IP/Ethernet-wrapped) is
silently DROPPED when read back that same way — `add_udp_pcap_file_reader`
takes only `(filename, filter)`, with no exposed way to raise whatever
internal packet-size limit is causing this (not a `StreamConfig` option
either). This is a real limitation of spead2's OWN pcap reader, not a
flaw in the pcap this module produces — to inspect full-size heaps,
use `tcpdump`/Wireshark or an external unpacker (this codebase's
"rudimentary" one) instead of spead2.recv for reading this file back.

Run: python -m ska_low_station_beam_simulator.generate_test_pcap [output.pcap] [n_heaps]
"""

from __future__ import annotations

import socket
import struct
import sys
import time

import spead2
import spead2.send

from ska_low_station_beam_simulator.common import (
    ChannelHeap,
    HeapAccumulator,
    SpsPacketizer,
    StationConfig,
)
from ska_low_station_beam_simulator.direct_synthesis import DirectSynthesisStreamer

# Synthetic network addressing -- arbitrary, content doesn't matter for
# testing SPEAD encoding, just needs to be well-formed.
SRC_MAC = bytes.fromhex("020000000001")  # locally-administered, avoids clashing with a real vendor OUI
DST_MAC = bytes.fromhex("020000000002")
SRC_IP = "10.0.0.1"
DST_IP = "10.0.0.2"
SRC_PORT = 8000
DST_PORT = 8000

PCAP_MAGIC_MICROSECONDS = 0xA1B2C3D4
LINKTYPE_ETHERNET = 1


def _ipv4_checksum(header: bytes) -> int:
    """Standard one's-complement checksum over an IPv4 header (with the
    checksum field itself zeroed when called). Mandatory for a
    well-formed IPv4 header, unlike the UDP checksum below."""
    if len(header) % 2:
        header += b"\x00"
    total = sum(struct.unpack(f"!{len(header) // 2}H", header))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _wrap_udp_frame(
    payload: bytes, src_ip: str, dst_ip: str, src_port: int, dst_port: int
) -> bytes:
    """Wraps `payload` in a synthetic Ethernet + IPv4 + UDP frame.

    UDP checksum is left as 0 ("no checksum computed"), which is
    explicitly valid for IPv4 (RFC 768) — avoids needing the UDP
    pseudo-header checksum for test content that doesn't need to survive
    real-world corruption detection. The IPv4 header checksum IS
    computed correctly, since that one is mandatory for a well-formed
    header (some parsers, and spead2's own pcap reader, may reject or
    warn on a bad one)."""
    udp_len = 8 + len(payload)
    udp_header = struct.pack("!HHHH", src_port, dst_port, udp_len, 0)

    total_len = 20 + udp_len
    src_addr = socket.inet_aton(src_ip)
    dst_addr = socket.inet_aton(dst_ip)
    ip_header_no_checksum = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, total_len, 0, 0, 64, socket.IPPROTO_UDP, 0,
        src_addr, dst_addr,
    )
    checksum = _ipv4_checksum(ip_header_no_checksum)
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, total_len, 0, 0, 64, socket.IPPROTO_UDP, checksum,
        src_addr, dst_addr,
    )

    eth_header = DST_MAC + SRC_MAC + struct.pack("!H", 0x0800)  # 0x0800 = IPv4
    return eth_header + ip_header + udp_header + payload


def _pcap_global_header() -> bytes:
    return struct.pack(
        "<IHHiIII", PCAP_MAGIC_MICROSECONDS, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET
    )


def _pcap_record(frame: bytes, timestamp: float) -> bytes:
    ts_sec = int(timestamp)
    ts_usec = int(round((timestamp - ts_sec) * 1_000_000))
    return struct.pack("<IIII", ts_sec, ts_usec, len(frame), len(frame)) + frame


def _capture_heap_bytes(heap: ChannelHeap, station: StationConfig) -> bytes:
    """Uses SpsPacketizer backed by a fresh BytesStream to capture
    exactly this heap's raw SPEAD bytes -- a fresh stream per heap avoids
    needing to track byte offsets into a shared, growing buffer."""
    stream = spead2.send.BytesStream(spead2.ThreadPool())
    packetizer = SpsPacketizer(station, stream=stream)
    packetizer.send_channel_heap(heap)
    return stream.getvalue()


def generate_test_pcap(output_path: str, n_heaps: int = 5) -> None:
    station = StationConfig(station_id=1, substation_id=0, subarray_id=1, beam_id=1, scan_id=1)
    obs_time = time.time()
    noise_cfg = {"std": 0.05, "seed": station.station_id}

    streamer = DirectSynthesisStreamer(
        station=station, source_cfgs=[], noise_cfg=noise_cfg, obs_time_ref=obs_time,
    )
    accumulator = HeapAccumulator(
        streamer.num_channels, obs_time, streamer.channel_output_rate,
        channel_id_map=streamer.channel_id_map,
    )

    n_samples = streamer.tick_n_samples()
    tick_dt = n_samples / streamer.channel_output_rate

    heaps: list[ChannelHeap] = []
    tick = 0
    while len(heaps) < n_heaps:
        t = obs_time + tick * tick_dt
        raw = streamer.generate_next_tick(t, n_samples)
        for pol, chunk in raw.items():
            if chunk is not None:
                accumulator.add(pol, chunk)
        heaps.extend(accumulator.pop_ready_heaps())
        tick += 1

    with open(output_path, "wb") as f:
        f.write(_pcap_global_header())
        for heap in heaps[:n_heaps]:
            raw_bytes = _capture_heap_bytes(heap, station)
            frame = _wrap_udp_frame(raw_bytes, SRC_IP, DST_IP, SRC_PORT, DST_PORT)
            f.write(_pcap_record(frame, heap.heap_start_time))

    print(f"wrote {n_heaps} heaps ({len(heaps)} available) to {output_path}")


if __name__ == "__main__":
    _output_path = sys.argv[1] if len(sys.argv) > 1 else "test_heaps.pcap"
    _n_heaps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    generate_test_pcap(_output_path, _n_heaps)
