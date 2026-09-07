"""
Generates a small .pcap file containing a handful of heaps produced by
``DirectSynthesisStreamer``'s noise floor (content doesn't matter for
testing SPEAD encoding — see below), for feeding into an external SPEAD
unpacker.

``common.SpsPacketizer`` hand-rolls its own SPEAD-64-48 encoder rather
than using spead2 (see common.py's module docstring for why: spead2's
packet encoder always writes 4 reserved item pointers CBF's real 6-item
ICD heap has no room for, and there's no way to configure it not to).
One consequence of that: this file's output is NOT expected to be
parseable by a generic SPEAD reader like ``spead2.recv`` — it
deliberately omits HEAP_LENGTH and repurposes the payload-offset item's
ID, exactly per the ICD, not per generic SPEAD. So this module does its
own Ethernet/IPv4/UDP + pcap-record wrapping around
``SpsPacketizer.encode_channel_heap``'s raw bytes; verify the result
with ``tcpdump``/Wireshark or an external unpacker (this codebase's
"rudimentary" one), not ``spead2.recv``.

Only the noise floor is generated (no tone/pulsar, no delay_feed
needed) — the point of this file is exercising the SPEAD encoding path
(item packing, heap_counter, payload framing), not signal content. This
is the only thing in this project that exercises
``send_channel_heap()``/``encode_channel_heap()`` end-to-end (see
CLAUDE.md's bug #17 for a real production bug this caught).

The output IS a correct, well-formed pcap — verified independently via
``tcpdump -r``, not just by this module's own logic.

Run::

    python -m ska_low_station_beam_simulator.generate_test_pcap [output.pcap] [n_heaps]
"""

from __future__ import annotations

import socket
import struct
import sys
import time

from ska_low_station_beam_simulator.common import (
    ChannelHeap,
    HeapAccumulator,
    SpsPacketizer,
    StationConfig,
)
from ska_low_station_beam_simulator.direct_synthesis import (
    DirectSynthesisStreamer,
    NoiseConfig,
)

# Synthetic network addressing -- arbitrary, content doesn't matter for
# testing SPEAD encoding, just needs to be well-formed.
SRC_MAC = bytes.fromhex(
    "020000000001"
)  # locally-administered, avoids clashing with a real vendor OUI
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
    well-formed IPv4 header, unlike the UDP checksum below.

    :param header: the raw IPv4 header bytes, with the checksum field
        zeroed.
    :returns: the 16-bit one's-complement checksum.
    """
    if len(header) % 2:
        header += b"\x00"
    total = sum(struct.unpack(f"!{len(header) // 2}H", header))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _wrap_udp_frame(
    payload: bytes, src_ip: str, dst_ip: str, src_port: int, dst_port: int
) -> bytes:
    """Wraps ``payload`` in a synthetic Ethernet + IPv4 + UDP frame.

    UDP checksum is left as 0 ("no checksum computed"), which is
    explicitly valid for IPv4 (RFC 768) — avoids needing the UDP
    pseudo-header checksum for test content that doesn't need to survive
    real-world corruption detection. The IPv4 header checksum IS
    computed correctly, since that one is mandatory for a well-formed
    header (some parsers, and spead2's own pcap reader, may reject or
    warn on a bad one).

    :param payload: the UDP payload (a SPEAD-encoded heap, in this
        module's usage).
    :param src_ip: source IPv4 address, dotted-quad string.
    :param dst_ip: destination IPv4 address, dotted-quad string.
    :param src_port: source UDP port.
    :param dst_port: destination UDP port.
    :returns: the complete Ethernet + IPv4 + UDP frame, ready to write
        as one pcap record.
    """
    udp_len = 8 + len(payload)
    udp_header = struct.pack("!HHHH", src_port, dst_port, udp_len, 0)

    total_len = 20 + udp_len
    src_addr = socket.inet_aton(src_ip)
    dst_addr = socket.inet_aton(dst_ip)
    ip_header_no_checksum = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_len,
        0,
        0,
        64,
        socket.IPPROTO_UDP,
        0,
        src_addr,
        dst_addr,
    )
    checksum = _ipv4_checksum(ip_header_no_checksum)
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_len,
        0,
        0,
        64,
        socket.IPPROTO_UDP,
        checksum,
        src_addr,
        dst_addr,
    )

    eth_header = DST_MAC + SRC_MAC + struct.pack("!H", 0x0800)  # 0x0800 = IPv4
    return eth_header + ip_header + udp_header + payload


def _pcap_global_header() -> bytes:
    return struct.pack(
        "<IHHiIII", PCAP_MAGIC_MICROSECONDS, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET
    )


def _pcap_record(frame: bytes, timestamp: float) -> bytes:
    ts_sec = int(timestamp)
    ts_usec = round((timestamp - ts_sec) * 1_000_000)
    return struct.pack("<IIII", ts_sec, ts_usec, len(frame), len(frame)) + frame


def generate_test_pcap(output_path: str, n_heaps: int = 5) -> None:
    station = StationConfig(
        station_id=1, substation_id=0, subarray_id=1, beam_id=1, scan_id=1
    )
    obs_time = time.time()
    noise_cfg = NoiseConfig(std=0.05, seed=station.station_id)

    streamer = DirectSynthesisStreamer(
        station=station,
        source_cfgs=[],
        noise_cfg=noise_cfg,
        obs_time_ref=obs_time,
    )
    accumulator = HeapAccumulator(
        streamer.num_channels,
        obs_time,
        streamer.channel_output_rate,
        channel_id_map=streamer.channel_id_map,
    )
    # No dest_ip/sock needed -- this module only calls encode_channel_heap,
    # never send_channel_heap, so SpsPacketizer never touches a socket.
    packetizer = SpsPacketizer(station)

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
            raw_bytes = packetizer.encode_channel_heap(heap)
            frame = _wrap_udp_frame(raw_bytes, SRC_IP, DST_IP, SRC_PORT, DST_PORT)
            f.write(_pcap_record(frame, heap.heap_start_time))

    print(f"wrote {n_heaps} heaps ({len(heaps)} available) to {output_path}")


if __name__ == "__main__":
    _output_path = sys.argv[1] if len(sys.argv) > 1 else "test_heaps.pcap"
    _n_heaps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    generate_test_pcap(_output_path, _n_heaps)
