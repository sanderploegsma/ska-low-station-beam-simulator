// Package pcap hand-rolls a minimal classic-pcap file writer: given a
// UDP payload, it wraps it in a synthetic Ethernet/IPv4/UDP frame and
// appends it as one record to a libpcap-format capture file. Exists so
// cmd/pcap-dump can hand a generated SPEAD heap a wire-inspectable
// structure (openable directly in Wireshark/tcpdump) without ever
// touching a real socket.
//
// Source/destination MAC/IP/port are fixed placeholders, not meant to
// represent a real capture -- same "not representative, just structural"
// spirit as cmd/pcap-dump's noise-only sample content. No general-purpose
// pcap/packet library is used here, matching this codebase's existing
// stance on hand-rolling small, fully-understood wire encoders (see
// internal/spead's doc comment) rather than pulling in a dependency for a
// handful of fixed-layout header structs.
package pcap

import (
	"encoding/binary"
	"fmt"
	"io"
	"time"
)

const (
	linkTypeEthernet = 1 // pcap global header's "network" field: DLT_EN10MB

	etherTypeIPv4 = 0x0800
	ipProtoUDP    = 17

	ethernetHeaderLen = 14
	ipv4HeaderLen     = 20
	udpHeaderLen      = 8
)

// Writer appends synthetic Ethernet/IPv4/UDP-framed packets to a classic
// pcap file. Not safe for concurrent use.
type Writer struct {
	w io.Writer

	srcMAC, dstMAC [6]byte
	srcIP, dstIP   [4]byte
	srcPort        uint16
	dstPort        uint16

	ipID uint16 // IPv4 identification, incremented per packet
}

// NewWriter writes a pcap global header (Ethernet link type) to w and
// returns a Writer ready for WriteUDPPacket calls. Source/destination
// addresses are fixed placeholders -- see package doc comment.
func NewWriter(w io.Writer) (*Writer, error) {
	pw := &Writer{
		w:       w,
		srcMAC:  [6]byte{0x02, 0x00, 0x00, 0x00, 0x00, 0x01}, // locally-administered, arbitrary
		dstMAC:  [6]byte{0x02, 0x00, 0x00, 0x00, 0x00, 0x02},
		srcIP:   [4]byte{10, 0, 0, 1},
		dstIP:   [4]byte{10, 0, 0, 2},
		srcPort: 40000,
		dstPort: 8000, // matches this project's other CLIs' default -dest-port
	}
	if err := pw.writeGlobalHeader(); err != nil {
		return nil, fmt.Errorf("writing pcap global header: %w", err)
	}
	return pw, nil
}

func (pw *Writer) writeGlobalHeader() error {
	hdr := make([]byte, 24)
	binary.LittleEndian.PutUint32(hdr[0:4], 0xa1b2c3d4) // magic number (also fixes byte order for readers)
	binary.LittleEndian.PutUint16(hdr[4:6], 2)           // version_major
	binary.LittleEndian.PutUint16(hdr[6:8], 4)           // version_minor
	// hdr[8:16] thiszone/sigfigs left 0
	binary.LittleEndian.PutUint32(hdr[16:20], 65535) // snaplen
	binary.LittleEndian.PutUint32(hdr[20:24], linkTypeEthernet)
	_, err := pw.w.Write(hdr)
	return err
}

// WriteUDPPacket appends one packet record: payload wrapped in a
// synthetic Ethernet/IPv4/UDP header, timestamped ts. payload becomes the
// UDP datagram's body verbatim (e.g. one encoded SPEAD heap).
func (pw *Writer) WriteUDPPacket(payload []byte, ts time.Time) error {
	udpLen := udpHeaderLen + len(payload)
	ipLen := ipv4HeaderLen + udpLen
	if ipLen > 0xffff {
		return fmt.Errorf("payload too large for one IPv4 packet: %d bytes", len(payload))
	}

	frame := make([]byte, ethernetHeaderLen+ipLen)

	copy(frame[0:6], pw.dstMAC[:])
	copy(frame[6:12], pw.srcMAC[:])
	binary.BigEndian.PutUint16(frame[12:14], etherTypeIPv4)

	ip := frame[ethernetHeaderLen : ethernetHeaderLen+ipv4HeaderLen]
	ip[0] = 0x45 // version 4, IHL 5 (no options)
	ip[1] = 0    // DSCP/ECN
	binary.BigEndian.PutUint16(ip[2:4], uint16(ipLen))
	pw.ipID++
	binary.BigEndian.PutUint16(ip[4:6], pw.ipID)
	binary.BigEndian.PutUint16(ip[6:8], 0) // flags/fragment offset: never fragmented
	ip[8] = 64                             // TTL
	ip[9] = ipProtoUDP
	binary.BigEndian.PutUint16(ip[10:12], 0) // checksum: filled in below, once the rest of the header is final
	copy(ip[12:16], pw.srcIP[:])
	copy(ip[16:20], pw.dstIP[:])
	binary.BigEndian.PutUint16(ip[10:12], internetChecksum(ip))

	udp := frame[ethernetHeaderLen+ipv4HeaderLen:]
	binary.BigEndian.PutUint16(udp[0:2], pw.srcPort)
	binary.BigEndian.PutUint16(udp[2:4], pw.dstPort)
	binary.BigEndian.PutUint16(udp[4:6], uint16(udpLen))
	binary.BigEndian.PutUint16(udp[6:8], 0) // checksum: 0 ("not computed") is valid over IPv4
	copy(udp[8:], payload)

	recHdr := make([]byte, 16)
	binary.LittleEndian.PutUint32(recHdr[0:4], uint32(ts.Unix()))
	binary.LittleEndian.PutUint32(recHdr[4:8], uint32(ts.Nanosecond()/1000))
	binary.LittleEndian.PutUint32(recHdr[8:12], uint32(len(frame)))
	binary.LittleEndian.PutUint32(recHdr[12:16], uint32(len(frame)))

	if _, err := pw.w.Write(recHdr); err != nil {
		return err
	}
	_, err := pw.w.Write(frame)
	return err
}

// internetChecksum computes the standard one's-complement Internet
// checksum (RFC 791/1071) over b, which must be the IPv4 header with its
// own checksum field still zeroed.
func internetChecksum(b []byte) uint16 {
	var sum uint32
	for i := 0; i+1 < len(b); i += 2 {
		sum += uint32(b[i])<<8 | uint32(b[i+1])
	}
	if len(b)%2 == 1 {
		sum += uint32(b[len(b)-1]) << 8
	}
	for sum > 0xffff {
		sum = (sum & 0xffff) + (sum >> 16)
	}
	return ^uint16(sum)
}
