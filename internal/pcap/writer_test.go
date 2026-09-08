package pcap

import (
	"bytes"
	"encoding/binary"
	"testing"
	"time"
)

// readGlobalHeader validates and consumes the 24-byte pcap global header,
// returning the remaining bytes.
func readGlobalHeader(t *testing.T, data []byte) []byte {
	t.Helper()
	if len(data) < 24 {
		t.Fatalf("file too short for a pcap global header: %d bytes", len(data))
	}
	if magic := binary.LittleEndian.Uint32(data[0:4]); magic != 0xa1b2c3d4 {
		t.Fatalf("magic number = %#x, want 0xa1b2c3d4", magic)
	}
	if network := binary.LittleEndian.Uint32(data[20:24]); network != linkTypeEthernet {
		t.Fatalf("network = %d, want %d (Ethernet)", network, linkTypeEthernet)
	}
	return data[24:]
}

// readOnePacket parses one pcap record (16-byte record header + frame)
// off the front of data, asserts it's an Ethernet/IPv4/UDP frame with a
// valid IPv4 header checksum, and returns the UDP payload plus whatever
// bytes remain after this record.
func readOnePacket(t *testing.T, data []byte) (payload []byte, rest []byte) {
	t.Helper()
	if len(data) < 16 {
		t.Fatalf("not enough bytes for a pcap record header: %d", len(data))
	}
	inclLen := binary.LittleEndian.Uint32(data[8:12])
	origLen := binary.LittleEndian.Uint32(data[12:16])
	if inclLen != origLen {
		t.Fatalf("incl_len (%d) != orig_len (%d) -- writer never truncates", inclLen, origLen)
	}
	frame := data[16 : 16+inclLen]
	rest = data[16+inclLen:]

	if len(frame) < ethernetHeaderLen+ipv4HeaderLen+udpHeaderLen {
		t.Fatalf("frame too short: %d bytes", len(frame))
	}
	if etherType := binary.BigEndian.Uint16(frame[12:14]); etherType != etherTypeIPv4 {
		t.Fatalf("ethertype = %#x, want %#x", etherType, etherTypeIPv4)
	}

	ip := frame[ethernetHeaderLen : ethernetHeaderLen+ipv4HeaderLen]
	if versionIHL := ip[0]; versionIHL != 0x45 {
		t.Fatalf("IPv4 version/IHL = %#x, want 0x45", versionIHL)
	}
	if proto := ip[9]; proto != ipProtoUDP {
		t.Fatalf("IP protocol = %d, want %d (UDP)", proto, ipProtoUDP)
	}
	if sum := internetChecksum(ip); sum != 0 {
		t.Errorf("IPv4 header checksum does not validate: internetChecksum(ip) = %#x, want 0", sum)
	}
	ipTotalLen := binary.BigEndian.Uint16(ip[2:4])
	if int(ipTotalLen) != len(frame)-ethernetHeaderLen {
		t.Fatalf("IP total_length = %d, want %d", ipTotalLen, len(frame)-ethernetHeaderLen)
	}

	udp := frame[ethernetHeaderLen+ipv4HeaderLen:]
	udpLen := binary.BigEndian.Uint16(udp[4:6])
	if int(udpLen) != len(udp) {
		t.Fatalf("UDP length = %d, want %d", udpLen, len(udp))
	}
	return udp[udpHeaderLen:], rest
}

func TestWriter_RoundTrip(t *testing.T) {
	var buf bytes.Buffer
	pw, err := NewWriter(&buf)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	payloads := [][]byte{
		bytes.Repeat([]byte{0xAB}, 8248), // matches internal/spead's heapWireSizeBytes
		{0x01, 0x02, 0x03},               // odd length, exercises internetChecksum's odd-byte tail
	}
	ts := time.Unix(1700000000, 123456000)
	for _, p := range payloads {
		if err := pw.WriteUDPPacket(p, ts); err != nil {
			t.Fatalf("WriteUDPPacket: %v", err)
		}
	}

	data := readGlobalHeader(t, buf.Bytes())
	for i, want := range payloads {
		var got []byte
		got, data = readOnePacket(t, data)
		if !bytes.Equal(got, want) {
			t.Errorf("packet %d payload mismatch: got %d bytes, want %d bytes", i, len(got), len(want))
		}
	}
	if len(data) != 0 {
		t.Errorf("%d trailing bytes after the expected %d packets", len(data), len(payloads))
	}
}

func TestWriter_PayloadTooLarge(t *testing.T) {
	var buf bytes.Buffer
	pw, err := NewWriter(&buf)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	huge := make([]byte, 0x10000) // ipLen would overflow the IPv4 total_length field
	if err := pw.WriteUDPPacket(huge, time.Now()); err == nil {
		t.Fatal("expected an error for an oversized payload, got nil")
	}
}

func TestInternetChecksum_SelfConsistent(t *testing.T) {
	// The defining property (RFC 1071 §4(B)): summing a header together
	// with its own already-computed checksum field, appended as one more
	// word, yields zero. internetChecksum is only ever called in
	// production on the fixed-size (always even-length) 20-byte IPv4
	// header, so this only needs to hold for even-length input --
	// appending a word to an ODD-length input shifts byte alignment and
	// doesn't preserve the property, which is a fact about word
	// alignment, not a bug in the function.
	b := []byte{0x45, 0x00, 0x00, 0x1c, 0x1c, 0x46, 0x40, 0x00, 0x40, 0x06, 0x00, 0x00, 0xac, 0x10, 0x0a, 0x63, 0xac, 0x10, 0x0a, 0x0c}
	sum := internetChecksum(b)
	withChecksum := append(append([]byte{}, b...), byte(sum>>8), byte(sum))
	if got := internetChecksum(withChecksum); got != 0 {
		t.Errorf("internetChecksum(header+checksum) = %#x, want 0", got)
	}
}
