package spead

import (
	"fmt"
	"log"
	"net"
	"sync"

	"golang.org/x/net/ipv4"

	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/common"
)

// Sender-pool defaults. Chosen to give a comfortable margin over the
// single-goroutine/single-syscall-per-heap baseline without needing any
// per-deployment tuning; override via cmd/server's flags or
// noise-stream's flags if a specific target host needs something else.
const (
	// DefaultNumSenders: parallel UDP sockets/goroutines draining the
	// send queue. Each gets its own source port, which also spreads
	// outbound traffic across the NIC's/receiver's flow hash (RSS)
	// instead of pinning everything to one queue/core.
	DefaultNumSenders = 4
	// DefaultSendBatchSize: max heaps per BatchSender.WriteBatch call
	// (sendmmsg(2) on Linux).
	DefaultSendBatchSize = 32
	// DefaultUDPSendBufferBytes: SO_SNDBUF per socket. The OS default
	// (net.core.wmem_default, often ~208KB) is easily overrun by this
	// project's per-tick heap bursts; 8MiB gives real headroom without
	// being a meaningful memory cost per pod.
	DefaultUDPSendBufferBytes = 8 << 20

	// fullBandNumSenders: sender goroutines confirmed on real 100G
	// SR-IOV target hardware to keep up at common.MaxNumChannels (384,
	// the full band) -- see DefaultNumSendersForChannels. This replaced
	// a flat DefaultNumSenders=4 default that the same real-hardware
	// testing outgrew twice in one session (4, then 8) once the
	// producer-side fixes elsewhere in this package's history (see
	// HeapAccumulator's doc comment) stopped being the bottleneck: once
	// generation got fast enough, the send side needed to absorb a much
	// higher sustained throughput than a fixed small sender count could
	// drain.
	fullBandNumSenders = 16
)

// DefaultNumSendersForChannels scales the sender-goroutine count with
// numChannels, using fullBandNumSenders at common.MaxNumChannels (384)
// as the confirmed real-hardware baseline and proportionally fewer for
// a narrower configuration -- a station running fewer channels sends
// proportionally less data, so needs proportionally fewer sockets
// draining it. Not used by every caller: cmd/server's gRPC server
// creates its SenderPool once at process Start(), before any scan (and
// therefore its num_channels) is known, so it still uses the flat
// DefaultNumSenders unless a caller overrides it explicitly via
// WithNumSenders -- only cmd/noise-stream, which takes -num-channels
// upfront on the command line, can compute this at pool-creation time.
func DefaultNumSendersForChannels(numChannels int) int {
	n := (fullBandNumSenders*numChannels + common.MaxNumChannels - 1) / common.MaxNumChannels
	if n < 1 {
		n = 1
	}
	return n
}

// BatchSender is anything that can send multiple already-encoded heap
// payloads in one call — a batched UDP socket write in production, or a
// fake in tests.
type BatchSender interface {
	WriteBatch(bufs [][]byte) (n int, err error)
}

// UDPBatchSender wraps one connected *net.UDPConn for batched sends via
// golang.org/x/net/ipv4's PacketConn.WriteBatch, which uses Linux's
// sendmmsg(2) under the hood on this project's deployment target — one
// syscall for a whole batch of datagrams instead of one syscall per
// datagram. x/net/ipv4 falls back to a plain per-message loop on
// non-Linux platforms, so this is safe to construct anywhere (e.g. this
// module's darwin/arm64 dev machines); it only gets the syscall-count
// win on Linux.
type UDPBatchSender struct {
	pc *ipv4.PacketConn
}

// NewUDPBatchSender wraps conn (already net.DialUDP'd to its
// destination) for batched sends.
func NewUDPBatchSender(conn *net.UDPConn) *UDPBatchSender {
	return &UDPBatchSender{pc: ipv4.NewPacketConn(conn)}
}

// WriteBatch implements BatchSender.
func (s *UDPBatchSender) WriteBatch(bufs [][]byte) (int, error) {
	if len(bufs) == 0 {
		return 0, nil
	}
	msgs := make([]ipv4.Message, len(bufs))
	for i, b := range bufs {
		msgs[i].Buffers = [][]byte{b}
	}
	return s.pc.WriteBatch(msgs, 0)
}

// UDPSenderSockets dials n independent UDP sockets to destAddr, each
// bound to localAddr (nil: let the OS pick), each with its send buffer
// sized to sndBufBytes (0: leave at the OS default). Using n separate
// sockets — rather than n goroutines sharing one — is deliberate: a
// *net.UDPConn is safe for concurrent use by multiple goroutines, but
// sharing one socket still funnels every packet through one 5-tuple
// (fixed source+dest IP/port), which is what pins a flow to a single
// NIC queue/CPU core on both ends regardless of link speed; n distinct
// sockets get n distinct (OS-assigned) source ports instead.
func UDPSenderSockets(localAddr, destAddr *net.UDPAddr, n, sndBufBytes int) ([]*net.UDPConn, error) {
	if n < 1 {
		n = 1
	}
	conns := make([]*net.UDPConn, 0, n)
	for i := 0; i < n; i++ {
		conn, err := net.DialUDP("udp", localAddr, destAddr)
		if err != nil {
			for _, c := range conns {
				c.Close()
			}
			return nil, fmt.Errorf("dialing SPEAD destination %s (socket %d/%d): %w", destAddr, i+1, n, err)
		}
		if sndBufBytes > 0 {
			if err := conn.SetWriteBuffer(sndBufBytes); err != nil {
				// Not fatal -- some kernels/permission setups cap or
				// reject this (e.g. below net.core.wmem_max without
				// CAP_NET_ADMIN); surface it and keep the OS default
				// rather than failing the whole pod over it.
				log.Printf("warning: failed to set SO_SNDBUF=%d on outbound SPEAD/UDP socket %d/%d: %v", sndBufBytes, i+1, n, err)
			}
		}
		conns = append(conns, conn)
	}
	return conns, nil
}

// SenderPool owns n UDP sockets and one BatchSendLoop goroutine per
// socket, all draining the same heap queue in parallel — the shared
// setup/teardown both the gRPC-served simulator and the standalone
// noise-only CLI need around UDPSenderSockets/BatchSendLoop.
type SenderPool struct {
	conns      []*net.UDPConn
	wg         sync.WaitGroup
	packetizer *SpsPacketizer
}

// NewSenderPool dials n sockets (see UDPSenderSockets) and starts one
// BatchSendLoop per socket pulling heaps off recv until shutdown is
// closed.
func NewSenderPool(station *common.StationConfig, recv <-chan *common.ChannelHeap, localAddr, destAddr *net.UDPAddr, n, sndBufBytes, batchSize int, shutdown <-chan struct{}) (*SenderPool, error) {
	conns, err := UDPSenderSockets(localAddr, destAddr, n, sndBufBytes)
	if err != nil {
		return nil, err
	}
	// EncodeChannelHeap doesn't touch the packetizer's sender field, so
	// one packetizer (constructed with no sender of its own) is safely
	// shared read-only across every BatchSendLoop goroutine -- each
	// supplies its own BatchSender/socket as a separate argument.
	packetizer := NewSpsPacketizer(station, nil)

	pool := &SenderPool{conns: conns, packetizer: packetizer}
	for _, conn := range conns {
		sender := NewUDPBatchSender(conn)
		pool.wg.Add(1)
		go func() {
			defer pool.wg.Done()
			BatchSendLoop(recv, packetizer, sender, shutdown, batchSize)
		}()
	}
	return pool, nil
}

// SetQuantizeScale updates the shared packetizer's fixed quantization
// scale (see SpsPacketizer.SetQuantizeScale) -- safe to call while
// sender goroutines are already running (atomic under the hood), and
// safe to call more than once across a process's lifetime, since one
// SenderPool -- and the single packetizer its goroutines share -- can
// outlive many scans with different noise/tone configs on the
// gRPC-served path (see server.Server.Start's doc comment).
func (p *SenderPool) SetQuantizeScale(scale float64) {
	p.packetizer.SetQuantizeScale(scale)
}

// Close waits for every sender goroutine to return, then closes all
// sockets. Callers must close shutdown (or otherwise ensure the send
// loops are stopping) before calling Close -- waiting first avoids a
// heap already pulled off the queue losing its race against a socket
// Close() and failing with "use of closed network connection".
func (p *SenderPool) Close() {
	p.wg.Wait()
	for _, c := range p.conns {
		c.Close()
	}
}
