package spead

import (
	"log"

	"github.com/skao/station-beam-simulator-go/internal/common"
)

// BatchSendLoop reads heaps from recv, encodes them, and writes them to
// sender in batches (up to batchSize heaps per BatchSender.WriteBatch
// call — one sendmmsg(2) syscall per batch on Linux, instead of one
// sendto(2)-equivalent syscall per heap) until shutdown is closed.
//
// Any number of BatchSendLoop calls may run concurrently against the
// SAME recv channel with no extra coordination — Go fans a channel's
// receives out across goroutines for free — as long as each call is
// given its OWN sender/socket (see SenderPool/UDPSenderSockets: sharing
// one socket across goroutines would just move the bottleneck from
// "one goroutine" to "one socket", and using separate sockets also
// spreads outbound traffic across separate source ports, avoiding a
// single flow-hash/RSS queue on the wire). UDP heaps carry no ordering
// requirement CBF depends on — each heap self-identifies via its own
// SPEAD header (scan_id/frequency_id/heap_counter) — so heaps completing
// out of order across parallel senders is harmless.
func BatchSendLoop(recv <-chan *common.ChannelHeap, packetizer *SpsPacketizer, sender BatchSender, shutdown <-chan struct{}, batchSize int) {
	if batchSize < 1 {
		batchSize = 1
	}
	bufs := make([][]byte, 0, batchSize)
	for {
		bufs = bufs[:0]

		// Block for at least one heap (or shutdown) so this goroutine
		// doesn't spin when the queue is empty.
		select {
		case <-shutdown:
			return
		case heap := <-recv:
			bufs = encodeHeapInto(bufs, packetizer, heap)
		}

		// Then drain up to batchSize-1 more WITHOUT blocking, so a batch
		// never waits around for heaps that aren't there yet.
	drain:
		for len(bufs) < batchSize {
			select {
			case heap := <-recv:
				bufs = encodeHeapInto(bufs, packetizer, heap)
			default:
				break drain
			}
		}

		if len(bufs) == 0 {
			continue
		}
		if _, err := sender.WriteBatch(bufs); err != nil {
			log.Printf("batch send failed (n=%d heaps): %v", len(bufs), err)
		}
	}
}

// encodeHeapInto encodes heap and appends it to bufs, logging (not
// failing the whole batch) on a per-heap encode error — matching the
// old per-heap SendLoop's failure granularity.
func encodeHeapInto(bufs [][]byte, packetizer *SpsPacketizer, heap *common.ChannelHeap) [][]byte {
	encoded, err := packetizer.EncodeChannelHeap(heap)
	if err != nil {
		log.Printf("failed to encode heap ch=%d t=%.4f: %v", heap.ChannelID, heap.HeapStartTime, err)
		return bufs
	}
	return append(bufs, encoded)
}
