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
	// A pool of batchSize wire-size buffers, allocated ONCE for the life
	// of this goroutine and reused batch after batch — see
	// EncodeChannelHeapInto's doc comment for why this is safe:
	// profiling a real run on the target Linux hardware found the
	// per-heap `buf` allocation EncodeChannelHeap used to make was still
	// a dominant cost even after every other per-heap allocation had
	// been removed (~1.4GB/s of allocation traffic at 384 channels). A
	// UDP send (sendmmsg included) copies each buffer into the kernel
	// synchronously before returning, so once sender.WriteBatch(bufs)
	// below returns, every buffer in that batch is free to reuse for the
	// next one.
	pool := make([][]byte, batchSize)
	for i := range pool {
		pool[i] = make([]byte, heapWireSizeBytes)
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
			bufs = encodeHeapInto(bufs, pool, packetizer, heap)
		}

		// Then drain up to batchSize-1 more WITHOUT blocking, so a batch
		// never waits around for heaps that aren't there yet.
	drain:
		for len(bufs) < batchSize {
			select {
			case heap := <-recv:
				bufs = encodeHeapInto(bufs, pool, packetizer, heap)
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

// encodeHeapInto encodes heap into the next unused buffer in pool
// (pool[len(bufs)] — always the first slot not yet claimed by this
// batch) and appends it to bufs, logging (not failing the whole batch)
// on a per-heap encode error — matching the old per-heap SendLoop's
// failure granularity. A failed encode leaves that pool slot unclaimed,
// so the next heap tried this batch reuses the same slot.
//
// Releases heap's sample buffers back to common.ReleaseSampleBuffers
// either way (success or failure) — this is the last point anything
// reads heap.VSamples/HSamples, so it's also the earliest safe point to
// hand them back to HeapAccumulator.PrepareWrite's pool for a future
// tick to reuse (see that method's doc comment for why this exists: it
// avoids paying make()'s zero-fill on every fresh per-channel buffer,
// every tick).
func encodeHeapInto(bufs [][]byte, pool [][]byte, packetizer *SpsPacketizer, heap *common.ChannelHeap) [][]byte {
	dst := pool[len(bufs)]
	err := packetizer.EncodeChannelHeapInto(dst, heap)
	common.ReleaseSampleBuffers(heap)
	if err != nil {
		log.Printf("failed to encode heap ch=%d t=%.4f: %v", heap.ChannelID, heap.HeapStartTime, err)
		return bufs
	}
	return append(bufs, dst)
}
