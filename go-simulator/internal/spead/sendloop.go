package spead

import (
	"log"

	"github.com/skao/station-beam-simulator-go/internal/common"
)

// SendLoop reads heaps from recv and sends them via packetizer until
// shutdown is closed — the equivalent of Python's common.sender_loop.
// Shared by every entrypoint that pairs a common.HeapQueue with an
// SpsPacketizer (the gRPC-served simulator and the standalone
// noise-only CLI); lives here rather than in internal/common because
// common must not depend on spead (spead already depends on common).
func SendLoop(recv <-chan *common.ChannelHeap, packetizer *SpsPacketizer, shutdown <-chan struct{}) {
	for {
		select {
		case <-shutdown:
			return
		case heap := <-recv:
			if err := packetizer.SendChannelHeap(heap); err != nil {
				log.Printf("failed to send heap ch=%d t=%.4f: %v", heap.ChannelID, heap.HeapStartTime, err)
			}
		}
	}
}
