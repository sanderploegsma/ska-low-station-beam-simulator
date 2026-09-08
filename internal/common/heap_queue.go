package common

// HeapQueue is a bounded, non-blocking HeapSender backed by a channel —
// the Go equivalent of Python's queue.Queue(maxsize=...) send queue
// between the producer (ScanRunner) and a sender goroutine. Shared by
// every entrypoint that needs to hand ScanRunner's output to a separate
// sender goroutine (the gRPC-served simulator and the standalone
// noise-only CLI).
type HeapQueue struct {
	ch chan *ChannelHeap
}

// NewHeapQueue creates a HeapQueue with the given buffer size.
func NewHeapQueue(size int) *HeapQueue {
	return &HeapQueue{ch: make(chan *ChannelHeap, size)}
}

// Send implements HeapSender — non-blocking, returns false (dropped) if
// the queue is full rather than blocking the producer.
func (q *HeapQueue) Send(heap *ChannelHeap) bool {
	select {
	case q.ch <- heap:
		return true
	default:
		return false
	}
}

// Recv returns the receiving side of the queue, for a sender loop to
// range/select over.
func (q *HeapQueue) Recv() <-chan *ChannelHeap {
	return q.ch
}

// Len reports the number of heaps currently queued.
func (q *HeapQueue) Len() int {
	return len(q.ch)
}
