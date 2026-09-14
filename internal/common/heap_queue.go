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

// ShardedHeapQueue fans a HeapSender out across n independent HeapQueues
// ("shards"), routing every heap for a given ChannelID to the SAME shard
// every time (see shardFor) -- so a sender goroutine that exclusively
// drains one shard (see spead.NewSenderPool) always sends a given
// channel's heaps via the same UDP socket, in the order Send received
// them. This exists specifically to fix a real ordering bug: a single
// shared queue drained by N sender goroutines/sockets (the previous
// design) let heaps for the SAME channel race across sockets, so two
// consecutive ticks for one channel could reach CBF out of order even
// though nothing was lost -- and CBF's real ingest firmware
// (ska-low-cbf-fw-corr's LFAAProcess100G.vhd) flags any non-consecutive
// heap_counter per virtual channel (one VC = one exact station/
// substation/subarray/beam/frequency_id tuple) as "out of order", with
// no reordering tolerance of its own. Different channels are still free
// to land on different shards for parallelism, same as before sharding.
type ShardedHeapQueue struct {
	shards []*HeapQueue
}

// NewShardedHeapQueue creates a ShardedHeapQueue with n shards (n<1 is
// treated as 1), each an independent HeapQueue of the given buffer size.
func NewShardedHeapQueue(n, sizePerShard int) *ShardedHeapQueue {
	if n < 1 {
		n = 1
	}
	shards := make([]*HeapQueue, n)
	for i := range shards {
		shards[i] = NewHeapQueue(sizePerShard)
	}
	return &ShardedHeapQueue{shards: shards}
}

// Send implements HeapSender, routing heap to the shard matching its
// ChannelID -- non-blocking, returning false (dropped) if that specific
// shard is full, exactly like HeapQueue.Send. A full shard only drops
// heaps for the channels hashed onto it, not every channel sharing the
// queue (an improvement over the single shared queue this replaces, not
// just a side effect of sharding).
func (q *ShardedHeapQueue) Send(heap *ChannelHeap) bool {
	return q.shards[shardFor(heap.ChannelID, len(q.shards))].Send(heap)
}

// NumShards returns the number of shards.
func (q *ShardedHeapQueue) NumShards() int {
	return len(q.shards)
}

// Recv returns the receiving side of shard i, for one sender goroutine
// to exclusively drain (see this type's doc comment for why exclusivity
// per shard is what preserves per-channel send order).
func (q *ShardedHeapQueue) Recv(i int) <-chan *ChannelHeap {
	return q.shards[i].Recv()
}

// Len reports the number of heaps currently queued, summed across every
// shard.
func (q *ShardedHeapQueue) Len() int {
	n := 0
	for _, s := range q.shards {
		n += s.Len()
	}
	return n
}

// shardFor picks channelID's shard out of n: a plain modulo is enough
// since ChannelID is a small, densely-packed, non-negative index (the
// ICD's global channel IDs run 64..447, never negative) -- no need for a
// stronger hash to spread channels evenly across shards.
func shardFor(channelID, n int) int {
	if channelID < 0 {
		channelID = -channelID
	}
	return channelID % n
}
