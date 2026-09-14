package common

import "testing"

func TestShardedHeapQueue_SameChannelAlwaysSameShard(t *testing.T) {
	// 20 channels % 4 shards divides evenly (5 channels/shard) x 5 sends
	// each = 25 heaps for the busiest shard; sizePerShard is set with
	// headroom above that so no send is dropped as an artifact of this
	// test's own volume rather than the property under test.
	q := NewShardedHeapQueue(4, 32)

	// Sending the same channel repeatedly must land every heap in the
	// SAME shard every time -- this is the whole point of sharding by
	// ChannelID (see this type's doc comment): a sender goroutine
	// exclusively draining one shard is what preserves per-channel send
	// order, so if this property doesn't hold, the fix doesn't work.
	for ch := 0; ch < 20; ch++ {
		for i := 0; i < 5; i++ {
			if !q.Send(&ChannelHeap{ChannelID: ch}) {
				t.Fatalf("channel %d, send %d: unexpectedly dropped (queue full)", ch, i)
			}
		}
	}

	for shard := 0; shard < q.NumShards(); shard++ {
		recv := q.Recv(shard)
		for len(recv) > 0 {
			heap := <-recv
			if got, want := shardFor(heap.ChannelID, q.NumShards()), shard; got != want {
				t.Fatalf("channel %d arrived on shard %d, want %d (its shardFor result)", heap.ChannelID, shard, want)
			}
		}
	}
}

func TestShardedHeapQueue_DifferentChannelsSpreadAcrossShards(t *testing.T) {
	q := NewShardedHeapQueue(4, 16)

	for ch := 0; ch < 8; ch++ {
		q.Send(&ChannelHeap{ChannelID: ch})
	}

	used := map[int]bool{}
	for shard := 0; shard < q.NumShards(); shard++ {
		if len(q.Recv(shard)) > 0 {
			used[shard] = true
		}
	}
	if len(used) < 2 {
		t.Fatalf("8 distinct channels across 4 shards landed on only %d shard(s), want spread across more than one", len(used))
	}
}

func TestShardedHeapQueue_LenSumsAllShards(t *testing.T) {
	q := NewShardedHeapQueue(3, 16)
	for ch := 0; ch < 6; ch++ {
		if !q.Send(&ChannelHeap{ChannelID: ch}) {
			t.Fatalf("channel %d: unexpectedly dropped", ch)
		}
	}
	if got, want := q.Len(), 6; got != want {
		t.Fatalf("Len() = %d, want %d", got, want)
	}
}

func TestShardedHeapQueue_FullShardOnlyDropsItsOwnChannels(t *testing.T) {
	q := NewShardedHeapQueue(2, 2)

	// Fill shard 0's queue (capacity 2) with channel 0's heaps (channel 0
	// hashes to shard 0 -- see shardFor).
	if !q.Send(&ChannelHeap{ChannelID: 0}) {
		t.Fatal("first send for channel 0 unexpectedly dropped")
	}
	if !q.Send(&ChannelHeap{ChannelID: 0}) {
		t.Fatal("second send for channel 0 unexpectedly dropped")
	}
	if q.Send(&ChannelHeap{ChannelID: 0}) {
		t.Fatal("third send for channel 0 should have been dropped (shard 0 full)")
	}

	// Channel 1 hashes to a different shard (shard 1), so it must still
	// succeed even though shard 0 is full -- this is the isolation
	// sharding buys over one shared queue, where a burst on one channel
	// could previously starve every other channel's sends.
	if !q.Send(&ChannelHeap{ChannelID: 1}) {
		t.Fatal("send for channel 1 unexpectedly dropped by an unrelated full shard")
	}
}

func TestShardedHeapQueue_NLessThanOneTreatedAsOne(t *testing.T) {
	q := NewShardedHeapQueue(0, 4)
	if got, want := q.NumShards(), 1; got != want {
		t.Fatalf("NumShards() = %d, want %d", got, want)
	}
}
