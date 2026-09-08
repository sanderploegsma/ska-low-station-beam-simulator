// Package spead hand-rolls a minimal SPEAD-64-48 heap encoder for the
// SPS-CBF ICD wire format — ported from the Python project's spead.py.
//
// Deliberately NOT using a general-purpose SPEAD library: the Python
// side found that spead2's own packet encoder unconditionally writes 4
// reserved item pointers on every packet it emits, with no way to
// suppress any of them (checked directly against spead2==4.4.1's C++
// source) — but CBF's real ICD heap has exactly 6 items total, fewer
// than that mandatory minimum, so spead2 cannot produce a compliant
// packet at all. A firmware receiver that addresses payload bytes
// directly at a fixed offset (rather than doing a generic SPEAD parse)
// has no way to skip items it doesn't expect. This package is scoped to
// exactly those 6 items and nothing else — see SpsPacketizer's doc
// comment for the item layout.
package spead

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync/atomic"

	"github.com/skao/station-beam-simulator-go/internal/common"
)

const (
	SpeadVersion         = 4
	SpeadItemPointerBits = 64
	SpeadHeapAddressBits = 48 // SPEAD-64-48, confirmed against the real ICD
	speadIDBits          = SpeadItemPointerBits - 1 - SpeadHeapAddressBits
	speadHeapCounterBits = 40 // top 8 bits of the 0x0001 item are reserved
)

const (
	speadIDMask          = (1 << speadIDBits) - 1
	speadValueMask       = (uint64(1) << SpeadHeapAddressBits) - 1
	speadHeapCounterMask = (uint64(1) << speadHeapCounterBits) - 1
)

// PayloadLengthBytes: HeapLen * 4 bytes/sample, fixed by the ICD.
const PayloadLengthBytes = 0x2000

// numHeapItems: the ICD's fixed 6-item heap (see SpsPacketizer's doc
// comment for the layout) -- never varies, so the wire buffer's total
// size is a compile-time constant, not something EncodeChannelHeap needs
// to compute per call.
const (
	numHeapItems      = 6
	speadHeaderSize   = 8
	itemPointerBytes  = 8
	heapPayloadOffset = speadHeaderSize + numHeapItems*itemPointerBytes // 56
	heapWireSizeBytes = heapPayloadOffset + PayloadLengthBytes          // 8248
)

// packChannelInfo builds the 0x3000 item value: 16 bits reserved | 16
// bits beam_id | 16 bits frequency_id.
func packChannelInfo(beamID, frequencyID uint32) uint64 {
	return (uint64(beamID&0xFFFF) << 16) | uint64(frequencyID&0xFFFF)
}

// packAntennaInfo builds the 0x3001 item value: 8 bits substation_id | 8
// bits subarray_id | 16 bits station_id | 16 bits reserved.
func packAntennaInfo(substationID, subarrayID uint8, stationID uint16) uint64 {
	return (uint64(substationID) << 40) | (uint64(subarrayID) << 32) | (uint64(stationID) << 16)
}

// quantizeComponent rounds v to the nearest int8 (half away from zero,
// matching math.Round's convention) and clamps to [-128, 127], in one
// branch-light pass. Replaces a separate math.Round + clampToInt8 call
// pair: profiling a real end-to-end run at 384 channels found math.Round
// alone at ~8% of ALL CPU time on this hot path (quantize8bitIntoPayload
// runs it twice per sample, ~1.57M samples/tick at 384 channels) --
// disproportionate for what it does here, since math.Round's
// implementation spends real work on NaN/Inf/magnitude-≥2^52 cases that
// can never occur for a synthesized sample scaled into roughly [-127,
// 127]. v+copysign(0.5, v) then truncating (via the int8 conversion,
// which truncates toward zero) gives the identical round-half-away-from-
// zero result for every value in that range, without math.Round's
// call/branch overhead.
func quantizeComponent(v float64) int8 {
	v += math.Copysign(0.5, v)
	if v > 127 {
		return 127
	}
	if v < -128 {
		return -128
	}
	return int8(v)
}

// clampToInt8 clamps an already-rounded value to [-128, 127]. Kept
// separate from quantizeComponent for quantize8bit/tests, which want the
// clamp alone against a value someone else already rounded.
func clampToInt8(v float64) int8 {
	if v > 127 {
		v = 127
	} else if v < -128 {
		v = -128
	}
	return int8(v)
}

// quantize8bitScale computes the shared per-call scale factor quantize8bit
// and quantize8bitIntoPayload both use. Tracks the max SQUARED magnitude
// across the loop and takes a single math.Sqrt at the end, not one per
// sample: sqrt is monotonic for non-negative inputs, so
// sqrt(re²+im²) > maxAbs is equivalent to re²+im² > maxAbs² without ever
// computing the intermediate sqrt. Profiling found this loop's own
// math.Sqrt call (one per sample, e.g. ~1.57M/tick at 384 channels) at
// ~10% of ALL CPU time -- this eliminates all but one sqrt call per
// channel/pol/tick (2047 of 2048 at HeapLen=2048), leaving the loop as a
// plain multiply-add-compare. math.Sqrt, not cmplx.Abs (=math.Hypot):
// Hypot's overflow/underflow-safe scaling is unneeded here (synthesized
// sample magnitudes are always small and finite, nowhere near float64's
// under/overflow range), so the plain, faster form is used for the final
// sqrt too.
func quantize8bitScale(samples []complex64) float64 {
	maxSq := 0.0
	for _, s := range samples {
		re, im := float64(real(s)), float64(imag(s))
		if sq := re*re + im*im; sq > maxSq {
			maxSq = sq
		}
	}
	return 127.0 / (math.Sqrt(maxSq) + 1e-12)
}

// quantize8bit does per-call independent scaling to int8, matching
// Python's quantize_8bit. Allocates a fresh pair of slices every call --
// fine for tests and other one-off callers, but NOT used by
// EncodeChannelHeap's hot path (see quantize8bitIntoPayload).
func quantize8bit(samples []complex64) (outReal, outImag []int8) {
	scale := quantize8bitScale(samples)
	outReal = make([]int8, len(samples))
	outImag = make([]int8, len(samples))
	for i, s := range samples {
		outReal[i] = quantizeComponent(float64(real(s)) * scale)
		outImag[i] = quantizeComponent(float64(imag(s)) * scale)
	}
	return outReal, outImag
}

// quantize8bitIntoPayload quantizes samples directly into dst at
// dst[i*stride+realOffset] (real component) / dst[i*stride+realOffset+1]
// (imag component) for each sample i -- no intermediate []int8
// allocation at all, unlike quantize8bit. dst is the heap's own final
// wire buffer (see EncodeChannelHeap): profiling a real end-to-end run
// at 384 channels found EncodeChannelHeap consuming roughly a third of
// all CPU time, dominated by per-heap allocation/zeroing overhead (~12
// small allocations/heap previously: 4 quantize8bit slices, 1 payload
// slice, 7 more from speadHeaderBytes/speadItemPointer each returning
// their own []byte) at the required per-tick heap rate (e.g. ~174k
// heaps/sec at 384 channels) -- competing for CPU with the producer
// goroutine badly enough to explain "falling behind pacing" even though
// the producer's OWN cost, measured in isolation, was well within
// budget. Fixed by writing every piece of a heap directly into its
// (exactly-once-allocated) wire buffer instead: this function for the
// payload, and EncodeChannelHeap's header/item-pointer writes below.
func quantize8bitIntoPayload(samples []complex64, dst []byte, realOffset, stride int, scale float64) {
	for i, s := range samples {
		dst[i*stride+realOffset] = byte(quantizeComponent(float64(real(s)) * scale))
		dst[i*stride+realOffset+1] = byte(quantizeComponent(float64(imag(s)) * scale))
	}
}

// writeSpeadHeader writes the 8-byte SPEAD packet header into dst[:8]:
// magic (0x53)+version(0x04), the item-ID field width and heap-address
// field width (fixed at SPEAD-64-48's 2 and 6 bytes), then the
// item-pointer count. Layout matches the real SPEAD wire format (checked
// against spead2's own source on the Python side); what differs from
// spead2 is only WHICH items follow this header.
func writeSpeadHeader(dst []byte, nItems int) {
	const heapAddressBytes = uint64(SpeadHeapAddressBits / 8)
	const idBytes = uint64(SpeadItemPointerBits/8) - heapAddressBytes
	word := (uint64(0x5300|SpeadVersion) << 48) | (idBytes << 40) | (heapAddressBytes << 32) | uint64(nItems)
	binary.BigEndian.PutUint64(dst, word)
}

// writeSpeadItemPointer writes one IMMEDIATE SPEAD-64-48 item pointer
// (mode bit set, value embedded directly in the pointer's low 48 bits)
// into dst[:8]. CBF's 6-item heap never needs an ADDRESS-mode pointer.
func writeSpeadItemPointer(dst []byte, itemID uint16, value uint64) error {
	if uint64(itemID) > speadIDMask {
		return fmt.Errorf("item id %#x does not fit in %d bits", itemID, speadIDBits)
	}
	if value > speadValueMask {
		return fmt.Errorf("value %#x for item %#x does not fit in %d bits", value, itemID, SpeadHeapAddressBits)
	}
	pointer := (uint64(1) << 63) | (uint64(itemID) << SpeadHeapAddressBits) | value
	binary.BigEndian.PutUint64(dst, pointer)
	return nil
}

// Sender is anything the encoded heap bytes can be written to — a
// connected UDP net.Conn in production (net.Conn implements io.Writer),
// or a fake in tests. Kept minimal (just Write) so SpsPacketizer imposes
// no networking-specific requirement of its own.
type Sender interface {
	Write(p []byte) (n int, err error)
}

// SpsPacketizer encodes (and optionally sends) one common.ChannelHeap
// per SPS-CBF ICD, via the hand-rolled SPEAD-64-48 encoder above.
//
// ITEM LAYOUT (six items total, confirmed against the real ICD on the
// Python side):
//
//	0x0001  8 bits reserved | 40 bits heap_counter
//	0x0004  48 bits packet_payload_length (fixed: 0x2000)
//	0x3010  48 bits scan_id
//	0x3000  16 bits reserved | 16 bits beam_id | 16 bits frequency_id
//	0x3001  8 bits substation_id | 8 bits subarray_id | 16 bits
//	        station_id | 16 bits reserved
//	0x3300  48 bits payload_offset (fixed: 0x0 -- heaps are always
//	        exactly one packet, never fragmented)
//
// ...followed immediately by the 8192-byte interleaved V/H I/Q payload.
// There is no 7th "payload" item pointer — CBF firmware addresses the
// payload bytes directly at a fixed offset after the 6 item pointers.
type SpsPacketizer struct {
	station *common.StationConfig
	sender  Sender // nil: EncodeChannelHeap still works, SendChannelHeap does not

	// quantizeScaleBits: math.Float64bits of a FIXED quantization scale
	// (see SetQuantizeScale), 0 (its zero value) meaning "none set --
	// fall back to the original per-heap ADAPTIVE scale" (quantize8bit's
	// own quantize8bitScale scan). atomic, not a plain float64: a
	// SpsPacketizer is shared read-only across every BatchSendLoop
	// goroutine in a SenderPool (see NewSenderPool's doc comment), but
	// this ONE field is the exception -- callers with a fixed noise/
	// tone config known up front (synth.DirectSynthesisStreamer.
	// QuantizeScale) set it once before a scan starts; the gRPC-served
	// path's SenderPool outlives many scans with potentially different
	// noise/tone configs (see server.Server.Start's doc comment: the
	// pool is created once at process Start(), before any scan's config
	// is known), so StartScan must be able to update it later, safely,
	// while sender goroutines are already running against the previous
	// scan's heaps draining out of the queue.
	quantizeScaleBits atomic.Uint64
}

// NewSpsPacketizer constructs a packetizer. sender may be nil if only
// EncodeChannelHeap (not SendChannelHeap) will be used. Quantization
// defaults to the original adaptive per-heap scale until/unless
// SetQuantizeScale is called.
func NewSpsPacketizer(station *common.StationConfig, sender Sender) *SpsPacketizer {
	return &SpsPacketizer{station: station, sender: sender}
}

// SetQuantizeScale sets a FIXED per-sample quantization scale, replacing
// the default adaptive behavior (quantize8bitScale re-scanning every
// heap's actual samples for their max magnitude, every tick). Pass 0 to
// go back to adaptive. See synth.DirectSynthesisStreamer.QuantizeScale
// for how a safe fixed scale is derived from a streamer's noise/tone
// config -- this method only stores whatever value it's given, with no
// opinion of its own about where it came from.
//
// Safe to call concurrently with EncodeChannelHeapInto (atomic store) --
// required, not just convenient, since BatchSendLoop goroutines may
// already be running against a SenderPool's shared packetizer by the
// time a new scan's StartScan call wants to update this.
func (p *SpsPacketizer) SetQuantizeScale(scale float64) {
	p.quantizeScaleBits.Store(math.Float64bits(scale))
}

// resolveQuantizeScale returns the fixed scale if one is set (SKIPPING
// samples entirely -- no per-heap scan, the whole point: see
// quantize8bitScale's doc comment for the cost this avoids), otherwise
// falls back to scanning samples adaptively, preserving this package's
// original per-heap-optimal-scale behavior for any caller that hasn't
// opted into a fixed scale.
func (p *SpsPacketizer) resolveQuantizeScale(samples []complex64) float64 {
	if bits := p.quantizeScaleBits.Load(); bits != 0 {
		return math.Float64frombits(bits)
	}
	return quantize8bitScale(samples)
}

// EncodeChannelHeap builds the raw SPEAD-64-48 heap bytes for one
// channel — the full on-wire payload of one UDP packet. Allocates a
// fresh, exactly-once-per-call buffer; BatchSendLoop's hot path instead
// calls EncodeChannelHeapInto against a reused buffer (see that
// function's doc comment for why the extra allocation here matters at
// the real per-tick heap rate) -- this wrapper exists for
// SendChannelHeap and other one-off callers where reuse doesn't apply.
func (p *SpsPacketizer) EncodeChannelHeap(heap *common.ChannelHeap) ([]byte, error) {
	buf := make([]byte, heapWireSizeBytes)
	if err := p.EncodeChannelHeapInto(buf, heap); err != nil {
		return nil, err
	}
	return buf, nil
}

// EncodeChannelHeapInto builds the raw SPEAD-64-48 heap bytes for one
// channel directly into dst (which must be exactly heapWireSizeBytes
// long) — no allocation at all. Everything (header, item pointers,
// quantized V/H payload) is written directly into dst at fixed offsets,
// never through an intermediate slice.
//
// Profiling a real end-to-end run on the actual Linux/SR-IOV target (not
// just this dev machine) showed EncodeChannelHeap still dominating total
// CPU (~46-52%) even after removing the ~11 OTHER per-heap allocations
// (see quantize8bitIntoPayload's doc comment) -- the one remaining
// allocation (the returned buf itself) was still costing real
// mallocgc/memclr time at the required heap rate (~1.4GB/s of allocation
// traffic at 384 channels: ~174k heaps/sec * 8248 bytes/heap). Fixed by
// letting BatchSendLoop supply a REUSED buffer from a small pool instead
// -- safe because a UDP write (sendmmsg included) copies the buffer's
// contents into the kernel synchronously before returning; once
// BatchSender.WriteBatch returns, every buffer in that batch is free to
// reuse for the next one, so an encode-into-a-pooled-buffer + send +
// reuse cycle never races the actual wire send.
func (p *SpsPacketizer) EncodeChannelHeapInto(dst []byte, heap *common.ChannelHeap) error {
	if len(dst) != heapWireSizeBytes {
		return fmt.Errorf("dst must be exactly %d bytes, got %d", heapWireSizeBytes, len(dst))
	}
	if len(heap.VSamples) != PayloadLengthBytes/4 || len(heap.HSamples) != PayloadLengthBytes/4 {
		return fmt.Errorf("heap ch=%d: VSamples/HSamples must have length %d, got v=%d h=%d", heap.ChannelID, PayloadLengthBytes/4, len(heap.VSamples), len(heap.HSamples))
	}

	// "packet count since SKA epoch" -- count of HeapLen-sample BLOCKS
	// (BlockDurationS each) since epoch, NOT a count of individual
	// samples. The Python codebase hit a real bug here once (multiplying
	// by the sample rate instead of dividing by the block duration,
	// inflating this by HeapLen) -- see CLAUDE.md bug #17. This divides,
	// matching the fixed formula.
	heapCounterF := math.Round(common.UnixToTAI2000Seconds(heap.HeapStartTime) / common.BlockDurationS)
	if heapCounterF < 0 || heapCounterF > float64(speadHeapCounterMask) {
		return fmt.Errorf("heap_counter %v does not fit in the ICD's %d-bit field (top 8 bits of the 0x0001 item are reserved)", heapCounterF, speadHeapCounterBits)
	}
	heapCounter := uint64(heapCounterF)

	type item struct {
		id    uint16
		value uint64
	}
	items := [numHeapItems]item{
		{0x0001, heapCounter},
		{0x0004, PayloadLengthBytes},
		{0x3010, uint64(p.station.ScanID)},
		{0x3000, packChannelInfo(uint32(p.station.BeamID), uint32(common.ChannelStart+heap.ChannelID))},
		{0x3001, packAntennaInfo(uint8(p.station.SubstationID), uint8(p.station.SubarrayID), uint16(p.station.StationID))},
		{0x3300, 0x0},
	}

	writeSpeadHeader(dst[:speadHeaderSize], numHeapItems)
	offset := speadHeaderSize
	for _, it := range items {
		if err := writeSpeadItemPointer(dst[offset:offset+itemPointerBytes], it.id, it.value); err != nil {
			return err
		}
		offset += itemPointerBytes
	}

	payload := dst[heapPayloadOffset:]
	quantize8bitIntoPayload(heap.VSamples, payload, 0, 4, p.resolveQuantizeScale(heap.VSamples)) // Vreal at +0, Vimag at +1 of each 4-byte sample
	quantize8bitIntoPayload(heap.HSamples, payload, 2, 4, p.resolveQuantizeScale(heap.HSamples)) // Hreal at +2, Himag at +3
	return nil
}

// SendChannelHeap encodes heap and writes it to this packetizer's
// sender.
func (p *SpsPacketizer) SendChannelHeap(heap *common.ChannelHeap) error {
	if p.sender == nil {
		return fmt.Errorf("SpsPacketizer has no sender -- construct with NewSpsPacketizer(station, sender) to use SendChannelHeap")
	}
	encoded, err := p.EncodeChannelHeap(heap)
	if err != nil {
		return err
	}
	_, err = p.sender.Write(encoded)
	return err
}
