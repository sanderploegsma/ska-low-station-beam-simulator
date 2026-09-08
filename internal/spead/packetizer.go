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

// QuantizeComponent rounds v to the nearest int8 (half away from zero,
// matching math.Round's convention) and clamps to [-128, 127], in one
// branch-light pass, instead of a separate math.Round + clampToInt8 call
// pair: math.Round's implementation spends real work on NaN/Inf/
// magnitude-≥2^52 cases that can never occur for a synthesized sample
// scaled into roughly [-127, 127]. v+copysign(0.5, v) then truncating
// (via the int8 conversion, which truncates toward zero) gives the
// identical round-half-away-from-zero result for every value in that
// range, without math.Round's call/branch overhead.
func QuantizeComponent(v float64) int8 {
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
// separate from QuantizeComponent for quantize8bit/tests, which want the
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
// computing the intermediate sqrt, leaving the loop as a plain
// multiply-add-compare with only one sqrt call per channel/pol/tick.
// math.Sqrt, not cmplx.Abs (=math.Hypot): Hypot's overflow/underflow-safe
// scaling is unneeded here (synthesized sample magnitudes are always
// small and finite, nowhere near float64's under/overflow range), so the
// plain, faster form is used for the final sqrt too.
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
		outReal[i] = QuantizeComponent(float64(real(s)) * scale)
		outImag[i] = QuantizeComponent(float64(imag(s)) * scale)
	}
	return outReal, outImag
}

// quantize8bitIntoPayload quantizes samples directly into dst at
// dst[i*stride+realOffset] (real component) / dst[i*stride+realOffset+1]
// (imag component) for each sample i -- no intermediate []int8
// allocation at all, unlike quantize8bit. dst is the heap's own final
// wire buffer (see EncodeChannelHeap): every piece of a heap (header,
// item pointers, and this payload) is written directly into its
// exactly-once-allocated wire buffer, rather than building each piece as
// its own separately-allocated slice first.
func quantize8bitIntoPayload(samples []complex64, dst []byte, realOffset, stride int, scale float64) {
	for i, s := range samples {
		dst[i*stride+realOffset] = byte(QuantizeComponent(float64(real(s)) * scale))
		dst[i*stride+realOffset+1] = byte(QuantizeComponent(float64(imag(s)) * scale))
	}
}

// copyQuantizedIntoPayload is quantize8bitIntoPayload's PRE-QUANTIZED
// counterpart: quantized holds int8 (real,imag) pairs ALREADY computed
// (quantized[i*2]/quantized[i*2+1] for sample i) -- typically once, at
// noise-tile-bank construction time (see
// synth.DirectSynthesisStreamer.GenerateQuantizedHeaps), not per heap.
// No scale, no rounding, no clamping here -- just moving bytes that were
// already computed, which is the entire point: this is what
// ChannelHeap.VQuantized/HQuantized exist to let EncodeChannelHeapInto
// skip.
//
// Used only as a FALLBACK for a pol whose OTHER pol isn't also
// pre-quantized (not the case for any real DirectSynthesisStreamer
// channel today -- see ChannelHeap's doc comment -- but ChannelHeap
// allows it per-pol independently, so this stays correct for that case).
// EncodeChannelHeapInto's common case is copyQuantizedVHIntoPayload
// below instead, since this function's strided two-bytes-out-of-four
// write pattern (each of V's and H's passes only half-writes every
// 4-byte block in dst, needing its own read-for-ownership of that cache
// line -- twice per block, once per pol -- instead of one full write) is
// more expensive than writing both pols' bytes in one pass.
func copyQuantizedIntoPayload(quantized []byte, dst []byte, realOffset, stride int) {
	for i := 0; i < len(quantized)/2; i++ {
		dst[i*stride+realOffset] = quantized[i*2]
		dst[i*stride+realOffset+1] = quantized[i*2+1]
	}
}

// copyQuantizedVHIntoPayload is EncodeChannelHeapInto's common-case fast
// path: writes BOTH pols' pre-quantized bytes for each sample in ONE
// pass (all 4 bytes of dst's per-sample block: Vreal, Vimag, Hreal,
// Himag), instead of copyQuantizedIntoPayload's two separate
// half-writing passes. Halves how many times each of dst's cache lines
// needs touching, and every 4-byte block gets fully populated by a
// single sequential write instead of two interleaved partial ones.
func copyQuantizedVHIntoPayload(vQuantized, hQuantized, dst []byte) {
	n := len(vQuantized) / 2
	for i := 0; i < n; i++ {
		j := i * 4
		dst[j] = vQuantized[i*2]
		dst[j+1] = vQuantized[i*2+1]
		dst[j+2] = hQuantized[i*2]
		dst[j+3] = hQuantized[i*2+1]
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

// writeSpeadItemPointer writes one SPEAD-64-48 item pointer into dst[:8].
// immediate selects IMMEDIATE mode (mode bit set, value embedded directly
// in the pointer's low 48 bits) vs. ADDRESS mode (mode bit clear, value
// interpreted as a byte offset into the heap's payload area). Every item
// in CBF's 6-item heap is IMMEDIATE except 0x3300 payload_offset, which
// is the one item that addresses the payload rather than carrying a
// scalar value of its own -- see EncodeChannelHeapInto's item table and
// docs/history.md for how this was found (an earlier version of this
// function hardcoded every item as IMMEDIATE, including 0x3300, which a
// real CNIC reference capture's SPEAD traffic contradicted).
func writeSpeadItemPointer(dst []byte, itemID uint16, value uint64, immediate bool) error {
	if uint64(itemID) > speadIDMask {
		return fmt.Errorf("item id %#x does not fit in %d bits", itemID, speadIDBits)
	}
	if value > speadValueMask {
		return fmt.Errorf("value %#x for item %#x does not fit in %d bits", value, itemID, SpeadHeapAddressBits)
	}
	var modeBit uint64
	if immediate {
		modeBit = 1
	}
	pointer := (modeBit << 63) | (uint64(itemID) << SpeadHeapAddressBits) | value
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
// Python side, and against a real CNIC reference capture -- see
// docs/history.md):
//
//	0x0001  IMMEDIATE  8 bits reserved | 40 bits heap_counter
//	0x0004  IMMEDIATE  48 bits packet_payload_length (fixed: 0x2000)
//	0x3010  IMMEDIATE  48 bits scan_id
//	0x3000  IMMEDIATE  16 bits reserved | 16 bits beam_id | 16 bits frequency_id
//	0x3001  IMMEDIATE  8 bits substation_id | 8 bits subarray_id | 16 bits
//	                   station_id | 16 bits reserved
//	0x3300  ADDRESS    48-bit byte offset of the payload, always 0x0 -- the
//	                   only non-immediate item, and the payload always
//	                   starts immediately after the last item pointer
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
// BatchSendLoop supplies a REUSED buffer from a small pool rather than
// letting this allocate a fresh one per heap -- safe because a UDP write
// (sendmmsg included) copies the buffer's contents into the kernel
// synchronously before returning; once BatchSender.WriteBatch returns,
// every buffer in that batch is free to reuse for the next one, so an
// encode-into-a-pooled-buffer + send + reuse cycle never races the
// actual wire send.
func (p *SpsPacketizer) EncodeChannelHeapInto(dst []byte, heap *common.ChannelHeap) error {
	if len(dst) != heapWireSizeBytes {
		return fmt.Errorf("dst must be exactly %d bytes, got %d", heapWireSizeBytes, len(dst))
	}
	// Each pol independently uses EITHER the complex path (VSamples/
	// HSamples, quantized here) OR the pre-quantized path (VQuantized/
	// HQuantized, already quantized -- see ChannelHeap's doc comment).
	// vLen/hLen normalize both to a sample count for one length check
	// covering either representation.
	vLen, hLen := len(heap.VSamples), len(heap.HSamples)
	if heap.VQuantized != nil {
		vLen = len(heap.VQuantized) / 2
	}
	if heap.HQuantized != nil {
		hLen = len(heap.HQuantized) / 2
	}
	if vLen != PayloadLengthBytes/4 || hLen != PayloadLengthBytes/4 {
		return fmt.Errorf("heap ch=%d: V/H sample count must be %d, got v=%d h=%d", heap.ChannelID, PayloadLengthBytes/4, vLen, hLen)
	}

	// "packet count since SKA epoch" -- count of HeapLen-sample BLOCKS
	// (BlockDurationS each) since epoch, NOT a count of individual
	// samples.
	heapCounterF := math.Round(common.UnixToTAI2000Seconds(heap.HeapStartTime) / common.BlockDurationS)
	if heapCounterF < 0 || heapCounterF > float64(speadHeapCounterMask) {
		return fmt.Errorf("heap_counter %v does not fit in the ICD's %d-bit field (top 8 bits of the 0x0001 item are reserved)", heapCounterF, speadHeapCounterBits)
	}
	heapCounter := uint64(heapCounterF)

	type item struct {
		id        uint16
		value     uint64
		immediate bool
	}
	items := [numHeapItems]item{
		{0x0001, heapCounter, true},
		{0x0004, PayloadLengthBytes, true},
		{0x3010, uint64(p.station.ScanID), true},
		{0x3000, packChannelInfo(uint32(p.station.BeamID), uint32(common.ChannelStart+heap.ChannelID)), true},
		{0x3001, packAntennaInfo(uint8(p.station.SubstationID), uint8(p.station.SubarrayID), uint16(p.station.StationID)), true},
		// ADDRESS mode, not immediate -- see this type's ITEM LAYOUT doc
		// comment: this is the one item that addresses the payload rather
		// than carrying a scalar value, and the payload always starts
		// immediately after the last item pointer (offset 0).
		{0x3300, 0x0, false},
	}

	writeSpeadHeader(dst[:speadHeaderSize], numHeapItems)
	offset := speadHeaderSize
	for _, it := range items {
		if err := writeSpeadItemPointer(dst[offset:offset+itemPointerBytes], it.id, it.value, it.immediate); err != nil {
			return err
		}
		offset += itemPointerBytes
	}

	payload := dst[heapPayloadOffset:]
	switch {
	case heap.VQuantized != nil && heap.HQuantized != nil:
		// The common case in production: every noise-only channel sets
		// both (see synth.DirectSynthesisStreamer.GenerateQuantizedHeaps).
		// One combined pass, not two -- see copyQuantizedVHIntoPayload's
		// doc comment for why that matters.
		copyQuantizedVHIntoPayload(heap.VQuantized, heap.HQuantized, payload)
	default:
		if heap.VQuantized != nil { // Vreal at +0, Vimag at +1 of each 4-byte sample
			copyQuantizedIntoPayload(heap.VQuantized, payload, 0, 4)
		} else {
			quantize8bitIntoPayload(heap.VSamples, payload, 0, 4, p.resolveQuantizeScale(heap.VSamples))
		}
		if heap.HQuantized != nil { // Hreal at +2, Himag at +3
			copyQuantizedIntoPayload(heap.HQuantized, payload, 2, 4)
		} else {
			quantize8bitIntoPayload(heap.HSamples, payload, 2, 4, p.resolveQuantizeScale(heap.HSamples))
		}
	}
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
