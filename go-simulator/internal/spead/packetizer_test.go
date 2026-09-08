package spead

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/skao/station-beam-simulator-go/internal/common"
)

// decodedItem is one parsed SPEAD-64-48 item pointer.
type decodedItem struct {
	id    uint16
	value uint64
}

// decodeHeap parses raw SPEAD-64-48 bytes back into (itemID, value)
// pairs plus the trailing payload, entirely independent of this
// package's own encoder logic -- the same "parse the bytes back, don't
// just trust the encoder" check the Python test suite uses.
func decodeHeap(t *testing.T, raw []byte) (items []decodedItem, payload []byte) {
	t.Helper()
	if len(raw) < 8 {
		t.Fatalf("heap too short for a SPEAD header: %d bytes", len(raw))
	}
	header := binary.BigEndian.Uint64(raw[:8])
	magicVersion := header >> 48
	if magicVersion != 0x5300|SpeadVersion {
		t.Fatalf("bad magic/version: got %#x", magicVersion)
	}
	idBytes := (header >> 40) & 0xFF
	heapAddressBytes := (header >> 32) & 0xFF
	nItems := int(header & 0xFFFFFFFF)
	if idBytes != 2 || heapAddressBytes != 6 {
		t.Fatalf("unexpected field widths: id_bytes=%d heap_address_bytes=%d", idBytes, heapAddressBytes)
	}

	offset := 8
	for i := 0; i < nItems; i++ {
		if offset+8 > len(raw) {
			t.Fatalf("heap truncated while reading item pointer %d", i)
		}
		ptr := binary.BigEndian.Uint64(raw[offset : offset+8])
		offset += 8
		mode := ptr >> 63
		if mode != 1 {
			t.Fatalf("item pointer %d is not IMMEDIATE mode", i)
		}
		id := uint16((ptr >> 48) & 0x7FFF)
		value := ptr & ((uint64(1) << 48) - 1)
		items = append(items, decodedItem{id: id, value: value})
	}
	payload = raw[offset:]
	return items, payload
}

func findItem(items []decodedItem, id uint16) (uint64, bool) {
	for _, it := range items {
		if it.id == id {
			return it.value, true
		}
	}
	return 0, false
}

func testHeap(channelID int, heapStartTime float64) *common.ChannelHeap {
	v := make([]complex64, common.HeapLen)
	h := make([]complex64, common.HeapLen)
	for i := range v {
		v[i] = complex64(complex(float64(i%128)-64, float64((i+10)%128)-64))
		h[i] = complex64(complex(float64((i+30)%128)-64, float64((i+60)%128)-64))
	}
	return &common.ChannelHeap{ChannelID: channelID, VSamples: v, HSamples: h, HeapStartTime: heapStartTime}
}

func TestEncodeChannelHeap_ItemLayoutMatchesICD(t *testing.T) {
	station := &common.StationConfig{StationID: 7, SubstationID: 2, SubarrayID: 3, BeamID: 5, ScanID: 99}
	p := NewSpsPacketizer(station, nil)

	heap := testHeap(12, 1_700_000_000.0)
	raw, err := p.EncodeChannelHeap(heap)
	if err != nil {
		t.Fatalf("EncodeChannelHeap: %v", err)
	}

	items, payload := decodeHeap(t, raw)
	if len(items) != 6 {
		t.Fatalf("got %d items, want exactly 6 (per the ICD -- no 7th 'payload' item)", len(items))
	}
	if len(payload) != PayloadLengthBytes {
		t.Fatalf("payload length = %d, want %d", len(payload), PayloadLengthBytes)
	}

	if v, ok := findItem(items, 0x0004); !ok || v != PayloadLengthBytes {
		t.Fatalf("0x0004 packet_payload_length = %v (ok=%v), want %d", v, ok, PayloadLengthBytes)
	}
	if v, ok := findItem(items, 0x3010); !ok || v != uint64(station.ScanID) {
		t.Fatalf("0x3010 scan_id = %v (ok=%v), want %d", v, ok, station.ScanID)
	}
	if v, ok := findItem(items, 0x3300); !ok || v != 0 {
		t.Fatalf("0x3300 payload_offset = %v (ok=%v), want 0", v, ok)
	}
	channelInfo, ok := findItem(items, 0x3000)
	if !ok {
		t.Fatal("missing 0x3000 channel_info item")
	}
	gotBeamID := (channelInfo >> 16) & 0xFFFF
	gotFreqID := channelInfo & 0xFFFF
	if gotBeamID != uint64(station.BeamID) {
		t.Fatalf("channel_info beam_id = %d, want %d", gotBeamID, station.BeamID)
	}
	if wantFreqID := uint64(common.ChannelStart + heap.ChannelID); gotFreqID != wantFreqID {
		t.Fatalf("channel_info frequency_id = %d, want %d (ChannelStart+ChannelID)", gotFreqID, wantFreqID)
	}
	antennaInfo, ok := findItem(items, 0x3001)
	if !ok {
		t.Fatal("missing 0x3001 antenna_info item")
	}
	gotSubstation := (antennaInfo >> 40) & 0xFF
	gotSubarray := (antennaInfo >> 32) & 0xFF
	gotStation := (antennaInfo >> 16) & 0xFFFF
	if gotSubstation != uint64(station.SubstationID) || gotSubarray != uint64(station.SubarrayID) || gotStation != uint64(station.StationID) {
		t.Fatalf("antenna_info fields = (substation=%d subarray=%d station=%d), want (%d %d %d)",
			gotSubstation, gotSubarray, gotStation, station.SubstationID, station.SubarrayID, station.StationID)
	}
	if _, ok := findItem(items, 0x0001); !ok {
		t.Fatal("missing 0x0001 heap_counter item")
	}
}

func TestEncodeChannelHeap_PayloadInterleavesVHRealImag(t *testing.T) {
	station := &common.StationConfig{StationID: 1}
	p := NewSpsPacketizer(station, nil)

	// Build a heap whose samples are already exactly representable as
	// int8 after quantize8bit's scaling, so we can predict the exact
	// interleaved bytes rather than just checking structure. Every V
	// sample is 3+4i (magnitude 5) and every H sample is 0+5i (magnitude
	// 5) -- quantize8bit scales V and H INDEPENDENTLY, each by
	// 127/max(|samples|), so both scale by 127/5=25.4 here.
	v := make([]complex64, common.HeapLen)
	h := make([]complex64, common.HeapLen)
	for i := range v {
		v[i] = complex(3.0, 4.0)
		h[i] = complex(0.0, 5.0)
	}
	heap := &common.ChannelHeap{ChannelID: 0, VSamples: v, HSamples: h, HeapStartTime: 1_700_000_000.0}

	raw, err := p.EncodeChannelHeap(heap)
	if err != nil {
		t.Fatalf("EncodeChannelHeap: %v", err)
	}
	_, payload := decodeHeap(t, raw)

	wantVReal := int8(76)  // round(3 * 127/5) = round(76.2)
	wantVImag := int8(102) // round(4 * 127/5) = round(101.6)
	wantHReal := int8(0)   // round(0 * 127/5)
	wantHImag := int8(127) // round(5 * 127/5) = 127 exactly

	if got := int8(payload[0]); got != wantVReal {
		t.Fatalf("payload[0] (Vreal) = %d, want %d", got, wantVReal)
	}
	if got := int8(payload[1]); got != wantVImag {
		t.Fatalf("payload[1] (Vimag) = %d, want %d", got, wantVImag)
	}
	if got := int8(payload[2]); got != wantHReal {
		t.Fatalf("payload[2] (Hreal) = %d, want %d", got, wantHReal)
	}
	if got := int8(payload[3]); got != wantHImag {
		t.Fatalf("payload[3] (Himag) = %d, want %d", got, wantHImag)
	}
}

// TestCopyQuantizedVHIntoPayload_MatchesTwoIndependentPasses is the
// correctness proof behind EncodeChannelHeapInto's combined fast path
// (copyQuantizedVHIntoPayload): given the SAME V/H pre-quantized bytes,
// it must produce byte-for-byte identical payload output to calling
// copyQuantizedIntoPayload independently for each pol -- the combined
// pass is a performance change ONLY (see its doc comment for why it's
// faster), never a behavior change.
func TestCopyQuantizedVHIntoPayload_MatchesTwoIndependentPasses(t *testing.T) {
	v := make([]byte, common.HeapLen*2)
	h := make([]byte, common.HeapLen*2)
	for i := range v {
		v[i] = byte(7*i + 1)
		h[i] = byte(11*i + 2)
	}

	wantPayload := make([]byte, PayloadLengthBytes)
	copyQuantizedIntoPayload(v, wantPayload, 0, 4)
	copyQuantizedIntoPayload(h, wantPayload, 2, 4)

	gotPayload := make([]byte, PayloadLengthBytes)
	copyQuantizedVHIntoPayload(v, h, gotPayload)

	if !bytes.Equal(gotPayload, wantPayload) {
		t.Fatalf("copyQuantizedVHIntoPayload result differs from two independent copyQuantizedIntoPayload passes")
	}
}

func TestEncodeChannelHeap_FixedQuantizeScaleOverridesAdaptive(t *testing.T) {
	station := &common.StationConfig{StationID: 1}
	p := NewSpsPacketizer(station, nil)

	v := make([]complex64, common.HeapLen)
	h := make([]complex64, common.HeapLen)
	for i := range v {
		v[i] = complex(3.0, 4.0) // magnitude 5
		h[i] = complex(0.0, 5.0) // magnitude 5
	}
	heap := &common.ChannelHeap{ChannelID: 0, VSamples: v, HSamples: h, HeapStartTime: 1_700_000_000.0}

	// A fixed scale of exactly 1.0 (nothing like the adaptive 127/5=25.4
	// TestEncodeChannelHeap_PayloadInterleavesVHRealImag exercises) --
	// confirms SetQuantizeScale actually takes effect instead of being
	// silently ignored in favor of the adaptive scan.
	p.SetQuantizeScale(1.0)
	raw, err := p.EncodeChannelHeap(heap)
	if err != nil {
		t.Fatalf("EncodeChannelHeap: %v", err)
	}
	_, payload := decodeHeap(t, raw)

	wantVReal, wantVImag := int8(3), int8(4) // round(3*1.0), round(4*1.0) -- NOT the adaptive 76/102
	wantHReal, wantHImag := int8(0), int8(5)
	if got := int8(payload[0]); got != wantVReal {
		t.Fatalf("payload[0] (Vreal) = %d, want %d (fixed scale=1.0 should give exact unscaled rounding)", got, wantVReal)
	}
	if got := int8(payload[1]); got != wantVImag {
		t.Fatalf("payload[1] (Vimag) = %d, want %d", got, wantVImag)
	}
	if got := int8(payload[2]); got != wantHReal {
		t.Fatalf("payload[2] (Hreal) = %d, want %d", got, wantHReal)
	}
	if got := int8(payload[3]); got != wantHImag {
		t.Fatalf("payload[3] (Himag) = %d, want %d", got, wantHImag)
	}

	// SetQuantizeScale(0) must restore the original adaptive behavior --
	// same expectation as TestEncodeChannelHeap_PayloadInterleavesVHRealImag.
	p.SetQuantizeScale(0)
	raw, err = p.EncodeChannelHeap(heap)
	if err != nil {
		t.Fatalf("EncodeChannelHeap after resetting to adaptive: %v", err)
	}
	_, payload = decodeHeap(t, raw)
	if got := int8(payload[0]); got != 76 {
		t.Fatalf("payload[0] (Vreal) after SetQuantizeScale(0) = %d, want 76 (adaptive scale should be restored)", got)
	}
}

func TestEncodeChannelHeap_RejectsHeapCounterOverflow(t *testing.T) {
	station := &common.StationConfig{StationID: 1}
	p := NewSpsPacketizer(station, nil)

	// A time far enough in the future to overflow the ICD's 40-bit
	// heap_counter field (regression guard for the Python CLAUDE.md's bug
	// #17 -- a wrong formula inflated heap_counter by HeapLen=2048x,
	// which this test would also have caught).
	heap := testHeap(0, 1e15)
	if _, err := p.EncodeChannelHeap(heap); err == nil {
		t.Fatal("expected an error for a heap_counter that overflows the 40-bit field, got none")
	}
}

func TestEncodeChannelHeap_CurrentEraTimestampFits(t *testing.T) {
	station := &common.StationConfig{StationID: 1}
	p := NewSpsPacketizer(station, nil)
	heap := testHeap(0, 1_800_000_000.0) // ~2027
	if _, err := p.EncodeChannelHeap(heap); err != nil {
		t.Fatalf("unexpected error for a current-era timestamp: %v", err)
	}
}

// fakeSender records everything written to it.
type fakeSender struct {
	buf bytes.Buffer
}

func (f *fakeSender) Write(p []byte) (int, error) {
	return f.buf.Write(p)
}

func TestSendChannelHeap_WritesEncodedBytes(t *testing.T) {
	station := &common.StationConfig{StationID: 1}
	sender := &fakeSender{}
	p := NewSpsPacketizer(station, sender)

	heap := testHeap(3, 1_700_000_000.0)
	if err := p.SendChannelHeap(heap); err != nil {
		t.Fatalf("SendChannelHeap: %v", err)
	}

	wantEncoded, err := p.EncodeChannelHeap(heap)
	if err != nil {
		t.Fatalf("EncodeChannelHeap: %v", err)
	}
	if !bytes.Equal(sender.buf.Bytes(), wantEncoded) {
		t.Fatal("bytes written to sender do not match EncodeChannelHeap's output")
	}
}

func TestSendChannelHeap_ErrorsWithoutSender(t *testing.T) {
	p := NewSpsPacketizer(&common.StationConfig{StationID: 1}, nil)
	if err := p.SendChannelHeap(testHeap(0, 1_700_000_000.0)); err == nil {
		t.Fatal("expected an error when sending without a configured sender")
	}
}

// TestEncodeChannelHeapInto_MatchesEncodeChannelHeap guards
// BatchSendLoop's zero-allocation hot path (EncodeChannelHeapInto,
// writing into a reused buffer) against the allocating convenience
// wrapper (EncodeChannelHeap) ever silently diverging.
func TestEncodeChannelHeapInto_MatchesEncodeChannelHeap(t *testing.T) {
	station := &common.StationConfig{StationID: 7, SubstationID: 2, SubarrayID: 3, BeamID: 5, ScanID: 99}
	p := NewSpsPacketizer(station, nil)
	heap := testHeap(12, 1_700_000_000.0)

	want, err := p.EncodeChannelHeap(heap)
	if err != nil {
		t.Fatalf("EncodeChannelHeap: %v", err)
	}

	dst := make([]byte, heapWireSizeBytes)
	if err := p.EncodeChannelHeapInto(dst, heap); err != nil {
		t.Fatalf("EncodeChannelHeapInto: %v", err)
	}
	if !bytes.Equal(dst, want) {
		t.Fatal("EncodeChannelHeapInto's output does not match EncodeChannelHeap's")
	}
}

func TestEncodeChannelHeapInto_RejectsWrongDstLength(t *testing.T) {
	p := NewSpsPacketizer(&common.StationConfig{StationID: 1}, nil)
	heap := testHeap(0, 1_700_000_000.0)
	if err := p.EncodeChannelHeapInto(make([]byte, heapWireSizeBytes-1), heap); err == nil {
		t.Fatal("expected an error for a dst buffer of the wrong length")
	}
}

func TestQuantize8bit_ScalesByComplexMagnitude(t *testing.T) {
	// Both samples have magnitude 5 (3-4-5 triangle) -> scale = 127/5 =
	// 25.4, applied per-component.
	samples := []complex64{complex(3, 4), complex(-3, -4)}
	realOut, imagOut := quantize8bit(samples)
	if realOut[0] != 76 { // round(3 * 25.4) = round(76.2)
		t.Fatalf("realOut[0] = %d, want 76", realOut[0])
	}
	if imagOut[0] != 102 { // round(4 * 25.4) = round(101.6)
		t.Fatalf("imagOut[0] = %d, want 102", imagOut[0])
	}
	if realOut[1] != -76 {
		t.Fatalf("realOut[1] = %d, want -76", realOut[1])
	}
	if imagOut[1] != -102 {
		t.Fatalf("imagOut[1] = %d, want -102", imagOut[1])
	}
}

func TestQuantize8bit_ClipsToInt8Range(t *testing.T) {
	// One large outlier sets the scale; a component that would otherwise
	// round past +/-127 due to floating point must be clipped, not
	// overflow/wrap.
	samples := []complex64{complex(1000, 1000), complex(1, 0)}
	realOut, _ := quantize8bit(samples)
	if realOut[0] > 127 || realOut[0] < -128 {
		t.Fatalf("realOut[0] = %d, out of int8 range", realOut[0])
	}
}

func TestQuantize8bit_AllZeroSamplesDoesNotPanic(t *testing.T) {
	samples := make([]complex64, common.HeapLen)
	realOut, imagOut := quantize8bit(samples)
	for i := range realOut {
		if realOut[i] != 0 || imagOut[i] != 0 {
			t.Fatalf("expected all-zero output for all-zero input, got (%d, %d) at %d", realOut[i], imagOut[i], i)
		}
	}
}

// BenchmarkEncodeChannelHeapInto isolates the per-heap encode cost (scale
// pass + quantize/round/clamp + header/item writes) from noise
// generation/copying -- the real-hardware profile that motivated
// QuantizeComponent/quantize8bitScale's sqrt/round fixes measured this
// path (spead.BatchSendLoop's per-heap work) at ~46% of ALL CPU time on
// the target EPYC box, dominated by a per-sample math.Sqrt and two
// per-sample math.Round calls (see those functions' doc comments). Not a
// substitute for a real target-hardware profile -- this dev machine's
// core count/microarchitecture differ -- but a same-machine before/after
// comparison here is a direct, cheap way to confirm those fixes actually
// reduced CPU time before the target hardware confirms the real-world
// magnitude.
func BenchmarkEncodeChannelHeapInto(b *testing.B) {
	station := &common.StationConfig{StationID: 1, SubstationID: 0, SubarrayID: 1, BeamID: 1, ScanID: 1}
	heap := testHeap(0, 1_700_000_000.0)
	dst := make([]byte, heapWireSizeBytes)

	// Sub-benchmarks the DEFAULT adaptive per-heap scale (quantize8bitScale
	// re-scanning heap.VSamples/HSamples for their own max magnitude, every
	// call) against a FIXED scale (SetQuantizeScale, see
	// synth.DirectSynthesisStreamer.QuantizeScale) -- isolates exactly the
	// cost the fixed-scale path is meant to remove.
	b.Run("adaptive", func(b *testing.B) {
		p := NewSpsPacketizer(station, nil)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := p.EncodeChannelHeapInto(dst, heap); err != nil {
				b.Fatalf("EncodeChannelHeapInto: %v", err)
			}
		}
	})
	b.Run("fixed_scale", func(b *testing.B) {
		p := NewSpsPacketizer(station, nil)
		p.SetQuantizeScale(127.0 / 200.0) // arbitrary but representative fixed scale
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := p.EncodeChannelHeapInto(dst, heap); err != nil {
				b.Fatalf("EncodeChannelHeapInto: %v", err)
			}
		}
	})
	// Pre-quantized (ChannelHeap.VQuantized/HQuantized set, see
	// synth.DirectSynthesisStreamer.GenerateQuantizedHeaps): no scale, no
	// scan, no rounding/clamping at all -- just a byte copy. This is the
	// path noise-only channels actually take in production; "adaptive"/
	// "fixed_scale" above only apply to tone-affected channels now.
	quantizedHeap := &common.ChannelHeap{
		ChannelID:     0,
		VQuantized:    make([]byte, common.HeapLen*2),
		HQuantized:    make([]byte, common.HeapLen*2),
		HeapStartTime: 1_700_000_000.0,
	}
	b.Run("quantized", func(b *testing.B) {
		p := NewSpsPacketizer(station, nil)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := p.EncodeChannelHeapInto(dst, quantizedHeap); err != nil {
				b.Fatalf("EncodeChannelHeapInto: %v", err)
			}
		}
	})
}
