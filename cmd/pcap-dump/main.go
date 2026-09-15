// Command pcap-dump writes a small pcap file containing generated SPEAD
// heaps -- no gRPC, no UDP socket, no Tango, just internal/synth's noise
// (and, if -tone-freq is set, tone) generation and internal/spead's
// encoder, wrapped in synthetic Ethernet/IPv4/UDP framing (internal/pcap)
// so the result opens directly in Wireshark/tcpdump for inspecting the
// ICD's SPEAD-64-48 heap structure.
//
// This exists to look at STRUCTURE, not to produce a large or realistic
// capture: content is noise-only by default (no tone, no real delay
// tracking) and not meaningful, and defaults are chosen to keep the file
// small (8 channels x 1 tick = 8 heaps, ~65KB). -tone-freq/-tone-amp
// route their channel through the SAME GenerateNextTick+HeapAccumulator
// path ScanRunner uses in a real scan (see internal/common/scan_runner.go's
// run()) -- every OTHER channel still goes through GenerateQuantizedHeaps,
// exactly as ScanRunner also calls both, alongside each other, every tick.
package main

import (
	"flag"
	"log"
	"os"
	"time"

	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/common"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/pcap"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/spead"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/synth"
)

func main() {
	out := flag.String("out", "spead.pcap", "output pcap file path")
	startChannel := flag.Int("start-channel", int(common.ChannelStart), "first channel to generate (64-447)")
	numChannels := flag.Int("num-channels", common.MinNumChannels, "number of channels (8-384, in steps of 8) -- kept at the ICD minimum by default to keep the pcap file small")
	numTicks := flag.Int("num-ticks", 1, "number of ticks (one heap per channel per tick) to generate")
	toneFreqHz := flag.Float64("tone-freq", 0, "tone frequency in Hz (0 = no tone, default)")
	toneAmplitude := flag.Float64("tone-amp", 1.0, "tone amplitude (default 1.0)")
	negateDelay := flag.Bool("negate-delay", false, "negate tone delay polynomial coefficients before evaluating delay -- mirrors CBF's ska-low-cbf-proc STN_DELAY_SIGN=neg (see synth.StreamerConfig.NegateDelay)")
	flag.Parse()

	if *numTicks < 1 {
		log.Fatalf("-num-ticks must be >= 1, got %d", *numTicks)
	}

	obsTime := float64(time.Now().UnixNano()) / 1e9
	// StartChannel: common.ChannelStart -- the band's first channel,
	// matching this tool's fixed "no sub-band selection" scope.
	stationCfg := &common.StationConfig{StationID: 1, SubstationID: 1, SubarrayID: 1, BeamID: 1, ScanID: 1, StartChannel: int32(*startChannel)}

	var toneSources []synth.ToneSourceConfig
	if *toneFreqHz > 0 {
		// Static, always-zero-delay, never expires -- see this file's
		// package doc comment for why that's fine here (isolating
		// computational cost only) but never acceptable for a real
		// scan.
		toneFeed := common.NewDelayFeed("pcap-dump-static-test-tone")
		toneFeed.Update(&common.DelayPolynomial{
			// Anchored at obsTime, not 0 -- obsTime defaults to "now"
			// (a real, large Unix epoch value), so a validity window
			// anchored at epoch 0 would already be expired by the time
			// this runs.
			StartValiditySec:  obsTime,
			ValidityPeriodSec: 1e9,
			XYPolCoeffsNs: []float64{
				8501.19040321062, -0.7822609065561508, -1.6592202916405917e-05, 6.932753103396767e-10, 7.355307614008184e-15, -1.8744975492671921e-19,
			},
		})
		toneSources = []synth.ToneSourceConfig{
			{DelayFeed: toneFeed, FreqHz: *toneFreqHz, Amplitude: *toneAmplitude},
		}
	}

	// Noise only, no tone sources: every channel then comes out of
	// GenerateQuantizedHeaps directly, with no HeapAccumulator/ScanRunner
	// pacing loop needed at all -- this tool has no real-time deadline to
	// meet, it just wants N ticks' worth of heaps as fast as possible.
	streamer, err := synth.NewDirectSynthesisStreamer(synth.StreamerConfig{
		Station:      stationCfg,
		ObsTimeRef:   obsTime,
		NumChannels:  *numChannels,
		StartChannel: int(stationCfg.StartChannel),
		Noise:        &synth.NoiseConfig{Std: 0.05, Seed: int64(stationCfg.StationID)},
		ToneSources:  toneSources,
		NegateDelay:  *negateDelay,
	})
	if err != nil {
		log.Fatalf("constructing streamer: %v", err)
	}

	packetizer := spead.NewSpsPacketizer(stationCfg, nil)
	packetizer.SetQuantizeScale(streamer.QuantizeScale())

	f, err := os.Create(*out)
	if err != nil {
		log.Fatalf("creating -out file %q: %v", *out, err)
	}
	defer f.Close()

	pw, err := pcap.NewWriter(f)
	if err != nil {
		log.Fatalf("%v", err)
	}

	// Tone-bearing channels never come out of GenerateQuantizedHeaps
	// (that only covers noise-only channels -- see
	// DirectSynthesisStreamer.GenerateQuantizedHeaps's doc comment): they
	// need the complex64 dst-write/HeapAccumulator path instead, same as
	// ScanRunner drives in a real scan. Sized to ComplexPathChannelIDMap,
	// not the full channel range, matching NewScanRunner.
	nSamplesPerTick := streamer.TickNSamples()
	complexChannelIDMap := streamer.ComplexPathChannelIDMap()
	accumulator := common.NewHeapAccumulator(
		len(complexChannelIDMap), obsTime, common.ChannelOutputRateHz, complexChannelIDMap,
	)

	writeHeap := func(heap *common.ChannelHeap) {
		encoded, err := packetizer.EncodeChannelHeap(heap)
		common.ReleaseSampleBuffers(heap)
		if err != nil {
			log.Fatalf("encoding heap ch=%d: %v", heap.ChannelID, err)
		}
		if err := pw.WriteUDPPacket(encoded, time.Now()); err != nil {
			log.Fatalf("writing pcap packet: %v", err)
		}
	}

	numHeaps := 0
	for tick := 0; tick < *numTicks; tick++ {
		t := obsTime + float64(tick)*common.BlockDurationS

		dst := map[string][][]complex64{
			"V": accumulator.PrepareWrite("V", nSamplesPerTick),
			"H": accumulator.PrepareWrite("H", nSamplesPerTick),
		}
		streamer.GenerateNextTick(t, nSamplesPerTick, dst)
		for _, heap := range accumulator.PopReadyHeaps() {
			writeHeap(heap)
			numHeaps++
		}

		for _, heap := range streamer.GenerateQuantizedHeaps(t) {
			writeHeap(heap)
			numHeaps++
		}
	}

	log.Printf("wrote %d heaps (%d channels x %d tick(s)) to %s", numHeaps, *numChannels, *numTicks, *out)
}
