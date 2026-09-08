// Command pcap-dump writes a small pcap file containing generated SPEAD
// heaps -- no gRPC, no UDP socket, no Tango, just internal/synth's noise
// generation and internal/spead's encoder, wrapped in synthetic
// Ethernet/IPv4/UDP framing (internal/pcap) so the result opens directly
// in Wireshark/tcpdump for inspecting the ICD's SPEAD-64-48 heap
// structure.
//
// This exists to look at STRUCTURE, not to produce a large or realistic
// capture: content is noise-only (no tone, no real delay tracking) and
// not meaningful, and defaults are chosen to keep the file small (8
// channels x 1 tick = 8 heaps, ~65KB).
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
	numChannels := flag.Int("num-channels", common.MinNumChannels, "number of channels (8-384, in steps of 8) -- kept at the ICD minimum by default to keep the pcap file small")
	numTicks := flag.Int("num-ticks", 1, "number of ticks (one heap per channel per tick) to generate")
	flag.Parse()

	if *numTicks < 1 {
		log.Fatalf("-num-ticks must be >= 1, got %d", *numTicks)
	}

	obsTime := float64(time.Now().UnixNano()) / 1e9
	stationCfg := &common.StationConfig{StationID: 1, SubstationID: 1, SubarrayID: 1, BeamID: 1, ScanID: 1}

	// Noise only, no tone sources: every channel then comes out of
	// GenerateQuantizedHeaps directly, with no HeapAccumulator/ScanRunner
	// pacing loop needed at all -- this tool has no real-time deadline to
	// meet, it just wants N ticks' worth of heaps as fast as possible.
	streamer, err := synth.NewDirectSynthesisStreamer(synth.StreamerConfig{
		Station:     stationCfg,
		ObsTimeRef:  obsTime,
		NumChannels: *numChannels,
		Noise:       &synth.NoiseConfig{Std: 0.05, Seed: int64(stationCfg.StationID)},
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

	numHeaps := 0
	for tick := 0; tick < *numTicks; tick++ {
		t := obsTime + float64(tick)*common.BlockDurationS
		for _, heap := range streamer.GenerateQuantizedHeaps(t) {
			encoded, err := packetizer.EncodeChannelHeap(heap)
			common.ReleaseSampleBuffers(heap)
			if err != nil {
				log.Fatalf("encoding heap ch=%d: %v", heap.ChannelID, err)
			}
			if err := pw.WriteUDPPacket(encoded, time.Now()); err != nil {
				log.Fatalf("writing pcap packet: %v", err)
			}
			numHeaps++
		}
	}

	log.Printf("wrote %d heaps (%d channels x %d tick(s)) to %s", numHeaps, *numChannels, *numTicks, *out)
}
