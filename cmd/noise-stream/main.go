// Command noise-stream directly starts a noise (+ optional tone)
// SPEAD/UDP stream — no gRPC control plane, no real delay sources, just
// the noise tile bank and (if -tone-freq-hz is set) a tone source (see
// internal/synth) driven by a ScanRunner exactly like a real scan, for
// quickly exercising the numeric core + SPEAD packetizer end-to-end
// (e.g. against a packet capture tool, or CBF's receive path) without
// standing up the gRPC service or a Tango-facing counterpart process.
//
// -tone-freq-hz uses a STATIC, always-zero-delay feed, never updated
// again after construction — real scans always get their tone's delay
// from a live Tango attribute subscription (see simulator.py's StartScan;
// a real delay path is otherwise required, no exceptions). A static
// feed here is fine ONLY because this tool exists to isolate tone's
// computational cost on top of noise, not to validate delay-tracking
// correctness — don't read anything about delay behavior from a run of
// this tool.
package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"runtime/pprof"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/common"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/netutil"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/spead"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/synth"
)

func main() {
	cpuProfile := flag.String("cpuprofile", "", "write a CPU profile (whole process, generation + sending) to this file, for use with 'go tool pprof'")
	destIP := flag.String("dest-ip", "127.0.0.1", "CBF SPEAD/UDP destination IP")
	destPort := flag.Int("dest-port", 8000, "CBF SPEAD/UDP destination port")
	sourceInterface := flag.String("spead-interface", "", "network interface to bind the outbound SPEAD/UDP socket to (e.g. net1 for a Multus-attached secondary NIC); empty leaves this to the OS's default route selection")

	stationID := flag.Int("station-id", 1, "station ID")
	substationID := flag.Int("substation-id", 1, "substation ID")
	subarrayID := flag.Int("subarray-id", 1, "subarray ID")
	beamID := flag.Int("beam-id", 1, "beam ID")
	scanID := flag.Int("scan-id", 1, "scan ID")

	numChannels := flag.Int("num-channels", 96, "number of channels (8-384, in steps of 8)")

	obsTimeFlag := flag.String("obs-time", "now", `observation time reference: "now", or a Unix epoch seconds value`)
	scanDuration := flag.Float64("scan-duration", 60.0, "scan duration, in seconds")

	noiseStd := flag.Float64("noise-std", 0.05, "noise standard deviation (both real and imaginary parts)")
	noiseSeed := flag.Int64("noise-seed", 0, "noise seed (default: same as -station-id, unless explicitly set)")

	toneFreqHz := flag.Float64("tone-freq-hz", 0, "if > 0, add a single tone source at this frequency (Hz) on top of the noise, using a STATIC always-zero-delay feed -- for measuring tone's computational cost, NOT a real delay path (see this file's package doc comment)")
	toneAmplitude := flag.Float64("tone-amplitude", 1.0, "amplitude for -tone-freq-hz's tone source (only used if -tone-freq-hz > 0)")

	numSenders := flag.Int("sender-goroutines", 0, "number of parallel UDP sender sockets/goroutines for outbound SPEAD/UDP (0: auto -- scales with -num-channels, spead.DefaultNumSendersForChannels)")
	sendBatchSize := flag.Int("send-batch-size", spead.DefaultSendBatchSize, "max heaps per batched UDP send (uses sendmmsg on Linux)")
	udpSendBufferBytes := flag.Int("udp-send-buffer-bytes", spead.DefaultUDPSendBufferBytes, "SO_SNDBUF size for each outbound SPEAD/UDP socket, in bytes (0: leave at OS default)")

	flag.Parse()

	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatalf("creating -cpuprofile file %q: %v", *cpuProfile, err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("starting CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	obsTime, err := parseObsTime(*obsTimeFlag)
	if err != nil {
		log.Fatalf("invalid -obs-time: %v", err)
	}

	seed := *noiseSeed
	if !isFlagSet("noise-seed") {
		seed = int64(*stationID)
	}

	if !isFlagSet("sender-goroutines") {
		*numSenders = spead.DefaultNumSendersForChannels(*numChannels)
	}

	stationCfg := &common.StationConfig{
		StationID:    int32(*stationID),
		SubstationID: int32(*substationID),
		SubarrayID:   int32(*subarrayID),
		BeamID:       int32(*beamID),
		ScanID:       int64(*scanID),
	}

	var toneSources []synth.ToneSourceConfig
	if *toneFreqHz > 0 {
		// Static, always-zero-delay, never expires -- see this file's
		// package doc comment for why that's fine here (isolating
		// computational cost only) but never acceptable for a real
		// scan.
		toneFeed := common.NewDelayFeed("noise-stream-static-test-tone")
		toneFeed.Update(&common.DelayPolynomial{
			// Anchored at obsTime, not 0 -- obsTime defaults to "now"
			// (a real, large Unix epoch value), so a validity window
			// anchored at epoch 0 would already be expired by the time
			// this runs.
			StartValiditySec:  obsTime,
			ValidityPeriodSec: 1e9,
			XYPolCoeffsNs:     []float64{0.0},
		})
		toneSources = []synth.ToneSourceConfig{
			{DelayFeed: toneFeed, FreqHz: *toneFreqHz, Amplitude: *toneAmplitude},
		}
	}

	streamer, err := synth.NewDirectSynthesisStreamer(synth.StreamerConfig{
		Station:     stationCfg,
		ObsTimeRef:  obsTime,
		NumChannels: *numChannels,
		Noise:       &synth.NoiseConfig{Std: *noiseStd, Seed: seed},
		ToneSources: toneSources,
	})
	if err != nil {
		log.Fatalf("constructing streamer: %v", err)
	}

	var localAddr *net.UDPAddr
	if *sourceInterface != "" {
		ip, err := netutil.InterfaceIPv4Addr(*sourceInterface)
		if err != nil {
			log.Fatalf("resolving -spead-interface=%q: %v", *sourceInterface, err)
		}
		localAddr = &net.UDPAddr{IP: ip}
		log.Printf("binding outbound SPEAD/UDP socket to interface %q (%s)", *sourceInterface, ip)
	}
	destAddr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(*destIP, strconv.Itoa(*destPort)))
	if err != nil {
		log.Fatalf("resolving destination %s:%d: %v", *destIP, *destPort, err)
	}

	queue := common.NewHeapQueue(common.QueueMaxSize)
	shutdown := make(chan struct{})
	pool, err := spead.NewSenderPool(stationCfg, queue.Recv(), localAddr, destAddr, *numSenders, *udpSendBufferBytes, *sendBatchSize, shutdown)
	if err != nil {
		log.Fatalf("starting SPEAD/UDP sender pool: %v", err)
	}
	pool.SetQuantizeScale(streamer.QuantizeScale())

	runner := common.NewScanRunner(streamer, queue, obsTime, *scanDuration)
	runner.Start()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	toneInfo := "none"
	if len(toneSources) > 0 {
		toneInfo = fmt.Sprintf("freq_hz=%v amplitude=%v (STATIC zero-delay test feed, not a real delay path)", *toneFreqHz, *toneAmplitude)
	}
	log.Printf(
		"streaming: station_id=%d substation_id=%d subarray_id=%d beam_id=%d scan_id=%d num_channels=%d dest=%s:%d duration=%.1fs noise_std=%v noise_seed=%d tone=%s sender_goroutines=%d send_batch_size=%d udp_send_buffer_bytes=%d",
		*stationID, *substationID, *subarrayID, *beamID, *scanID, *numChannels, *destIP, *destPort, *scanDuration, *noiseStd, seed,
		toneInfo, *numSenders, *sendBatchSize, *udpSendBufferBytes,
	)

	select {
	case <-runner.Done():
		log.Printf("scan finished")
	case sig := <-sigCh:
		log.Printf("received %s, stopping", sig)
		runner.Stop(5 * time.Second)
	}
	// Stop the sender goroutines and wait for them to actually return
	// BEFORE closing their sockets -- otherwise a heap already pulled off
	// the queue can lose its race against Close() and fail with "use of
	// closed network connection" on the way out. SenderPool.Close()
	// handles that ordering internally.
	close(shutdown)
	pool.Close()
}

// isFlagSet reports whether the named flag was explicitly passed on the
// command line, as opposed to left at its declared default.
func isFlagSet(name string) bool {
	set := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

// parseObsTime accepts "now" (case-insensitive) or a Unix epoch seconds
// value, matching the reference time every generation kernel in
// internal/synth is relative to (see that package's doc comments on why
// raw epoch-scale time isn't used directly).
func parseObsTime(s string) (float64, error) {
	if strings.EqualFold(s, "now") {
		return float64(time.Now().UnixNano()) / 1e9, nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("must be \"now\" or a Unix epoch seconds value: %w", err)
	}
	return v, nil
}
