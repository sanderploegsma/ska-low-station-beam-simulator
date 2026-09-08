// Command noise-stream directly starts a noise-only SPEAD/UDP stream —
// no gRPC control plane, no tone/delay sources, just the noise tile bank
// (see internal/synth) driven by a ScanRunner exactly like a real scan,
// for quickly exercising the numeric core + SPEAD packetizer end-to-end
// (e.g. against a packet capture tool, or CBF's receive path) without
// standing up the gRPC service or a Tango-facing counterpart process.
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

	"github.com/skao/station-beam-simulator-go/internal/common"
	"github.com/skao/station-beam-simulator-go/internal/netutil"
	"github.com/skao/station-beam-simulator-go/internal/spead"
	"github.com/skao/station-beam-simulator-go/internal/synth"
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

	numSenders := flag.Int("sender-goroutines", spead.DefaultNumSenders, "number of parallel UDP sender sockets/goroutines for outbound SPEAD/UDP")
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

	stationCfg := &common.StationConfig{
		StationID:    int32(*stationID),
		SubstationID: int32(*substationID),
		SubarrayID:   int32(*subarrayID),
		BeamID:       int32(*beamID),
		ScanID:       int64(*scanID),
	}

	streamer, err := synth.NewDirectSynthesisStreamer(synth.StreamerConfig{
		Station:     stationCfg,
		ObsTimeRef:  obsTime,
		NumChannels: *numChannels,
		Noise:       &synth.NoiseConfig{Std: *noiseStd, Seed: seed},
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

	runner := common.NewScanRunner(streamer, queue, obsTime, *scanDuration)
	runner.Start()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	log.Printf(
		"streaming noise-only: station_id=%d substation_id=%d subarray_id=%d beam_id=%d scan_id=%d num_channels=%d dest=%s:%d duration=%.1fs noise_std=%v noise_seed=%d sender_goroutines=%d send_batch_size=%d udp_send_buffer_bytes=%d",
		*stationID, *substationID, *subarrayID, *beamID, *scanID, *numChannels, *destIP, *destPort, *scanDuration, *noiseStd, seed,
		*numSenders, *sendBatchSize, *udpSendBufferBytes,
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
