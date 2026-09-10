// Package server implements the gRPC control plane for the Go
// simulator — the equivalent of simulator.py's StationSimulatorDevice,
// minus the Tango-facing bits (device properties, attribute
// subscriptions to CBF's delay-poly emulator), which are expected to
// live on whatever process sits on the other side of this gRPC service
// (see api/simulator.proto's doc comment for the intended split: a
// Tango device server owns Tango, forwards delay updates here via
// PushDelayUpdate, and this process owns signal generation + SPEAD/UDP
// sending).
package server

import (
	"context"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/sanderploegsma/ska-low-station-beam-simulator/api/simulatorpb"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/common"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/netutil"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/spead"
	"github.com/sanderploegsma/ska-low-station-beam-simulator/internal/synth"
)

// Server implements pb.StationSimulatorServer.
type Server struct {
	pb.UnimplementedStationSimulatorServer

	destIP          string
	destPort        int
	sourceInterface string // "" -- let the OS pick the outbound interface/address

	numSenders         int
	sendBatchSize      int
	udpSendBufferBytes int

	mu         sync.Mutex
	stationCfg *common.StationConfig
	scanRunner *common.ScanRunner
	delayFeeds map[string]*common.DelayFeed

	sendQueue  *common.HeapQueue
	senderPool *spead.SenderPool
	shutdown   chan struct{}
}

// Sender-pool defaults, matching internal/spead's own so a Server built
// with no Options behaves the same as one explicitly given these.
const (
	DefaultNumSenders         = spead.DefaultNumSenders
	DefaultSendBatchSize      = spead.DefaultSendBatchSize
	DefaultUDPSendBufferBytes = spead.DefaultUDPSendBufferBytes
)

// Option configures optional Server behavior beyond NewServer's required
// per-pod identity/destination arguments.
type Option func(*Server)

// WithNumSenders sets the number of parallel UDP sender sockets/goroutines.
func WithNumSenders(n int) Option { return func(s *Server) { s.numSenders = n } }

// WithSendBatchSize sets the max heaps per batched UDP send.
func WithSendBatchSize(n int) Option { return func(s *Server) { s.sendBatchSize = n } }

// WithUDPSendBufferBytes sets each sender socket's SO_SNDBUF size.
func WithUDPSendBufferBytes(n int) Option { return func(s *Server) { s.udpSendBufferBytes = n } }

// NewServer constructs a Server for one station. stationID/substationID
// identify this pod; destIP/destPort are the CBF SPEAD/UDP endpoint —
// both static for the pod's lifetime, matching simulator.py's device
// properties (subarray_id/beam_id/source_cfgs vary per scan instead, see
// StartScan).
//
// sourceInterface, if non-empty, names a local network interface (e.g.
// "net1" for a Multus-attached secondary NIC) whose IPv4 address the
// outbound SPEAD/UDP socket is bound to, forcing that traffic out over
// that interface rather than whatever the OS's default route would pick
// -- see Start(), which resolves the interface's address at socket-dial
// time (it isn't known ahead of time: a Multus secondary interface's
// address comes from an IPAM pool assigned at pod start). Pass "" to
// leave this to the OS, as before.
func NewServer(stationID, substationID int32, destIP string, destPort int, sourceInterface string, opts ...Option) *Server {
	s := &Server{
		destIP:          destIP,
		destPort:        destPort,
		sourceInterface: sourceInterface,
		stationCfg: &common.StationConfig{
			StationID:    stationID,
			SubstationID: substationID,
		},
		numSenders:         DefaultNumSenders,
		sendBatchSize:      DefaultSendBatchSize,
		udpSendBufferBytes: DefaultUDPSendBufferBytes,
		sendQueue:          common.NewHeapQueue(common.QueueMaxSize),
		shutdown:           make(chan struct{}),
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Start dials the CBF SPEAD/UDP destination and starts the sender
// goroutine (the equivalent of Python's sender_loop). Call once before
// serving gRPC traffic.
func (s *Server) Start() error {
	log.Println("server starting")
	var localAddr *net.UDPAddr
	if s.sourceInterface != "" {
		ip, err := netutil.InterfaceIPv4Addr(s.sourceInterface)
		if err != nil {
			return fmt.Errorf("resolving source_interface %q for outbound SPEAD/UDP: %w", s.sourceInterface, err)
		}
		localAddr = &net.UDPAddr{IP: ip}
		log.Printf("binding outbound SPEAD/UDP socket to interface %q (%s)", s.sourceInterface, ip)
	}

	destAddr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(s.destIP, fmt.Sprintf("%d", s.destPort)))
	if err != nil {
		return fmt.Errorf("resolving SPEAD destination %s:%d: %w", s.destIP, s.destPort, err)
	}
	pool, err := spead.NewSenderPool(s.stationCfg, s.sendQueue.Recv(), localAddr, destAddr, s.numSenders, s.udpSendBufferBytes, s.sendBatchSize, s.shutdown)
	if err != nil {
		return fmt.Errorf("starting SPEAD/UDP sender pool: %w", err)
	}
	s.senderPool = pool
	return nil
}

// Stop stops any running scan and the sender goroutines.
func (s *Server) Stop() {
	log.Println("server stopping")
	s.mu.Lock()
	if s.scanRunner != nil {
		s.scanRunner.Stop(5 * time.Second)
	}
	s.mu.Unlock()
	close(s.shutdown)
	if s.senderPool != nil {
		s.senderPool.Close()
	}
}

// StartScan implements pb.StationSimulatorServer. Fails if a scan is
// already running — call StopScan first, matching simulator.py.
func (s *Server) StartScan(ctx context.Context, req *pb.StartScanRequest) (*pb.StartScanResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.scanRunner != nil && s.scanRunner.IsRunning() {
		return nil, status.Error(codes.FailedPrecondition, "scan already running -- call StopScan first")
	}

	s.stationCfg.SubarrayID = req.SubarrayId
	s.stationCfg.BeamID = req.BeamId
	s.stationCfg.ScanID = req.ScanId

	// Fresh delay feeds per scan -- a source_id from a previous scan
	// must not silently keep receiving updates meant for a different
	// scan's source of the same name.
	delayFeeds := make(map[string]*common.DelayFeed, len(req.ToneSources))
	toneCfgs := make([]synth.ToneSourceConfig, 0, len(req.ToneSources))
	for _, ts := range req.ToneSources {
		if ts.SourceId == "" {
			return nil, status.Errorf(codes.InvalidArgument, "tone source with freq_hz=%v is missing required source_id", ts.FreqHz)
		}
		if _, dup := delayFeeds[ts.SourceId]; dup {
			return nil, status.Errorf(codes.InvalidArgument, "duplicate tone source_id %q", ts.SourceId)
		}
		feed := common.NewDelayFeed(ts.SourceId)
		delayFeeds[ts.SourceId] = feed
		toneCfgs = append(toneCfgs, synth.ToneSourceConfig{
			DelayFeed: feed,
			FreqHz:    ts.FreqHz,
			Amplitude: ts.Amplitude,
		})
	}

	var noiseCfg *synth.NoiseConfig
	if req.Noise != nil {
		noiseCfg = &synth.NoiseConfig{Std: req.Noise.Std, Seed: req.Noise.Seed}
	}

	streamer, err := synth.NewDirectSynthesisStreamer(synth.StreamerConfig{
		Station:     s.stationCfg,
		ToneSources: toneCfgs,
		ObsTimeRef:  req.ObsTimeEpochS,
		Noise:       noiseCfg,
		NumChannels: int(req.NumChannels),
	})
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
	}

	// Updates the SHARED SenderPool's packetizer -- that pool was created
	// once at Start(), before this (or any) scan's noise/tone config was
	// known, and may outlive many scans with different configs, so this
	// must be a live update, not a construction-time value (see
	// spead.SpsPacketizer.SetQuantizeScale's doc comment).
	if s.senderPool != nil {
		s.senderPool.SetQuantizeScale(streamer.QuantizeScale())
	}

	log.Printf("starting scan %d with %d tone sources and %d channels (subarray=%d beam=%d)", req.ScanId, len(req.ToneSources), req.NumChannels, req.SubarrayId, req.BeamId)
	s.delayFeeds = delayFeeds
	s.scanRunner = common.NewScanRunner(streamer, s.sendQueue, req.ObsTimeEpochS, req.ScanDurationS)
	s.scanRunner.Start()

	return &pb.StartScanResponse{Ok: true, Message: "scan started"}, nil
}

// StopScan implements pb.StationSimulatorServer. Idempotent.
func (s *Server) StopScan(ctx context.Context, req *pb.StopScanRequest) (*pb.StopScanResponse, error) {
	s.mu.Lock()
	runner := s.scanRunner
	s.mu.Unlock()
	if runner != nil {
		log.Println("stopping current scan")
		runner.Stop(5 * time.Second)
	}
	return &pb.StopScanResponse{Ok: true}, nil
}

// PushDelayUpdate implements pb.StationSimulatorServer, forwarding a
// freshly-received delay polynomial to the named source's DelayFeed.
func (s *Server) PushDelayUpdate(ctx context.Context, req *pb.PushDelayUpdateRequest) (*pb.PushDelayUpdateResponse, error) {
	s.mu.Lock()
	feed, ok := s.delayFeeds[req.SourceId]
	s.mu.Unlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "no source with source_id=%q in the current scan", req.SourceId)
	}
	if req.Polynomial == nil {
		return nil, status.Error(codes.InvalidArgument, "polynomial is required")
	}
	log.Printf("received delay polynomial for source %q: start=%v validity=%v coeffs=%v offset=%v", req.SourceId, req.Polynomial.StartValiditySec, req.Polynomial.ValidityPeriodSec, req.Polynomial.XypolCoeffsNs, req.Polynomial.YpolOffsetNs)
	feed.Update(&common.DelayPolynomial{
		StationID:         req.Polynomial.StationId,
		StartValiditySec:  req.Polynomial.StartValiditySec,
		ValidityPeriodSec: req.Polynomial.ValidityPeriodSec,
		XYPolCoeffsNs:     req.Polynomial.XypolCoeffsNs,
		YPolOffsetNs:      req.Polynomial.YpolOffsetNs,
	})
	return &pb.PushDelayUpdateResponse{Ok: true}, nil
}

// GetStatus implements pb.StationSimulatorServer.
func (s *Server) GetStatus(ctx context.Context, req *pb.GetStatusRequest) (*pb.StatusResponse, error) {
	return s.snapshotStatus(), nil
}

// snapshotStatus builds one point-in-time StatusResponse, shared by
// GetStatus and WatchStatus. Only ever holds s.mu long enough to
// snapshot scanRunner -- never across a Send -- so a long-lived
// WatchStatus stream never blocks StartScan/StopScan/PushDelayUpdate.
func (s *Server) snapshotStatus() *pb.StatusResponse {
	s.mu.Lock()
	runner := s.scanRunner
	running := runner != nil && runner.IsRunning()
	s.mu.Unlock()

	// A finished/never-started runner reports 0 for both -- there is no
	// "last tick" to report once nothing is running, matching
	// DriftSeconds/TickNumber's own pre-first-tick zero value.
	var drift float64
	var tickNumber int64
	if running {
		drift = runner.DriftSeconds()
		tickNumber = runner.TickNumber()
	}

	return &pb.StatusResponse{
		ScanRunning:  running,
		QueueDepth:   int32(s.sendQueue.Len()),
		DriftSeconds: drift,
		TickNumber:   tickNumber,
	}
}

// DefaultWatchStatusInterval is used when a WatchStatusRequest doesn't
// specify update_interval_s (or specifies a non-positive value).
const DefaultWatchStatusInterval = time.Second

// WatchStatus implements pb.StationSimulatorServer, pushing a
// StatusResponse every update_interval_s until the caller
// cancels/disconnects. Intended to live for as long as the caller wants
// updates (e.g. the Tango device server's lifetime) -- independent of,
// and never blocking, StartScan/StopScan/PushDelayUpdate, which gRPC
// multiplexes on the same channel as ordinary concurrent unary calls.
func (s *Server) WatchStatus(req *pb.WatchStatusRequest, stream pb.StationSimulator_WatchStatusServer) error {
	interval := DefaultWatchStatusInterval
	if req.UpdateIntervalS > 0 {
		interval = time.Duration(req.UpdateIntervalS * float64(time.Second))
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	ctx := stream.Context()
	for {
		if err := stream.Send(s.snapshotStatus()); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-s.shutdown:
			return nil
		case <-ticker.C:
		}
	}
}
