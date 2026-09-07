package server

import (
	"context"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	pb "github.com/skao/station-beam-simulator-go/api/simulatorpb"
	"github.com/skao/station-beam-simulator-go/internal/common"
)

// newTestClient spins up a Server over an in-memory bufconn listener and
// returns a connected client plus a cleanup func. destPort is a UDP port
// nothing needs to be listening on -- UDP is connectionless, so
// net.Dial("udp", ...) succeeds even with no receiver, and SendChannelHeap
// calls simply go nowhere, which is fine for exercising the control
// plane (StartScan/StopScan/PushDelayUpdate/GetStatus) in isolation.
func newTestClient(t *testing.T) (pb.StationSimulatorClient, *Server, func()) {
	t.Helper()
	srv := NewServer(1, 0, "127.0.0.1", 19999, "")
	if err := srv.Start(); err != nil {
		t.Fatalf("srv.Start(): %v", err)
	}

	lis := bufconn.Listen(1024 * 1024)
	grpcServer := grpc.NewServer()
	pb.RegisterStationSimulatorServer(grpcServer, srv)
	go grpcServer.Serve(lis)

	conn, err := grpc.NewClient("passthrough:///bufconn",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}

	cleanup := func() {
		conn.Close()
		grpcServer.Stop()
		srv.Stop()
	}
	return pb.NewStationSimulatorClient(conn), srv, cleanup
}

func TestServer_StartScanThenGetStatusReportsRunning(t *testing.T) {
	client, _, cleanup := newTestClient(t)
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.StartScan(ctx, &pb.StartScanRequest{
		ObsTimeEpochS: 1_700_000_000.0,
		ScanDurationS: 1000 * common.BlockDurationS, // long enough to still be running when we check
		ScanId:        1,
		SubarrayId:    1,
		BeamId:        1,
		NumChannels:   8,
		Noise:         &pb.NoiseConfig{Std: 0.05, Seed: 42},
	})
	if err != nil {
		t.Fatalf("StartScan: %v", err)
	}
	if !resp.Ok {
		t.Fatalf("StartScan response Ok=false, message=%q", resp.Message)
	}

	status, err := client.GetStatus(ctx, &pb.GetStatusRequest{})
	if err != nil {
		t.Fatalf("GetStatus: %v", err)
	}
	if !status.ScanRunning {
		t.Fatal("expected ScanRunning=true immediately after StartScan")
	}

	if _, err := client.StopScan(ctx, &pb.StopScanRequest{}); err != nil {
		t.Fatalf("StopScan: %v", err)
	}
	status, err = client.GetStatus(ctx, &pb.GetStatusRequest{})
	if err != nil {
		t.Fatalf("GetStatus after StopScan: %v", err)
	}
	if status.ScanRunning {
		t.Fatal("expected ScanRunning=false after StopScan")
	}
}

func TestServer_StartScanTwiceWithoutStopFails(t *testing.T) {
	client, _, cleanup := newTestClient(t)
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.StartScanRequest{
		ObsTimeEpochS: 1_700_000_000.0,
		ScanDurationS: 1000 * common.BlockDurationS,
		NumChannels:   8,
	}
	if _, err := client.StartScan(ctx, req); err != nil {
		t.Fatalf("first StartScan: %v", err)
	}
	defer client.StopScan(ctx, &pb.StopScanRequest{})

	_, err := client.StartScan(ctx, req)
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("second StartScan (while first still running): got err=%v, want FailedPrecondition", err)
	}
}

func TestServer_StartScanRejectsInvalidNumChannels(t *testing.T) {
	client, _, cleanup := newTestClient(t)
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := client.StartScan(ctx, &pb.StartScanRequest{
		ObsTimeEpochS: 1_700_000_000.0,
		ScanDurationS: common.BlockDurationS,
		NumChannels:   7, // not a valid SPS beam configuration
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("got err=%v, want InvalidArgument", err)
	}
}

func TestServer_StartScanRejectsToneSourceWithoutSourceID(t *testing.T) {
	client, _, cleanup := newTestClient(t)
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := client.StartScan(ctx, &pb.StartScanRequest{
		ObsTimeEpochS: 1_700_000_000.0,
		ScanDurationS: common.BlockDurationS,
		NumChannels:   8,
		ToneSources:   []*pb.ToneSourceConfig{{FreqHz: 60e6, Amplitude: 1.0}}, // no SourceId
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("got err=%v, want InvalidArgument", err)
	}
}

func TestServer_PushDelayUpdateForUnknownSourceFails(t *testing.T) {
	client, _, cleanup := newTestClient(t)
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if _, err := client.StartScan(ctx, &pb.StartScanRequest{
		ObsTimeEpochS: 1_700_000_000.0,
		ScanDurationS: 1000 * common.BlockDurationS,
		NumChannels:   8,
		ToneSources:   []*pb.ToneSourceConfig{{SourceId: "tone-a", FreqHz: 60e6, Amplitude: 1.0}},
	}); err != nil {
		t.Fatalf("StartScan: %v", err)
	}
	defer client.StopScan(ctx, &pb.StopScanRequest{})

	_, err := client.PushDelayUpdate(ctx, &pb.PushDelayUpdateRequest{
		SourceId:   "does-not-exist",
		Polynomial: &pb.DelayPolynomial{XypolCoeffsNs: []float64{0.0}},
	})
	if status.Code(err) != codes.NotFound {
		t.Fatalf("got err=%v, want NotFound", err)
	}
}

func TestServer_PushDelayUpdateForKnownSourceSucceeds(t *testing.T) {
	client, _, cleanup := newTestClient(t)
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if _, err := client.StartScan(ctx, &pb.StartScanRequest{
		ObsTimeEpochS: 1_700_000_000.0,
		ScanDurationS: 1000 * common.BlockDurationS,
		NumChannels:   8,
		ToneSources:   []*pb.ToneSourceConfig{{SourceId: "tone-a", FreqHz: 60e6, Amplitude: 1.0}},
	}); err != nil {
		t.Fatalf("StartScan: %v", err)
	}
	defer client.StopScan(ctx, &pb.StopScanRequest{})

	resp, err := client.PushDelayUpdate(ctx, &pb.PushDelayUpdateRequest{
		SourceId: "tone-a",
		Polynomial: &pb.DelayPolynomial{
			StationId:         1,
			StartValiditySec:  1_700_000_000.0,
			ValidityPeriodSec: 100.0,
			XypolCoeffsNs:     []float64{5.0},
			YpolOffsetNs:      1.0,
		},
	})
	if err != nil {
		t.Fatalf("PushDelayUpdate: %v", err)
	}
	if !resp.Ok {
		t.Fatal("PushDelayUpdate response Ok=false")
	}
}
