// Command simulator runs the Go station-beam simulator's gRPC service:
// one process per station pod, mirroring the Python project's one
// Tango-device-server-per-pod deployment model, minus Tango itself (see
// api/simulator.proto's doc comment for the intended split with a
// Tango-facing counterpart process).
package main

import (
	"flag"
	"fmt"
	"log"
	"net"

	"google.golang.org/grpc"

	pb "github.com/skao/station-beam-simulator-go/api/simulatorpb"
	"github.com/skao/station-beam-simulator-go/internal/server"
)

func main() {
	listenAddr := flag.String("listen", ":50051", "gRPC listen address")
	stationID := flag.Int("station-id", 1, "this pod's station ID")
	substationID := flag.Int("substation-id", 0, "this pod's substation ID")
	destIP := flag.String("dest-ip", "127.0.0.1", "CBF SPEAD/UDP destination IP")
	destPort := flag.Int("dest-port", 8000, "CBF SPEAD/UDP destination port")
	sourceInterface := flag.String("spead-interface", "", "network interface to bind the outbound SPEAD/UDP socket to (e.g. net1 for a Multus-attached secondary NIC); empty leaves this to the OS's default route selection")
	numSenders := flag.Int("sender-goroutines", server.DefaultNumSenders, "number of parallel UDP sender sockets/goroutines for outbound SPEAD/UDP")
	sendBatchSize := flag.Int("send-batch-size", server.DefaultSendBatchSize, "max heaps per batched UDP send (uses sendmmsg on Linux)")
	udpSendBufferBytes := flag.Int("udp-send-buffer-bytes", server.DefaultUDPSendBufferBytes, "SO_SNDBUF size for each outbound SPEAD/UDP socket, in bytes (0: leave at OS default)")
	flag.Parse()

	srv := server.NewServer(int32(*stationID), int32(*substationID), *destIP, *destPort, *sourceInterface,
		server.WithNumSenders(*numSenders),
		server.WithSendBatchSize(*sendBatchSize),
		server.WithUDPSendBufferBytes(*udpSendBufferBytes),
	)
	if err := srv.Start(); err != nil {
		log.Fatalf("starting server: %v", err)
	}
	defer srv.Stop()

	lis, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		log.Fatalf("listening on %s: %v", *listenAddr, err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterStationSimulatorServer(grpcServer, srv)

	log.Printf("station-beam-simulator-go listening on %s (station_id=%d, dest=%s:%d)", *listenAddr, *stationID, *destIP, *destPort)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatal(fmt.Errorf("serving gRPC: %w", err))
	}
}
