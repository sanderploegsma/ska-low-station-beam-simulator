package server

import (
	"net"
	"testing"
)

// findLoopbackInterface returns the name of a loopback interface on this
// system, portably -- the name itself differs by OS ("lo" on Linux,
// "lo0" on macOS/BSD), so this looks it up by flag rather than assuming
// a name.
func findLoopbackInterface(t *testing.T) string {
	t.Helper()
	ifaces, err := net.Interfaces()
	if err != nil {
		t.Fatalf("net.Interfaces(): %v", err)
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagLoopback != 0 {
			return iface.Name
		}
	}
	t.Skip("no loopback interface found on this system")
	return ""
}

func TestServer_StartBindsToSourceInterface(t *testing.T) {
	name := findLoopbackInterface(t)
	srv := NewServer(1, 0, "127.0.0.1", 19999, name)
	if err := srv.Start(); err != nil {
		t.Fatalf("Start() with source_interface=%q: %v", name, err)
	}
	defer srv.Stop()
}

func TestServer_StartFailsForUnknownSourceInterface(t *testing.T) {
	srv := NewServer(1, 0, "127.0.0.1", 19999, "this-interface-does-not-exist-xyz")
	if err := srv.Start(); err == nil {
		t.Fatal("expected Start() to fail for an unknown source_interface, got none")
	}
}
