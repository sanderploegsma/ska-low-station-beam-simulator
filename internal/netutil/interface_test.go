package netutil

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

func TestInterfaceIPv4Addr_ResolvesLoopback(t *testing.T) {
	name := findLoopbackInterface(t)
	ip, err := InterfaceIPv4Addr(name)
	if err != nil {
		t.Fatalf("InterfaceIPv4Addr(%q): %v", name, err)
	}
	if !ip.IsLoopback() {
		t.Fatalf("resolved IP %v for interface %q is not a loopback address", ip, name)
	}
}

func TestInterfaceIPv4Addr_UnknownInterfaceErrors(t *testing.T) {
	if _, err := InterfaceIPv4Addr("this-interface-does-not-exist-xyz"); err == nil {
		t.Fatal("expected an error for a nonexistent interface, got none")
	}
}
