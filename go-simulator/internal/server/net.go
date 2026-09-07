package server

import (
	"fmt"
	"net"
)

// interfaceIPv4Addr returns the first IPv4 address assigned to the named
// network interface — e.g. "net1", a Multus-attached secondary NIC whose
// address comes from an IPAM pool at pod start, so unlike dest_ip it
// can't be a fixed config value known ahead of time; it must be resolved
// at runtime from the interface itself.
func interfaceIPv4Addr(name string) (net.IP, error) {
	iface, err := net.InterfaceByName(name)
	if err != nil {
		return nil, fmt.Errorf("looking up network interface %q: %w", name, err)
	}
	addrs, err := iface.Addrs()
	if err != nil {
		return nil, fmt.Errorf("listing addresses on interface %q: %w", name, err)
	}
	for _, addr := range addrs {
		ipNet, ok := addr.(*net.IPNet)
		if !ok {
			continue
		}
		if ip4 := ipNet.IP.To4(); ip4 != nil {
			return ip4, nil
		}
	}
	return nil, fmt.Errorf("interface %q has no IPv4 address assigned", name)
}
