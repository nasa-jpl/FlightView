#!/bin/bash
# Host tuning for high-rate RTP frame replay.
# NB: Run as root.
#
# Usage: sudo ./network_setup.sh [interface]     (default: lo)

ADAPTER=${1:-lo}

echo "Tuning interface: $ADAPTER"

# Jumbo frames let one RTP packet carry ~6.5 KB instead of ~3.2 KB,
# which halves the syscall and per-packet overhead on both ends.
ip link set dev "$ADAPTER" mtu 9710 2>/dev/null \
    || echo "  note: could not set MTU on $ADAPTER"

# Ring buffer sizing only applies to real NICs; loopback has no rings.
if [ "$ADAPTER" != "lo" ]; then
    ethtool -G "$ADAPTER" rx 4096 tx 4096 2>/dev/null \
        || echo "  note: $ADAPTER does not support ring resizing"
fi

# --- receive side (FlightView) ---
sysctl -w net.core.rmem_max=1073741824
sysctl -w net.core.rmem_default=1073741824

# --- send side (the replay server) ---
# Without these, SO_SNDBUF is silently capped at ~426 KB, which is smaller
# than a single 820 KB frame burst.
sysctl -w net.core.wmem_max=134217728
sysctl -w net.core.wmem_default=134217728

# Deeper backlog so a full-frame burst is not dropped before it is read.
sysctl -w net.core.netdev_max_backlog=30000
sysctl -w net.core.netdev_budget=600

echo
echo "Current values:"
sysctl net.core.rmem_max net.core.wmem_max \
       net.core.rmem_default net.core.wmem_default \
       net.core.netdev_max_backlog net.core.netdev_budget
ip link show "$ADAPTER" | head -1
