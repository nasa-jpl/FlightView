#!/bin/bash
# fix_10g_rtp.sh
# Fix RTP packet loss over 10G fiber by addressing MTU mismatch and buffer limits

echo "================================================"
echo "Fix RTP Packet Loss Over 10G Fiber"
echo "================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: This script must be run with sudo"
    echo "Usage: sudo ./fix_10g_rtp.sh"
    exit 1
fi

# Detect the 10G interface (en8 from your command)
INTERFACE="en8"

echo "Step 1: Fix MTU Mismatch"
echo "------------------------"
echo "Current MTU on $INTERFACE:"
ifconfig $INTERFACE | grep mtu

echo ""
echo "Linux server has MTU=9710, Mac has MTU=9900"
echo "This mismatch causes packet fragmentation and drops."
echo "Setting Mac MTU to 9710 to match Linux..."
ifconfig $INTERFACE mtu 9710
if [ $? -eq 0 ]; then
    echo "✓ MTU set to 9710 on $INTERFACE"
    ifconfig $INTERFACE | grep mtu
else
    echo "✗ Failed to set MTU. Check if $INTERFACE is correct."
    echo "Available interfaces:"
    ifconfig | grep "^[a-z]" | cut -d: -f1
    exit 1
fi

echo ""
echo "Step 2: Increase macOS Network Buffers"
echo "---------------------------------------"

# Save original values
ORIG_SENDSPACE=$(sysctl -n net.inet.udp.sendspace 2>/dev/null)
ORIG_RECVSPACE=$(sysctl -n net.inet.udp.recvspace 2>/dev/null)
ORIG_MAXSOCKBUF=$(sysctl -n kern.ipc.maxsockbuf 2>/dev/null)

echo "Current values:"
echo "  UDP send buffer:    $ORIG_SENDSPACE bytes"
echo "  UDP receive buffer: $ORIG_RECVSPACE bytes"
echo "  Max socket buffer:  $ORIG_MAXSOCKBUF bytes"
echo ""

# Increase UDP send buffer (default ~9KB → 8MB)
sysctl -w net.inet.udp.sendspace=8388608
if [ $? -eq 0 ]; then
    echo "✓ UDP send buffer increased to 8MB"
else
    echo "✗ Failed to set UDP send buffer"
fi

# Increase UDP receive buffer (default ~42KB → 16MB) 
# This is critical - must be >= application SO_RCVBUF setting (16MB)
sysctl -w net.inet.udp.recvspace=16777216
if [ $? -eq 0 ]; then
    echo "✓ UDP receive buffer increased to 16MB"
else
    echo "✗ Failed to set UDP receive buffer"
fi

# Max socket buffer: 32MB (allows app to request 16MB)
sysctl -w kern.ipc.maxsockbuf=33554432
if [ $? -eq 0 ]; then
    echo "✓ Max socket buffer increased to 32MB"
else
    echo "✗ Failed to set max socket buffer"
fi

# Socket connection backlog
sysctl -w kern.ipc.somaxconn=2048
if [ $? -eq 0 ]; then
    echo "✓ Socket backlog increased to 2048"
else
    echo "✗ Failed to set socket backlog"
fi

# Network memory clusters
sysctl -w kern.ipc.nmbclusters=131072 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Network memory clusters increased to 131072"
else
    echo "⚠ Could not increase network memory clusters (may already be optimal)"
fi

echo ""
echo "Step 3: Verify Settings"
echo "-----------------------"
echo "Interface $INTERFACE:"
ifconfig $INTERFACE | grep -E "inet |mtu"
echo ""
echo "Network buffers:"
sysctl net.inet.udp.sendspace
sysctl net.inet.udp.recvspace
sysctl kern.ipc.maxsockbuf
sysctl kern.ipc.somaxconn
echo ""

echo "================================================"
echo "Configuration Complete!"
echo "================================================"
echo ""
echo "IMPORTANT: On Linux server, verify MTU matches:"
echo "  ip link show | grep mtu"
echo "  # Should show: mtu 9710"
echo ""
echo "If Linux MTU is not 9710, set it to match:"
echo "  sudo ip link set <interface> mtu 9710"
echo ""
echo "Now restart FlightView and test:"
echo "  ./liveview-bin --rtpcam --rtpnextgen --rtpheight 328 --rtpwidth 1280 \\"
echo "      --rtpinterface en8 --rtpport 5004 --datastoragelocation /tmp"
echo ""
echo "To restore original values after testing:"
echo "  sudo ifconfig $INTERFACE mtu 9900"
echo "  sudo sysctl -w net.inet.udp.sendspace=$ORIG_SENDSPACE"
echo "  sudo sysctl -w net.inet.udp.recvspace=$ORIG_RECVSPACE"
echo "  sudo sysctl -w kern.ipc.maxsockbuf=$ORIG_MAXSOCKBUF"
echo ""
echo "Or simply reboot (changes are temporary)."
echo ""
