#!/bin/bash
# optimize_macos_rtp.sh
# Optimize macOS network parameters for high-throughput RTP testing

echo "================================================"
echo "macOS RTP Testing Optimization Script"
echo "================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: This script must be run with sudo"
    echo "Usage: sudo ./optimize_macos_rtp.sh"
    exit 1
fi

# Save original values for reference
echo "Current system values:"
echo "---------------------"
ORIG_SENDSPACE=$(sysctl -n net.inet.udp.sendspace 2>/dev/null)
ORIG_RECVSPACE=$(sysctl -n net.inet.udp.recvspace 2>/dev/null)
ORIG_MAXSOCKBUF=$(sysctl -n kern.ipc.maxsockbuf 2>/dev/null)
ORIG_SOMAXCONN=$(sysctl -n kern.ipc.somaxconn 2>/dev/null)

# Fallback to defaults if capture failed
[ -z "$ORIG_SENDSPACE" ] && ORIG_SENDSPACE="9216"
[ -z "$ORIG_RECVSPACE" ] && ORIG_RECVSPACE="42080"
[ -z "$ORIG_MAXSOCKBUF" ] && ORIG_MAXSOCKBUF="8388608"
[ -z "$ORIG_SOMAXCONN" ] && ORIG_SOMAXCONN="128"

echo "UDP send buffer:      $ORIG_SENDSPACE bytes"
echo "UDP receive buffer:   $ORIG_RECVSPACE bytes"
echo "Max socket buffer:    $ORIG_MAXSOCKBUF bytes"
echo "Socket conn backlog:  $ORIG_SOMAXCONN"
echo ""

# Ask for confirmation
echo "This script will increase network buffer sizes for better RTP performance."
echo "Changes are temporary and will reset on reboot."
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Applying optimizations..."
echo "-------------------------"

# Set optimized values
# UDP send buffer: 4MB (from ~9KB default)
sysctl -w net.inet.udp.sendspace=4194304
if [ $? -eq 0 ]; then
    echo "✓ UDP send buffer increased to 4MB"
else
    echo "✗ Failed to set UDP send buffer"
fi

# UDP receive buffer: 8MB (from ~42KB default)
sysctl -w net.inet.udp.recvspace=8388608
if [ $? -eq 0 ]; then
    echo "✓ UDP receive buffer increased to 8MB"
else
    echo "✗ Failed to set UDP receive buffer"
fi

# Max socket buffer: 16MB
sysctl -w kern.ipc.maxsockbuf=16777216
if [ $? -eq 0 ]; then
    echo "✓ Max socket buffer increased to 16MB"
else
    echo "✗ Failed to set max socket buffer"
fi

# Socket connection backlog: 2048
sysctl -w kern.ipc.somaxconn=2048
if [ $? -eq 0 ]; then
    echo "✓ Socket backlog increased to 2048"
else
    echo "✗ Failed to set socket backlog"
fi

# Network memory clusters: 65536
sysctl -w kern.ipc.nmbclusters=65536 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Network memory clusters increased"
else
    echo "⚠ Could not increase network memory clusters (may already be optimal)"
fi

echo ""
echo "New values:"
echo "-----------"
sysctl net.inet.udp.sendspace
sysctl net.inet.udp.recvspace
sysctl kern.ipc.maxsockbuf
sysctl kern.ipc.somaxconn

echo ""
echo "================================================"
echo "Optimization complete!"
echo "================================================"
echo ""
echo "You can now test RTP at higher frame rates."
echo ""
echo "To restore original values:"
echo "  sudo sysctl -w net.inet.udp.sendspace=$ORIG_SENDSPACE"
echo "  sudo sysctl -w net.inet.udp.recvspace=$ORIG_RECVSPACE"
echo "  sudo sysctl -w kern.ipc.maxsockbuf=$ORIG_MAXSOCKBUF"
echo "  sudo sysctl -w kern.ipc.somaxconn=$ORIG_SOMAXCONN"
echo ""
echo "Or simply reboot your system (changes are temporary)."
echo ""
echo "To make changes permanent, add to /etc/sysctl.conf:"
echo "  net.inet.udp.sendspace=4194304"
echo "  net.inet.udp.recvspace=8388608"
echo "  kern.ipc.maxsockbuf=16777216"
echo "  kern.ipc.somaxconn=2048"
echo ""
