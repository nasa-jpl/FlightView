# macOS Network Performance Tuning for RTP Testing

## Problem
macOS loopback (lo0) performance is often lower than Linux, especially for high-throughput UDP applications. This affects localhost testing between the RTP server and FlightView receiver.

## Root Causes

1. **Smaller default socket buffers** - macOS defaults are more conservative
2. **Different scheduling** - macOS kernel handles loopback differently than Linux
3. **Context switching overhead** - More expensive on macOS
4. **Memory copy behavior** - Loopback may copy more than necessary
5. **UDP receive buffer limits** - System-wide limits may be lower

---

## Immediate Optimizations

### 1. Increase Socket Buffer Limits

**Check current values:**
```bash
# Send buffer max
sysctl net.inet.udp.sendspace

# Receive buffer max  
sysctl net.inet.udp.recvspace

# General socket buffer max
sysctl kern.ipc.maxsockbuf
```

**Increase limits (temporary, until reboot):**
```bash
# Increase UDP send buffer (default: 9216 bytes)
sudo sysctl -w net.inet.udp.sendspace=4194304    # 4MB

# Increase UDP receive buffer (default: 42080 bytes)
sudo sysctl -w net.inet.udp.recvspace=8388608    # 8MB

# Increase max socket buffer (default: varies)
sudo sysctl -w kern.ipc.maxsockbuf=16777216      # 16MB

# Increase socket buffer high water mark
sudo sysctl -w kern.ipc.somaxconn=2048           # Default: 128
```

**Make permanent (survives reboot):**
```bash
# Create or edit /etc/sysctl.conf
sudo nano /etc/sysctl.conf

# Add these lines:
net.inet.udp.sendspace=4194304
net.inet.udp.recvspace=8388608
kern.ipc.maxsockbuf=16777216
kern.ipc.somaxconn=2048
```

### 2. Increase Network Memory Buffers

```bash
# Check current network memory limits
sysctl kern.ipc.nmbclusters
sysctl kern.ipc.maxsockets

# Increase if needed (temporary)
sudo sysctl -w kern.ipc.nmbclusters=65536        # Network memory clusters
sudo sysctl -w kern.ipc.maxsockets=524288        # Max sockets

# Make permanent by adding to /etc/sysctl.conf
```

### 3. Loopback-Specific Tuning

```bash
# Check loopback MTU
ifconfig lo0 | grep mtu

# Note: macOS loopback MTU is typically 16384 and cannot be changed
# This is actually good - larger than Linux's 65536 for loopback

# Check loopback queue length
netstat -I lo0 -b
```

---

## Application-Level Optimizations

### 1. Update Server Socket Options

Your server already sets `SO_SNDBUF`, but ensure it's actually applied:

```cpp
// In server.cpp, verify the actual size set
int send_buffer_size = 16 * 1024 * 1024;
if(setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &send_buffer_size, sizeof(send_buffer_size)) < 0) {
    perror("setsockopt SO_SNDBUF failed");
}

// IMPORTANT: macOS may silently cap this to system limits
int actual_size = 0;
socklen_t optlen = sizeof(actual_size);
getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &actual_size, &optlen);
printf("Send buffer: requested %d, got %d\n", send_buffer_size, actual_size);
```

### 2. Update Receiver Socket Options

In `rtpnextgen.cpp`, the receiver already sets `SO_RCVBUF`. Verify it's effective:

```cpp
// After setting SO_RCVBUF, verify:
int actual_size = 0;
socklen_t optlen = sizeof(actual_size);
getsockopt(rtp.m_nHostSocket, SOL_SOCKET, SO_RCVBUF, &actual_size, &optlen);
LOG << "Receive buffer: requested " << recv_buffer_size << ", got " << actual_size;
```

### 3. Adjust Sender Pacing

macOS loopback can be overwhelmed by burst sending. Add microsleep between packets:

**In server.cpp, modify the packet send loop:**
```cpp
// After sendto(), add tiny delay for macOS
#ifdef __APPLE__
if(c > 0 && (c % 32) == 0) {
    // Brief yield every 32 packets to let receiver catch up
    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
}
#endif
```

This reduces burst pressure on loopback without significantly impacting throughput.

---

## Testing and Verification

### 1. Monitor Socket Buffer Usage

**Create monitoring script:**
```bash
#!/bin/bash
# monitor_rtp.sh
while true; do
    echo "=== $(date) ==="
    netstat -s -p udp | grep -E "datagram|dropped|full"
    echo ""
    sleep 2
done
```

Run this while testing:
```bash
chmod +x monitor_rtp.sh
./monitor_rtp.sh
```

Look for:
- `dropped due to full socket buffers` - Indicates buffer overflow
- `dropped due to no socket` - Indicates receiver not ready

### 2. Check for Packet Loss

```bash
# Monitor UDP statistics
netstat -s -p udp

# Look for:
# - "datagrams dropped due to full socket buffers"
# - "datagrams dropped due to no socket"

# Reset and monitor:
# (Note: Can't reset on macOS easily, compare before/after)
```

### 3. Test Actual Throughput

```bash
# Terminal 1: Start receiver
cd /path/to/FlightView
./liveview --rtp-port 5004

# Terminal 2: Monitor system
top -o cpu

# Terminal 3: Run sender with incrementing rates
cd /path/to/FlightView/utils/rtp
./ciduServer -f 150 test_data.raw   # Should work fine
./ciduServer -f 200 test_data.raw   # Test
./ciduServer -f 250 test_data.raw   # May show issues
```

---

## Advanced Optimizations

### 1. CPU Affinity (Limited on macOS)

macOS doesn't expose CPU affinity APIs like Linux, but you can:

```bash
# Run sender and receiver on performance cores (M1/M2)
# Use taskpolicy to set QoS class

# Run sender with high priority
sudo taskpolicy -b 0x00000001 ./ciduServer -f 220 data.raw

# Run receiver with high priority  
sudo taskpolicy -b 0x00000001 ./liveview --rtp-port 5004
```

### 2. Disable Energy Saving

```bash
# Prevent sleep during testing
caffeinate -i ./ciduServer -f 220 data.raw

# Or in separate terminal:
caffeinate -d -i -s &  # Prevents display sleep, idle sleep, system sleep
```

### 3. Reduce System Load

```bash
# Close unnecessary apps
# Disable Spotlight indexing temporarily:
sudo mdutil -a -i off

# Re-enable after testing:
sudo mdutil -a -i on

# Check what's using CPU:
top -o cpu -n 10
```

### 4. Use Kernel Extension (Advanced, Not Recommended)

For maximum performance, some options require kernel extensions which are:
- Difficult to implement on modern macOS (SIP restrictions)
- Not recommended for testing
- Better to test on actual Linux hardware for production validation

---

## Comparison: macOS vs Linux

### Why Linux is Faster for Loopback

| Feature | Linux | macOS |
|---------|-------|-------|
| Default UDP recv buffer | 212,992 bytes | 42,080 bytes |
| Default UDP send buffer | 212,992 bytes | 9,216 bytes |
| Max socket buffer | 134 MB+ | 8 MB (typical) |
| Loopback optimization | Extensive | Moderate |
| CPU scheduling | More aggressive | More conservative |
| Context switch cost | Lower | Higher |

### Realistic Expectations

**Linux localhost:**
- 300-500 FPS achievable at 1280×328
- Can sustain 2-3 Gbps UDP on loopback

**macOS localhost:**
- 200-300 FPS achievable at 1280×328  
- Can sustain 1-2 Gbps UDP on loopback
- M1/M2 Macs better than Intel Macs

---

## Recommended Configuration

### Complete Setup Script for macOS

```bash
#!/bin/bash
# optimize_macos_rtp.sh

echo "Optimizing macOS for RTP testing..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

# Save original values
echo "Saving original values..."
ORIG_SENDSPACE=$(sysctl -n net.inet.udp.sendspace)
ORIG_RECVSPACE=$(sysctl -n net.inet.udp.recvspace)
ORIG_MAXSOCKBUF=$(sysctl -n kern.ipc.maxsockbuf)

echo "Original UDP sendspace: $ORIG_SENDSPACE"
echo "Original UDP recvspace: $ORIG_RECVSPACE"
echo "Original max sockbuf: $ORIG_MAXSOCKBUF"

# Set optimized values
echo ""
echo "Setting optimized values..."

sysctl -w net.inet.udp.sendspace=4194304
sysctl -w net.inet.udp.recvspace=8388608
sysctl -w kern.ipc.maxsockbuf=16777216
sysctl -w kern.ipc.somaxconn=2048
sysctl -w kern.ipc.nmbclusters=65536

echo ""
echo "Current values:"
sysctl net.inet.udp.sendspace
sysctl net.inet.udp.recvspace
sysctl kern.ipc.maxsockbuf
sysctl kern.ipc.somaxconn

echo ""
echo "Optimization complete!"
echo ""
echo "To restore defaults:"
echo "  sudo sysctl -w net.inet.udp.sendspace=$ORIG_SENDSPACE"
echo "  sudo sysctl -w net.inet.udp.recvspace=$ORIG_RECVSPACE"
echo "  sudo sysctl -w kern.ipc.maxsockbuf=$ORIG_MAXSOCKBUF"
echo ""
echo "Or reboot your system."
```

**Usage:**
```bash
chmod +x optimize_macos_rtp.sh
sudo ./optimize_macos_rtp.sh
```

---

## Troubleshooting

### Issue: Receiver Shows Lag Events

**Symptoms:**
```
WARN, buffer LAG, utilization is 75.0%
```

**Solutions:**
1. Increase receiver buffer: Already optimized in rtpnextgen.cpp
2. Increase system buffer: `sudo sysctl -w net.inet.udp.recvspace=8388608`
3. Reduce sender FPS: Use `-f 200` instead of `-f 250`
4. Add sender pacing: Modify server.cpp as shown above

### Issue: "Socket Buffer Full" in netstat

**Check:**
```bash
netstat -s -p udp | grep "full socket"
```

**Solutions:**
1. Increase `kern.ipc.maxsockbuf`
2. Verify `SO_RCVBUF` is actually applied
3. Process frames faster in receiver
4. Reduce sender burst rate

### Issue: CPU Maxed Out

**Check:**
```bash
top -o cpu -n 5
```

**Solutions:**
1. Compile with `-O3 -march=native`
2. Close other applications
3. Use Performance mode (M1/M2): System Preferences > Battery > Performance
4. Reduce FPS target

### Issue: Inconsistent Performance

**Causes:**
- Thermal throttling
- Background processes
- Spotlight indexing
- Time Machine backup
- iCloud sync

**Solutions:**
```bash
# Disable Spotlight temporarily
sudo mdutil -a -i off

# Check what's running
ps aux | grep -E "mds|backupd|cloudd"

# Monitor temperature (M1/M2)
sudo powermetrics --samplers smc -n 1 | grep -i temp
```

---

## Testing Procedure

### Baseline Test

1. **Apply optimizations:**
   ```bash
   sudo ./optimize_macos_rtp.sh
   ```

2. **Start receiver:**
   ```bash
   ./liveview --rtp-port 5004 --debug
   ```

3. **Monitor in separate terminal:**
   ```bash
   ./monitor_rtp.sh
   ```

4. **Start sender at conservative rate:**
   ```bash
   ./ciduServer -f 200 test_data.raw
   ```

5. **Check receiver logs for lag events**

6. **Gradually increase:**
   ```bash
   ./ciduServer -f 225 test_data.raw
   ./ciduServer -f 250 test_data.raw
   # etc.
   ```

7. **Find maximum sustainable rate** (no lag events)

### Expected Results

**Good performance on macOS:**
- 200 FPS: No lag events, <25% buffer usage
- 225 FPS: Occasional lag, <50% buffer usage
- 250 FPS: Frequent lag, >75% buffer usage

**If significantly worse:**
- Check system buffers applied: `sysctl net.inet.udp`
- Check CPU usage: `top -o cpu`
- Check for background processes
- Try pacing modifications in server.cpp

---

## Alternative: Test Over Real Network

If localhost performance is insufficient, test over actual 10G network:

### Setup

1. **Connect two Macs with 10G Ethernet or Thunderbolt**

2. **Configure static IPs:**
   ```bash
   # Mac 1 (sender):
   sudo ifconfig en0 10.0.0.100 netmask 255.255.255.0
   
   # Mac 2 (receiver):
   sudo ifconfig en0 10.0.0.101 netmask 255.255.255.0
   ```

3. **Test connectivity:**
   ```bash
   ping 10.0.0.101  # From sender
   ```

4. **Increase MTU (if supported):**
   ```bash
   sudo ifconfig en0 mtu 9000
   ```

5. **Run test:**
   ```bash
   # Receiver:
   ./liveview --rtp-port 5004 --rtp-address 10.0.0.101
   
   # Sender (edit server.cpp line 457 first):
   # servaddr.sin_addr.s_addr = inet_addr("10.0.0.101");
   ./ciduServer -f 250 test_data.raw
   ```

**Benefits:**
- Real network path (like production)
- Better performance than loopback
- Tests actual network hardware
- More representative of flight conditions

---

## Summary

**Immediate actions:**
1. Run `sudo ./optimize_macos_rtp.sh` before testing
2. Verify socket buffers are actually increased
3. Test at 200 FPS first, gradually increase
4. Monitor for lag events in receiver
5. Consider real network testing for >250 FPS

**Expected improvements:**
- 30-50% better throughput with tuning
- Reduced lag events
- More stable performance

**Realistic limits:**
- macOS localhost: ~200-250 FPS sustainable
- With all optimizations: ~250-300 FPS possible
- For >300 FPS: Use real network or Linux

The optimizations we've already made to the receiver (`rtpnextgen.cpp`) and sender (`server.cpp`) are solid. The remaining bottleneck is macOS kernel/loopback architecture, which these system-level tunings help but cannot completely eliminate.

Date: January 27, 2026
Purpose: Optimize macOS networking for high-throughput RTP testing
