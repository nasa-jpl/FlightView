# RTP Test Server - Command-Line Usage Guide

## Quick Start

```bash
# Compile (if not already done)
clang++ -O3 -march=native server.cpp -o ciduServer

# Run with defaults (1280×328, 225 FPS, 256 packets/frame)
./ciduServer test_data.raw
```

## Command-Line Options

### Synopsis
```
ciduServer [options] <filename>
```

### Required Arguments
- `<filename>` - Raw binary file containing frame data (16-bit pixels)

### Optional Arguments

#### Frame Geometry
```bash
-w, --width <pixels>     # Frame width (default: 1280)
-h, --height <pixels>    # Frame height (default: 328)
```

#### Performance
```bash
-f, --fps <rate>         # Target frame rate (default: 225.0)
-p, --packets <count>    # Packets per frame (default: 256)
```

#### Help
```bash
--help                   # Show help message and exit
```

## Common Usage Examples

### Test Different Instruments

**AVIRIS-III (default):**
```bash
./ciduServer aviris3_data.raw
# Uses: 1280×328, 225 FPS, 256 packets/frame
```

**AVIRIS-III at different rate:**
```bash
./ciduServer -f 200 aviris3_data.raw
```

**CARBO Air:**
```bash
./ciduServer -w 512 -h 2048 -f 125 carbo_data.raw
```

**Custom geometry at high rate:**
```bash
./ciduServer --width 1280 --height 480 --fps 300 custom_data.raw
```

### Performance Testing

**Test maximum throughput:**
```bash
./ciduServer -f 500 test_data.raw
# Monitor for "WARNING: not meeting frame rate" messages
```

**Lower packet count (larger packets):**
```bash
./ciduServer -p 64 test_data.raw
# Requires higher MTU or will show errors
```

**Higher packet count (smaller packets):**
```bash
./ciduServer -p 512 test_data.raw
# More compatible with standard MTU
```

### Network Testing

**Test over localhost:**
```bash
# Default sends to 127.0.0.1:5004
./ciduServer -f 220 test_data.raw
```

**Test over network:**
```bash
# Edit server.cpp line 457 to set target IP:
# servaddr.sin_addr.s_addr = inet_addr("10.0.0.141");
./ciduServer -f 220 test_data.raw
```

## Configuration Output

The server prints detailed configuration on startup:

```
=== RTP Server Configuration ===
Frame geometry:       1280 × 328 pixels
Frame size:           839680 bytes (820.0 KB)
Target frame rate:    225.0 FPS
Frame period:         4444 µs
Packets per frame:    256
Bytes per packet:     3280 bytes (payload)
Packet size:          3292 bytes (with 12-byte header)
Target data rate:     1510.40 Mbps
================================
```

**Key metrics explained:**
- **Frame size**: Total bytes per frame (width × height × 2)
- **Frame period**: Microseconds between frames (1,000,000 / FPS)
- **Packets per frame**: May be adjusted from requested value for even division
- **Bytes per packet**: Payload size (frame size ÷ packets per frame)
- **Packet size**: Total including 12-byte RTP header
- **Target data rate**: Theoretical network throughput in Mbps

## Packet Count Adjustment

The server automatically adjusts packets per frame to ensure even division:

```bash
# Example: 1280×328 frame = 839,680 bytes
./ciduServer -p 250 test_data.raw
# Output: "Packets per frame: 248 (requested: 250, adjusted for even division)"
```

**Why?** Frame size must be evenly divisible by packet count to avoid partial packets.

**Common adjusted values for 1280×328 (839,680 bytes):**
- Requested 250 → Adjusted to 248
- Requested 300 → Adjusted to 256
- Requested 500 → Adjusted to 496

## Performance Tuning

### Frame Rate Guidelines

**Conservative (high reliability):**
```bash
./ciduServer -f 200 test_data.raw   # Plenty of headroom
```

**Aggressive (maximum throughput):**
```bash
./ciduServer -f 400 test_data.raw   # May show underspeed warnings
```

**Find your maximum:**
```bash
# Start at 300 and increase until you see warnings
./ciduServer -f 300 test_data.raw
./ciduServer -f 350 test_data.raw
./ciduServer -f 400 test_data.raw
# Use highest rate without sustained warnings
```

### Packet Count Guidelines

**Smaller packets (better compatibility):**
```bash
./ciduServer -p 512 test_data.raw
# Each packet ~1.6 KB, works with standard 1500-byte MTU
# More packets = more overhead
```

**Larger packets (better efficiency):**
```bash
./ciduServer -p 64 test_data.raw
# Each packet ~13 KB, requires jumbo frames (9000-byte MTU)
# Fewer packets = less overhead
```

**Balanced (default):**
```bash
./ciduServer -p 256 test_data.raw
# Each packet ~3.3 KB, requires ~3500-byte MTU
# Good balance of overhead vs compatibility
```

## Error Messages

### "Error: Invalid FPS"
```bash
./ciduServer -f -100 test_data.raw
# Error: Invalid FPS: -100
```
**Solution:** Use positive FPS value (typically 10-500)

### "Error: No filename specified"
```bash
./ciduServer -f 200 -w 1280
# Error: No filename specified
```
**Solution:** Always provide filename as last argument

### "Error: Unknown option"
```bash
./ciduServer --speed 200 test_data.raw
# Error: Unknown option: --speed
```
**Solution:** Use `--help` to see valid options

### "Error, packetSize: X, Bytes sent: Y"
```
Error, packetSize: 9000, Bytes sent: -1
```
**Solution:** 
- Reduce packets per frame: `-p 512`
- Increase network MTU: `sudo ifconfig en0 mtu 9000`
- Check that interface supports jumbo frames

### "WARNING: not meeting frame rate"
```
WARNING, not meeting frame rate. Effective rate: 180.5 FPS (intended 225.0 FPS)
```
**Solutions:**
1. Reduce target FPS: `-f 200`
2. Ensure compiled with optimizations: `clang++ -O3 -march=native`
3. Close other applications
4. Check CPU usage in Activity Monitor

## Integration Examples

### With FlightView Receiver

**Terminal 1 (Receiver):**
```bash
cd /path/to/FlightView
./liveview --rtp-port 5004 --rtp-interface en0
```

**Terminal 2 (Sender):**
```bash
cd /path/to/FlightView/utils/rtp
./ciduServer -f 220 aviris3_data.raw
```

### Scripted Testing

**Test multiple configurations:**
```bash
#!/bin/bash
for fps in 150 200 225 250 300; do
    echo "Testing at $fps FPS..."
    timeout 10 ./ciduServer -f $fps test_data.raw > results_${fps}fps.log 2>&1
    echo "Done with $fps FPS"
done
```

**Test different geometries:**
```bash
#!/bin/bash
# Test AVIRIS-III
./ciduServer -w 1280 -h 328 -f 225 aviris3.raw &
sleep 30
killall ciduServer

# Test CARBO Air
./ciduServer -w 512 -h 2048 -f 125 carbo.raw &
sleep 30
killall ciduServer
```

## Tips and Best Practices

1. **Always compile with optimizations:** `-O3 -march=native`
2. **Start with default values** and adjust as needed
3. **Monitor underspeed events** to find maximum sustainable rate
4. **Use larger packets** (fewer per frame) when possible for efficiency
5. **Test on target hardware** - performance varies by system
6. **Verify network MTU** matches packet size requirements
7. **Run receiver first** to ensure it's ready when sender starts
8. **Use `--help`** to verify current default values

## System Requirements

**Minimum for 225 FPS (default):**
- CPU: 4-core @ 2.5 GHz
- Network: 1 Gbps (10 Gbps for future 440+ FPS)
- MTU: 3500+ bytes (or use more packets per frame)

**Recommended for 300+ FPS:**
- CPU: 8-core @ 3.0 GHz (M1 or better)
- Network: 10 Gbps
- MTU: 9000 bytes (jumbo frames)

## Version History

- **v2.0**: Added command-line argument parsing
- **v1.1**: Performance optimizations (memcpy, socket buffers)
- **v1.0**: Initial hardcoded configuration

## Support

For issues or questions:
1. Run with `--help` to verify usage
2. Check `SERVER_OPTIMIZATIONS.md` for detailed performance information
3. Monitor system resources (CPU, network, memory)
4. Verify network configuration (MTU, buffers, interface)

---

Date: January 27, 2026
Purpose: Flexible runtime configuration for RTP test server
