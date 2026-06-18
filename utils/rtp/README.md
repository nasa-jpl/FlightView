# RTP Utilities

This directory contains utilities for testing and debugging RTP packet streaming.

## Programs

### server.cpp
Generates and sends RTP packets containing image data at a specified frame rate. Loads image data from a binary file and streams it via RTP/UDP.

**Usage:**
```bash
./server <image_data_file>
```

### pcap-rtp-sender.cpp
Reads RTP packets from a PCAP file and replays them at a configurable rate. Useful for debugging packet parsing issues by replaying captured packets.

**Usage:**
```bash
./pcap-rtp-sender <pcap_file> [target_ip] [port] [packets_per_second]
```

**Arguments:**
- `pcap_file` - Path to PCAP file containing RTP packets (required)
- `target_ip` - IP address to send packets to (default: 127.0.0.1)
- `port` - UDP port number (default: 5004)
- `packets_per_second` - Rate to send packets (default: 1000)

**Examples:**
```bash
# Send packets to localhost at default rate (1000 pps)
./pcap-rtp-sender captured_traffic.pcap

# Send to remote host at custom rate
./pcap-rtp-sender captured_traffic.pcap 10.0.0.141 5004 2000

# Send at slow rate for debugging
./pcap-rtp-sender problematic_packets.pcap 127.0.0.1 5004 10
```

**PCAP File Format:**
- The program expects standard PCAP format (not PCAPNG)
- Packets should contain: Ethernet → IPv4 → UDP → RTP
- VLAN tags are automatically handled if present
- Only UDP packets are extracted; other packets are skipped

**Converting PCAPNG to PCAP:**
If you have a PCAPNG file, convert it first:
```bash
tcpdump -r capture.pcapng -w capture.pcap
# or
editcap capture.pcapng capture.pcap
```

**Capturing RTP Traffic:**
To create a PCAP file for testing:
```bash
# Capture RTP traffic on port 5004
sudo tcpdump -i any udp port 5004 -w rtp_capture.pcap

# Capture specific number of packets
sudo tcpdump -i any udp port 5004 -c 1000 -w rtp_capture.pcap
```

## Building

Compile all programs:
```bash
make
```

Compile individual programs:
```bash
make server
make pcap-rtp-sender
```

Clean build artifacts:
```bash
make clean
```

## Notes

### pcap-rtp-sender Features:
- Automatically extracts RTP packets from UDP payloads in PCAP files
- Displays RTP header information for first and last packets
- Loops through packets continuously
- Real-time statistics (packets/sec, Mbps)
- Minimal packet parsing - sends packets as-is from the PCAP file
- Useful for reproducing packet reading issues without needing the original sender

### Target IP Configuration:
For localhost testing, use `127.0.0.1` (default). For network testing, specify the IP address of the machine running FlightView.

### Packet Rate:
The default rate of 1000 packets/second is conservative. You can increase it based on your frame structure. For example, if you have 256 packets per frame at 30 FPS, you'd need approximately 7680 pps.
