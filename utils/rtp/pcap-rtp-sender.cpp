// PCAP-based RTP packet sender
// Reads RTP packets from a pcap file and sends them at a specified rate
// Compile:
// clang++ -O3 -march=native pcap-rtp-sender.cpp -o pcap-rtp-sender
// Usage:
// ./pcap-rtp-sender <pcap_file> [target_ip] [port] [packets_per_second]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <chrono>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <bits/stdc++.h>
#else
#include <iostream>
#include <iomanip>
#endif

#include <unistd.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 

#define DEFAULT_PORT 5004
#define DEFAULT_TARGET_IP "127.0.0.1"
#define DEFAULT_PPS 1000  // packets per second

// PCAP file format structures
#pragma pack(push, 1)
struct pcap_file_header {
    uint32_t magic_number;   // 0xa1b2c3d4
    uint16_t version_major;  // 2
    uint16_t version_minor;  // 4
    int32_t  thiszone;       // GMT to local correction
    uint32_t sigfigs;        // accuracy of timestamps
    uint32_t snaplen;        // max length of captured packets
    uint32_t network;        // data link type
};

struct pcap_packet_header {
    uint32_t ts_sec;         // timestamp seconds
    uint32_t ts_usec;        // timestamp microseconds
    uint32_t incl_len;       // number of octets of packet saved
    uint32_t orig_len;       // actual length of packet
};

// Ethernet header
struct ethernet_header {
    uint8_t  dest_mac[6];
    uint8_t  src_mac[6];
    uint16_t ethertype;      // 0x0800 for IPv4
};

// IPv4 header (simplified)
struct ipv4_header {
    uint8_t  version_ihl;    // Version (4 bits) + IHL (4 bits)
    uint8_t  dscp_ecn;       // DSCP (6 bits) + ECN (2 bits)
    uint16_t total_length;
    uint16_t identification;
    uint16_t flags_fragment;
    uint8_t  ttl;
    uint8_t  protocol;       // 17 for UDP
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dest_ip;
};

// UDP header
struct udp_header {
    uint16_t src_port;
    uint16_t dest_port;
    uint16_t length;
    uint16_t checksum;
};
#pragma pack(pop)

// Structure to hold extracted RTP packets
struct rtp_packet {
    uint8_t* data;
    size_t length;
};

// Function to read pcap file and extract RTP packets
std::vector<rtp_packet> read_pcap_file(const char* filename) {
    std::vector<rtp_packet> packets;
    
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open pcap file: %s\n", filename);
        return packets;
    }
    
    // Read pcap file header
    pcap_file_header file_header;
    if (fread(&file_header, sizeof(file_header), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read pcap file header\n");
        fclose(fp);
        return packets;
    }
    
    // Verify magic number
    if (file_header.magic_number != 0xa1b2c3d4 && file_header.magic_number != 0xd4c3b2a1) {
        fprintf(stderr, "Error: Invalid pcap magic number: 0x%08x\n", file_header.magic_number);
        fprintf(stderr, "This might be a pcapng file. Please convert to pcap format.\n");
        fclose(fp);
        return packets;
    }
    
    bool swap_bytes = (file_header.magic_number == 0xd4c3b2a1);
    
    printf("Reading pcap file: %s\n", filename);
    printf("PCAP version: %d.%d\n", file_header.version_major, file_header.version_minor);
    printf("Snaplen: %u bytes\n", file_header.snaplen);
    
    // Read packets
    int packet_count = 0;
    while (!feof(fp)) {
        pcap_packet_header pkt_header;
        
        // Read packet header
        if (fread(&pkt_header, sizeof(pkt_header), 1, fp) != 1) {
            break;  // End of file or error
        }
        
        if (pkt_header.incl_len == 0 || pkt_header.incl_len > file_header.snaplen) {
            fprintf(stderr, "Warning: Invalid packet length: %u\n", pkt_header.incl_len);
            break;
        }
        
        // Allocate buffer for packet data
        uint8_t* packet_data = (uint8_t*)malloc(pkt_header.incl_len);
        if (!packet_data) {
            fprintf(stderr, "Error: Cannot allocate memory for packet\n");
            break;
        }
        
        // Read packet data
        if (fread(packet_data, pkt_header.incl_len, 1, fp) != 1) {
            free(packet_data);
            break;
        }
        
        packet_count++;
        
        // Parse Ethernet header (assuming Ethernet encapsulation)
        if (pkt_header.incl_len < sizeof(ethernet_header)) {
            free(packet_data);
            continue;
        }
        
        ethernet_header* eth = (ethernet_header*)packet_data;
        uint16_t ethertype = ntohs(eth->ethertype);
        
        size_t offset = sizeof(ethernet_header);
        
        // Handle VLAN tag if present (0x8100)
        if (ethertype == 0x8100) {
            offset += 4;  // Skip VLAN tag
            if (offset + 2 > pkt_header.incl_len) {
                free(packet_data);
                continue;
            }
            ethertype = ntohs(*(uint16_t*)(packet_data + offset - 2));
        }
        
        // Check if IPv4 packet
        if (ethertype != 0x0800) {
            free(packet_data);
            continue;
        }
        
        // Parse IPv4 header
        if (offset + sizeof(ipv4_header) > pkt_header.incl_len) {
            free(packet_data);
            continue;
        }
        
        ipv4_header* ip = (ipv4_header*)(packet_data + offset);
        uint8_t ip_header_len = (ip->version_ihl & 0x0F) * 4;
        
        // Check if UDP packet
        if (ip->protocol != 17) {
            free(packet_data);
            continue;
        }
        
        offset += ip_header_len;
        
        // Parse UDP header
        if (offset + sizeof(udp_header) > pkt_header.incl_len) {
            free(packet_data);
            continue;
        }
        
        udp_header* udp = (udp_header*)(packet_data + offset);
        offset += sizeof(udp_header);
        
        // Extract RTP payload (everything after UDP header)
        if (offset < pkt_header.incl_len) {
            size_t rtp_len = pkt_header.incl_len - offset;
            uint8_t* rtp_data = (uint8_t*)malloc(rtp_len);
            
            if (rtp_data) {
                memcpy(rtp_data, packet_data + offset, rtp_len);
                
                rtp_packet pkt;
                pkt.data = rtp_data;
                pkt.length = rtp_len;
                packets.push_back(pkt);
            }
        }
        
        free(packet_data);
    }
    
    fclose(fp);
    
    printf("Successfully extracted %zu RTP packets from pcap file\n", packets.size());
    printf("Total packets in pcap: %d\n", packet_count);
    
    return packets;
}

void print_rtp_header_info(const uint8_t* rtp_data, size_t length) {
    if (length < 12) {
        printf("  RTP packet too short: %zu bytes\n", length);
        return;
    }
    
    uint8_t version = (rtp_data[0] >> 6) & 0x03;
    bool padding = (rtp_data[0] >> 5) & 0x01;
    bool extension = (rtp_data[0] >> 4) & 0x01;
    uint8_t csrc_count = rtp_data[0] & 0x0F;
    bool marker = (rtp_data[1] >> 7) & 0x01;
    uint8_t payload_type = rtp_data[1] & 0x7F;
    uint16_t sequence = (rtp_data[2] << 8) | rtp_data[3];
    uint32_t timestamp = (rtp_data[4] << 24) | (rtp_data[5] << 16) | 
                         (rtp_data[6] << 8) | rtp_data[7];
    uint32_t ssrc = (rtp_data[8] << 24) | (rtp_data[9] << 16) | 
                    (rtp_data[10] << 8) | rtp_data[11];
    
    printf("  RTP Header:\n");
    printf("    Version: %d, Payload Type: %d, Marker: %d\n", version, payload_type, marker);
    printf("    Sequence: %u, Timestamp: %u, SSRC: 0x%08x\n", sequence, timestamp, ssrc);
    printf("    Packet length: %zu bytes (payload: %zu bytes)\n", 
           length, length - 12 - (csrc_count * 4));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <pcap_file> [target_ip] [port] [packets_per_second]\n", argv[0]);
        fprintf(stderr, "  pcap_file          - Path to pcap file containing RTP packets\n");
        fprintf(stderr, "  target_ip          - IP address to send packets to (default: %s)\n", DEFAULT_TARGET_IP);
        fprintf(stderr, "  port               - UDP port number (default: %d)\n", DEFAULT_PORT);
        fprintf(stderr, "  packets_per_second - Rate to send packets (default: %d)\n", DEFAULT_PPS);
        return 1;
    }
    
    const char* pcap_file = argv[1];
    const char* target_ip = (argc > 2) ? argv[2] : DEFAULT_TARGET_IP;
    int port = (argc > 3) ? atoi(argv[3]) : DEFAULT_PORT;
    int pps = (argc > 4) ? atoi(argv[4]) : DEFAULT_PPS;
    
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "Error: Invalid port number: %d\n", port);
        return 1;
    }
    
    if (pps <= 0) {
        fprintf(stderr, "Error: Invalid packets per second: %d\n", pps);
        return 1;
    }
    
    printf("Configuration:\n");
    printf("  PCAP file: %s\n", pcap_file);
    printf("  Target IP: %s\n", target_ip);
    printf("  Port: %d\n", port);
    printf("  Packet rate: %d packets/second\n", pps);
    printf("\n");
    
    // Read pcap file
    std::vector<rtp_packet> packets = read_pcap_file(pcap_file);
    
    if (packets.empty()) {
        fprintf(stderr, "Error: No RTP packets found in pcap file\n");
        return 1;
    }
    
    // Print info about first few packets
    printf("\nFirst packet details:\n");
    print_rtp_header_info(packets[0].data, packets[0].length);
    
    if (packets.size() > 1) {
        printf("\nLast packet details:\n");
        print_rtp_header_info(packets[packets.size()-1].data, packets[packets.size()-1].length);
    }
    
    // Create UDP socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Error: socket creation failed");
        return 1;
    }
    
    // Setup destination address
    struct sockaddr_in dest_addr;
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, target_ip, &dest_addr.sin_addr) <= 0) {
        fprintf(stderr, "Error: Invalid IP address: %s\n", target_ip);
        close(sockfd);
        return 1;
    }
    
    printf("\nStarting packet transmission...\n");
    printf("Press Ctrl+C to stop\n\n");
    
    // Calculate delay between packets (in microseconds)
    int delay_us = 1000000 / pps;
    
    size_t packets_sent = 0;
    size_t bytes_sent = 0;
    size_t packet_index = 0;
    
    auto start_time = std::chrono::steady_clock::now();
    auto last_status_time = start_time;
    
    // Send packets in a loop
    while (true) {
        auto loop_start = std::chrono::steady_clock::now();
        
        // Get current packet (loop through packets)
        rtp_packet& pkt = packets[packet_index];
        
        // Send packet
        ssize_t sent = sendto(sockfd, pkt.data, pkt.length, MSG_DONTWAIT,
                              (const struct sockaddr*)&dest_addr, sizeof(dest_addr));
        
        if (sent < 0) {
            perror("Warning: sendto failed");
        } else if ((size_t)sent != pkt.length) {
            fprintf(stderr, "Warning: Partial send (%zd/%zu bytes)\n", sent, pkt.length);
        } else {
            packets_sent++;
            bytes_sent += sent;
        }
        
        // Move to next packet
        packet_index = (packet_index + 1) % packets.size();
        
        // Print status every second
        auto now = std::chrono::steady_clock::now();
        auto status_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_status_time).count();
        
        if (status_elapsed >= 1000) {
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            double avg_pps = (total_elapsed > 0) ? (double)packets_sent / total_elapsed : 0;
            double avg_mbps = (total_elapsed > 0) ? (bytes_sent * 8.0 / 1000000.0) / total_elapsed : 0;
            
            printf("Sent: %zu packets, %zu bytes (%.2f Mbps, %.1f pps avg)\n",
                   packets_sent, bytes_sent, avg_mbps, avg_pps);
            
            last_status_time = now;
        }
        
        // Sleep to maintain packet rate
        auto loop_end = std::chrono::steady_clock::now();
        auto loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start).count();
        
        if (loop_duration < delay_us) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us - loop_duration));
        }
    }
    
    // Cleanup (never reached in this version, but good practice)
    close(sockfd);
    
    for (auto& pkt : packets) {
        free(pkt.data);
    }
    
    return 0;
}
