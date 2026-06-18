// Server side implementation of UDP client-server model 
// Compile:
// clang++ -O3 -march=native server.cpp
// Adjust frame rate and geometry within the source. 
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <thread>

#if defined(__linux__)
#include <bits/stdc++.h>
#else
// <bits/stdc++.h> is a non-standard GCC header not available on macOS
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib> 
#include <cstring>
#endif
#include <unistd.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 
   
#define PORT      5004
#define MAXLINE 1024 

// Frame rate reference (for manual calculation):
// 8E3 µs  = 125 FPS
// 5E3 µs  = 196 FPS
// 4444 µs = 225 FPS
// 4E3 µs  = 250 FPS
// 3333 µs = 300 FPS
// 3E3 µs  = 330 FPS
// 2500 µs = 400 FPS
// 2000 µs = 500 FPS

#define packetDelay_ns (0)  // Removed delay for maximum throughput
#define nFramesToDeliver (100000)

// Default values if not specified on command line
#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 328
#define DEFAULT_FPS 225.0
#define DEFAULT_PACKETS_PER_FRAME 256

struct SRTPData {
    bool	      m_bFirstPacket;
    bool		  m_bInitOK;
    uint16_t	  m_uPortNumber;
    sockaddr_in   m_siHost;
    int			  m_nHostSocket;
    uint8_t*      m_pPacketBuffer;
    ssize_t       m_uPacketBufferSize;
    uint8_t*      m_pOutputBuffer;
    size_t	      m_uOutputBufferSize;
    size_t	      m_uOutputBufferUsed;
    size_t	      m_uRTPChunkSize;
    size_t	      m_uRTPChunkCnt;
    uint16_t	  m_uFrameStartSeq;
    uint16_t      m_uSequenceNumber;
    uint32_t      m_uSource;
    uint32_t      m_timestamp;
};

char * loadFile(char* filename, size_t *length) {
    // Loads a file into memory. Returns pointer to memery.
    // Calling function is responsible to free the memory.
    // The entire file is loaded, be mindful of memory usage!
    unsigned long primarySize = 0;
    char *binBuffer = NULL;

    printf("Opening binary file %s\n", filename);
    FILE *fdPrimary = fopen(filename, "rb");
    if(fdPrimary==NULL) {
        fprintf(stderr, "Error, file is null.\n");
        return NULL;
    }

    fseek(fdPrimary, 0, SEEK_END);
    primarySize = ftell(fdPrimary);
    fseek(fdPrimary, 0, SEEK_SET);

    binBuffer = (char*)calloc(1, primarySize+1);

    if(!binBuffer) {
        fprintf(stderr, "Error, could not allocate for buffer. We might be out of memory.\n");
        fclose(fdPrimary);
        return NULL;
    } else {
        fread(binBuffer, primarySize, 1, fdPrimary);
        printf("Read %lu bytes from binary file.\n", primarySize);
    }

    *length = primarySize;

    fclose(fdPrimary);
    return binBuffer;
}

void buildHeader(uint8_t* buffer, bool isMark, 
        uint16_t sequenceNumber, uint8_t ver, 
        bool padding, bool extension, uint8_t uCRSCCount, 
        uint8_t payloadType, uint32_t timestamp, uint32_t source) {
    // This function assumes the buffer is large enough. 
    // The size needed is 12 bytes. 
    
    // From RFC 3550, section 5.1: 
    // https://datatracker.ietf.org/doc/html/rfc3550#section-5.1
    
    /*
      0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |V=2|P|X|  CC   |M|     PT      |       sequence number         |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                           timestamp                           |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |           synchronization source (SSRC) identifier            |
   +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
   |            contributing source (CSRC) identifiers             |
   |                             ....                              |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   
   Or per 8 bits, like a normal person:
   
   buffer[0] 
   01234567
   VVPXCCCC
   
   buffer[1]
   01234567
   MPPPPPPP (P = PT = Payload Type)
   
   buffer[2], buffer[3]
   01234567 01234567
   [SEQUENCE NUMBER], 16 bits used 
   
   buffer[4,5,6,7]: 32-bit timestamp
   buffer[8,9,10,11]: 32-bit SSRC (synchronization source identifier) 
   
   buffer[12,13,14,15] can contain CSRC if used, but we do not use it here. 
   
   

    // buffer[0]:
    // ver, padding, 
    // extension, uCRSCCount, 
    // payloadType
    
    // 	uCRSCCount = 0xF & pBuffer[0];
    buffer[0] = 0x00;
    buffer[0] = (ver<<6);
    buffer[0] = buffer[0] | ( ((int)padding) << 5);
    buffer[0] = buffer[0] | ( ((int)extension) << 6);
    buffer[0] = buffer[0] | ( payloadType & 0x7f);
    '0b10000000' version bit, 2 << 6
    '0b00100000' = padding bit, 0b1 << 5
    '0b00010000' = extension bit, 0b1 << 6
    '0b00001111' = uCRSCCount ("CC"), 15 << 0
    */

    buffer[0] = 0x00;
    buffer[0] = (ver<<6);
    buffer[0] = buffer[0] | ( ((int)padding) << 5);
    buffer[0] = buffer[0] | ( ((int)extension) << 4);
    buffer[0] = buffer[0] | ( uCRSCCount & 0x0F);

    // buffer[1]:
    // marker
    buffer[1] = (((int)isMark)<<7) & 0x80;
    buffer[1] = buffer[1] | (payloadType&0x7f); 
    /*
    '0b11111111'
    '0b10000000' = marker, 1 <<7 & 0x80
    */
    
    // buffer[2], buffer[3]:
    // sequence number
    // uSeqNumber = (((uint16_t)pBuffer[2]) << 8 ) | (uint16_t)pBuffer[3];

    buffer[3] = (sequenceNumber&0x00ff);
    buffer[2] = (sequenceNumber&0xff00)>>8;
    
    // buffer[4,5,6,7]:
    // timestamp
    buffer[7] = (timestamp&0x000000ff);
    buffer[6] = (timestamp&0x0000ff00)>>8;
    buffer[5] = (timestamp&0x00ff0000)>>16;
    buffer[4] = (timestamp&0xff000000)>>24;

    //buffer[4] = (timestamp&0x000000ff);
    //buffer[5] = (timestamp&0x0000ff00)>>8;
    //buffer[6] = (timestamp&0x00ff0000)>>16;
    //buffer[7] = (timestamp&0xff000000)>>24;
    
    // buffer[8,9,10,11]: 
    // Synchronization Source SSRC Identifier: 
    //buffer[8] =  (source&0x000000ff);
    //buffer[9] =  (source&0x0000ff00)>>8;
    //buffer[10] = (source&0x00ff0000)>>16;
    //buffer[11] = (source&0xff000000)>>24;
    
    buffer[11] =  (source&0x000000ff);
    buffer[10] =  (source&0x0000ff00)>>8;
    buffer[9] = (source&0x00ff0000)>>16;
    buffer[8] = (source&0xff000000)>>24;
    

    if(false) {
        std::cout << "Header buffer contents: " << std::endl;
        for(int b=0; b < 12; b++) {
            std::cout << std::setfill('0') << std::setw(2) << std::right << std::hex << (int)buffer[b] << std::dec << " ";
        }
        std::cout << std::endl;
    }

    return;
}

void buildPacket(uint8_t *header, uint8_t *frameImage,
                uint8_t *packetBuffer, 
                uint16_t sequenceNumber, 
                size_t bytesFramePerPacket) 
                {
   
    // Optimized: Use memcpy instead of byte-by-byte loops
    memcpy(packetBuffer, header, 12);
    
    size_t offset = bytesFramePerPacket*sequenceNumber;
    // Comment this out to always send 
    // the same frame of garbage data 
    // (much faster) 
    
    memcpy(packetBuffer + 12, frameImage + offset, bytesFramePerPacket);
}

void genFrame(uint8_t* buffer, uint16_t height, uint16_t width) {
    for(unsigned int p=0; p < height*width*2; p++) {
        buffer[p] = (uint16_t)p;
    }
}

void genFrameOffset(uint8_t* buffer, uint16_t height, uint16_t width, uint8_t offset) {
    for(unsigned int p=0; p < height*width*2; p++) {
        //buffer[p] = (uint16_t)p + offset; // straight column, moves sideways
        buffer[p] = ((uint8_t)p + offset) + (p/width); // diagional pattern, moves diagionally 
    }
}

void insertFrameHeader(uint8_t* frameImage, unsigned int frameCounter) {
    // Optimized: Use 32/64-bit writes instead of byte-by-byte
    uint16_t* frame16 = (uint16_t*)frameImage;
    frame16[0] = frameCounter; // First 2 bytes
    
    uint32_t* frame32 = (uint32_t*)frameImage;
    frame32[0] = (uint32_t)frameCounter | 0xffff0000; // Combines first operations
    frame32[1] = 0x00000000;
    frame32[2] = 0xffffffff;
}

void printUsage(const char* progName) {
    printf("\n=== RTP Test Server - High-Performance Frame Sender ===\n\n");
    printf("Usage: %s [options] <filename>\n\n", progName);
    printf("Required:\n");
    printf("  <filename>              Raw binary file containing frame data\n\n");
    printf("Options:\n");
    printf("  -w, --width <pixels>    Frame width (default: %d)\n", DEFAULT_WIDTH);
    printf("  -h, --height <pixels>   Frame height (default: %d)\n", DEFAULT_HEIGHT);
    printf("  -f, --fps <rate>        Target frame rate (default: %.1f)\n", DEFAULT_FPS);
    printf("  -p, --packets <count>   Packets per frame (default: %d)\n", DEFAULT_PACKETS_PER_FRAME);
    printf("  --help                  Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s data.raw\n", progName);
    printf("  %s -w 1280 -h 480 -f 200 data.raw\n", progName);
    printf("  %s --width 512 --height 2048 --fps 125 --packets 64 data.raw\n\n", progName);
    printf("Notes:\n");
    printf("  - Packets per frame will be rounded to ensure frame size is evenly divisible\n");
    printf("  - Higher FPS requires faster CPU and optimized compilation (-O3 -march=native)\n");
    printf("  - Frame size = width × height × 2 bytes (16-bit pixels)\n");
    printf("  - Actual FPS may vary slightly due to timing precision\n\n");
}

int main(int argc, char* argv[]) {

    std::chrono::steady_clock::time_point startMaintp;
    std::chrono::steady_clock::time_point begintp;
    std::chrono::steady_clock::time_point endtp;

    // Default values
    uint16_t width = DEFAULT_WIDTH;
    uint16_t height = DEFAULT_HEIGHT;
    double targetFPS = DEFAULT_FPS;
    int desiredPacketsPerFrame = DEFAULT_PACKETS_PER_FRAME;
    const char* filename = nullptr;

    // Parse command line arguments
    int i = 1;
    while(i < argc) {
        if(strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--width") == 0) {
            if(i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
                printUsage(argv[0]);
                return 1;
            }
            width = atoi(argv[i+1]);
            if(width <= 0) {
                fprintf(stderr, "Error: Invalid width: %s\n", argv[i+1]);
                return 1;
            }
            i += 2;
        } else if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--height") == 0) {
            if(i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
                printUsage(argv[0]);
                return 1;
            }
            height = atoi(argv[i+1]);
            if(height <= 0) {
                fprintf(stderr, "Error: Invalid height: %s\n", argv[i+1]);
                return 1;
            }
            i += 2;
        } else if(strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--fps") == 0) {
            if(i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
                printUsage(argv[0]);
                return 1;
            }
            targetFPS = atof(argv[i+1]);
            if(targetFPS <= 0) {
                fprintf(stderr, "Error: Invalid FPS: %s\n", argv[i+1]);
                return 1;
            }
            i += 2;
        } else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--packets") == 0) {
            if(i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
                printUsage(argv[0]);
                return 1;
            }
            desiredPacketsPerFrame = atoi(argv[i+1]);
            if(desiredPacketsPerFrame <= 0) {
                fprintf(stderr, "Error: Invalid packets per frame: %s\n", argv[i+1]);
                return 1;
            }
            i += 2;
        } else if(strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if(argv[i][0] == '-') {
            fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        } else {
            // This is the filename
            filename = argv[i];
            i++;
        }
    }

    if(filename == nullptr) {
        fprintf(stderr, "Error: No filename specified\n");
        printUsage(argv[0]);
        return 1;
    }

    // Calculate frame parameters from parsed arguments
    unsigned int frameSize = width * height * 2; // 16-bit pixels
    
    // Round packets per frame to ensure even division
    int chunksPerFrame = desiredPacketsPerFrame;
    size_t bytesPerPacket = frameSize / chunksPerFrame;
    
    // Adjust to ensure frame is evenly divisible
    while(frameSize % chunksPerFrame != 0 && chunksPerFrame > 0) {
        chunksPerFrame--;
    }
    if(chunksPerFrame <= 0) chunksPerFrame = 1;
    
    bytesPerPacket = frameSize / chunksPerFrame;
    
    // Calculate frame period from desired FPS
    int framePeriod = (int)(1000000.0 / targetFPS); // microseconds
    
    // Print configuration
    printf("\n=== RTP Server Configuration ===\n");
    printf("Frame geometry:       %d × %d pixels\n", width, height);
    printf("Frame size:           %u bytes (%.1f KB)\n", frameSize, frameSize/1024.0);
    printf("Target frame rate:    %.1f FPS\n", targetFPS);
    printf("Frame period:         %d µs\n", framePeriod);
    printf("Packets per frame:    %d", chunksPerFrame);
    if(chunksPerFrame != desiredPacketsPerFrame) {
        printf(" (requested: %d, adjusted for even division)", desiredPacketsPerFrame);
    }
    printf("\n");
    printf("Bytes per packet:     %zu bytes (payload)\n", bytesPerPacket);
    printf("Packet size:          %zu bytes (with 12-byte header)\n", bytesPerPacket + 12);
    printf("Target data rate:     %.2f Mbps\n", (frameSize * 8 * targetFPS) / 1000000.0);
    printf("================================\n\n");

    size_t fileLen = 0;
    printf("Loading file [%s]...\n", filename);
    uint8_t* imageData = (uint8_t *)loadFile((char*)filename, &fileLen);
    if(imageData) {
        printf("Loaded %zu MiB from file into memory.\n", fileLen/1024/1024);
    } else {
        perror("ERROR, could not load file.\n");
        abort();
    }
    
    printf("Allocating memory for header and packet buffer. Height = %d, width = %d\n",
            height, width); 
    uint8_t* headerBuffer = (uint8_t *)calloc(12, 1);
    uint8_t* frameImage;
    uint8_t* packetBuffer = (uint8_t*)malloc(12 + (height*width*2));
    printf("\tDone.\n");
    

    int sockfd; 
    struct sockaddr_in servaddr;
       
    // Creating socket file descriptor
    printf("Creating socket file descriptor.\n"); 
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("ERROR: socket creation failed\n");
        exit(EXIT_FAILURE); 
    } else {
        printf("\tDone.\n");
    }
    
    // Optimize socket for high-throughput sending (Mac and Linux compatible)
    int send_buffer_size = 16 * 1024 * 1024; // 16MB send buffer
    if(setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &send_buffer_size, sizeof(send_buffer_size)) < 0) {
        printf("WARNING: Failed to set socket send buffer size. May limit throughput.\n");
    } else {
        int actual_size = 0;
        socklen_t optlen = sizeof(actual_size);
        if(getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &actual_size, &optlen) == 0) {
            printf("Socket send buffer set to %d bytes (requested: %d).\n", actual_size, send_buffer_size);
        }
    }
       
    memset(&servaddr, 0, sizeof(servaddr)); 
    
    // Filling server information 
    servaddr.sin_family    = AF_INET; // IPv4 
    // IMPORTANT:
    //       The address specified here is the address we are 
    //       sending packets *to*. 
    //       Typically that is the address of the computer
    //       running FlightView. 
    //
    // For localhost testing (FlightView on same machine):
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");  // Use this for localhost on macOS/Linux
    
    // For network testing (FlightView on different machine):
    //servaddr.sin_addr.s_addr = inet_addr("10.0.0.141"); // Change to target IP address
    
    // NOTE: INADDR_ANY (0.0.0.0) is for binding/listening, NOT for sending!
    //       Use explicit IP addresses for sendto() destination.
    
    servaddr.sin_port = htons(PORT); 
       
    socklen_t len;
   
    len = sizeof(servaddr);
    
    // Print connection info
    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(servaddr.sin_addr), ipstr, INET_ADDRSTRLEN);
    printf("Configured to send RTP packets to: %s:%d\n", ipstr, PORT);
    printf("Sending frames...\n");
    size_t bytesSent = 0; 

    unsigned int framesSent = 0;

    // frameSize, chunksPerFrame, and bytesPerPacket already calculated above
    bool marker = false;
    size_t frameBytesPerPacket = bytesPerPacket;
    unsigned int chunksSent = 0;
    unsigned int chunks = 0;
    uint16_t sequenceNumber = 0;
    size_t packetSize = 0;
    uint8_t ver = 2;
    uint8_t payloadType = 1; // ?
    bool padding = false; 
    bool extension = false; 
    uint8_t uCRSCCount = 0;
    uint32_t ssrc = 0xdeadbeef; 
    uint32_t timestamp = 0;

    // framePeriod already calculated from command-line FPS argument
    int underspeedEvents = 0;
    uintmax_t bytesSentTotal = 0;
    size_t offsetIntoFrameData = 0;

    startMaintp = std::chrono::steady_clock::now();

    frameImage = imageData;
    
    // Pre-build the static parts of the header once (huge optimization)
    // These fields don't change packet-to-packet
    buildHeader(headerBuffer, false, 0, ver,
        padding, extension, uCRSCCount, 
        payloadType, 0, ssrc);
    // Note: We'll update marker (byte 1), sequence (bytes 2-3), 
    // and timestamp (bytes 4-7) in the tight loop

    bool keepGoing = true;

    while(keepGoing) {
        begintp = std::chrono::steady_clock::now();

        // Optional, modify frame to have a moving pattern
        //genFrameOffset(frameImage, height, width, framesSent); // slows down loop

        // Mark the frame, in case we save data and look at it later.
        insertFrameHeader(frameImage, framesSent);
        
        // Update timestamp in header buffer once per frame (bytes 4-7)
        headerBuffer[7] = (timestamp&0x000000ff);
        headerBuffer[6] = (timestamp&0x0000ff00)>>8;
        headerBuffer[5] = (timestamp&0x00ff0000)>>16;
        headerBuffer[4] = (timestamp&0xff000000)>>24;
        
        //printf("Sending frame %d\n", framesSent);
        // This loop sends ONE frame of data via chunksPerFrame number of packets.
        for(int c=0; c < chunksPerFrame; c++) {

            if( (chunks+1) * frameBytesPerPacket == frameSize) {
                 marker = true;
            } else {
                 marker = false;
            }

            // Optimized: Build header inline to avoid function call overhead
            // Only rebuild parts that change per packet
            headerBuffer[1] = (marker ? 0x80 : 0x00) | (payloadType&0x7f);
            headerBuffer[3] = (sequenceNumber&0x00ff);
            headerBuffer[2] = (sequenceNumber&0xff00)>>8;
            
            // Optimized packet build with memcpy
            memcpy(packetBuffer, headerBuffer, 12);
            memcpy(packetBuffer + 12, frameImage + (chunks * frameBytesPerPacket), frameBytesPerPacket);

            offsetIntoFrameData += frameBytesPerPacket;

            chunks++;
            packetSize = 12 + frameBytesPerPacket;
            
            bytesSent = sendto(sockfd, (const char *)packetBuffer, packetSize,  
                    MSG_DONTWAIT, (const struct sockaddr *) &servaddr, len);
            if(packetSize != bytesSent) {
                printf("Error, packetSize: %ld, Bytes sent: %ld. Consider increasing the MTU or chunks per frame.\n", packetSize, bytesSent);
            }
            bytesSentTotal += bytesSent;

            chunksSent++;
            sequenceNumber++;
            // Removed sleep - causes severe throughput limitation
            
            // Optional: macOS loopback pacing - reduces burst pressure on lo0
            // Uncomment the next 4 lines if experiencing drops on macOS localhost
            //#ifdef __APPLE__
            //if((c > 0) && ((c % 32) == 0)) {
            //    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            //}
            //#endif
        }

        frameImage += height*width*2; // next frame


        timestamp++;
        framesSent++;
        chunks = 0;
        endtp = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::microseconds>(endtp - begintp).count();
        if(duration < framePeriod) {
            if( (framesSent%200)==0) {
                printf("Sent: %d frames\n", framesSent);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(framePeriod-duration));
        } else {
            underspeedEvents++;
            if(duration > framePeriod*1.5) {
                // Don't print unless we're really slow. 
            printf("WARNING, not meeting frame rate. Effective rate (this frame): %3.1f FPS (intended %3.1f FPS). Can't keep up. Event: %d\n",
                    1E6*(1.0/duration), 1E6*(1.0/framePeriod), 
                    underspeedEvents); fflush(stdout);
        } }
        if(offsetIntoFrameData > (fileLen - (2*height*width))) {
            // If we are "within a frame" of the end, let's just loop now.
            // This prevents crashes at "partial" frame endings
            printf("Looping back around. offset=%zu bytes, filelen=%zu bytes..\n",
                   offsetIntoFrameData, fileLen);
            int duration = std::chrono::duration_cast<std::chrono::microseconds>(endtp - startMaintp).count();
            float gigabitsPerSec = 8*bytesSentTotal*(1E6*(1.0/duration))/1024/1024/1024;

            printf("Average frame rate: %3.3f FPS\n",
                (1E6*(1.0/duration)*framesSent));
            printf("Average Datarate: %0.3f gigabits/sec\n", gigabitsPerSec);

            offsetIntoFrameData = 0;
            frameImage = imageData;
        }


    }
    int duration = std::chrono::duration_cast<std::chrono::microseconds>(endtp - startMaintp).count();

    printf("Number of underspeed events: %d\n", underspeedEvents);
    printf("Average frame rate: %3.3f FPS\n",
        (1E6*(1.0/duration)*framesSent));

    float gigabitsPerSec = 8*bytesSentTotal*(1E6*(1.0/duration))/1024/1024/1024;

    printf("Average Datarate: %0.3f gigabits/sec\n", gigabitsPerSec);
    printf("\n");
    printf("Freeing %zu MiB of memory...\n", fileLen/1024/1024);
    free(imageData);

    printf("Done.\n");
    return 0; 
}

