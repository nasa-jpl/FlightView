// Server side implementation of UDP client-server model 
// Compile:
// clang++ -O3 -march=native server.cpp
// Adjust frame rate and geometry within the source. 
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <thread>

#include <bits/stdc++.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 
   
#define PORT      5004
#define MAXLINE 1024 

// 8E3  = 125 FPS
// 5E3  = 196 FPS
// 4444 = 225 FPS
// 4E3  = 250 FPS (243.5 typically)
// 3333 = 300 FPS (295 typically)
// 3E3  = 330 FPS (328 typically)
// 2500 = 400 FPS (385 typically)
// 2000 = 500 FPS (470 typically)

#define framePeriod_microsec (8E3)
#define packetDelay_ns (1)

#define nFramesToDeliver (100000)

// Frame size must be integer divisible
#define chunksPerFrame_d (32)

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
    buffer[0] = buffer[0] | ( ((int)extension) << 6);
    buffer[0] = buffer[0] | ( payloadType & 0x7f);

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
   
    // copy the header into the packet buffer: 
    size_t pos = 0;
    for(pos=0; pos < 12; pos++) {
        packetBuffer[pos] = header[pos];
    }
    
    size_t offset = bytesFramePerPacket*sequenceNumber;
    // Comment this out to always send 
    // the same frame of garbage data 
    // (much faster) 
    
    for(; pos < 12+bytesFramePerPacket; pos++) {
        packetBuffer[pos] = frameImage[pos-12 + offset];
    }
    
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
    frameImage[1] = (uint8_t)frameCounter&0x00ff;
    frameImage[0] = (uint8_t)frameCounter&0xff00>>8;

    //frameImage[1] = 0xf0;
    //frameImage[0] = 0x00;

    frameImage[2] = 0xff;
    frameImage[3] = 0xff;
    frameImage[4] = 0xff;
    frameImage[5] = 0xff;

    frameImage[6] = 0;
    frameImage[7] = 0;
    frameImage[8] = 0;
    frameImage[9] = 0;

    frameImage[10] = 0xff;
    frameImage[11] = 0xff;
    frameImage[12] = 0xff;
    frameImage[13] = 0xff;
}


int main() { 

    std::chrono::steady_clock::time_point startMaintp;
    std::chrono::steady_clock::time_point begintp;
    std::chrono::steady_clock::time_point endtp;

    uint16_t height = 480;
    uint16_t width = 1280;
    
    printf("Allocating memory for header and frame image. Height = %d, width = %d\n",
            height, width); 
    uint8_t* headerBuffer = (uint8_t *)calloc(12, 1);
    uint8_t* frameImage = (uint8_t *)malloc(height*width*2);
    uint8_t* packetBuffer = (uint8_t*)malloc(12 + (height*width*2));
    printf("\tDone.\n");
    
    printf("Generating test pattern\n");
    genFrame(frameImage, height, width); 
    printf("\tDone.\n");

    int sockfd; 
    struct sockaddr_in servaddr;
       
    // Creating socket file descriptor
    printf("Creating socket file descriptor.\n"); 
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } else {
        printf("\tDone.\n");
    }
       
    memset(&servaddr, 0, sizeof(servaddr)); 
    
    // Filling server information 
    servaddr.sin_family    = AF_INET; // IPv4 
    //servaddr.sin_addr.s_addr = INADDR_ANY; 
    servaddr.sin_addr.s_addr = inet_addr("0.0.0.0");
    servaddr.sin_port = htons(PORT); 
       
    socklen_t len;
   
    len = sizeof(servaddr);
     
    printf("Sending frames...\n");
    size_t bytesSent = 0; 

    unsigned int framesSent = 0;

    unsigned int frameSize = height*width*2; 
    bool marker = false; 
    int chunksPerFrame = chunksPerFrame_d;
    size_t frameBytesPerPacket = frameSize/chunksPerFrame; 
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

    // Higher frame rates are possible
    // if compiled with -O3 -march=native
    // 
    // There is some loss in accuracy due to function
    // return time in the timing function. 
    //
    // 8E3  = 125 FPS
    // 5E3  = 196 FPS
    // 4444 = 225 FPS
    // 4E3  = 250 FPS (243.5 typically)
    // 3333 = 300 FPS (295 typically)
    // 2500 = 400 FPS (385 typically)
    // 2000 = 500 FPS (470 typically)
    int framePeriod = framePeriod_microsec; // microseconds
    int underspeedEvents = 0;
    uintmax_t bytesSentTotal = 0;

    startMaintp = std::chrono::steady_clock::now();
    while(framesSent < nFramesToDeliver) {
        begintp = std::chrono::steady_clock::now();

        // Optional, modify frame to have a moving pattern
        genFrameOffset(frameImage, height, width, framesSent); // slows down loop

        // Mark the frame, in case we save data and look at it later.
        insertFrameHeader(frameImage, framesSent);

        for(int c=0; c < chunksPerFrame; c++) {

            if( (chunks+1) * frameBytesPerPacket == frameSize) {
                 marker = true;
            } else {
                 marker = false;
            }

            buildHeader(headerBuffer, marker, sequenceNumber, ver,
                padding, extension, uCRSCCount, 
                payloadType, timestamp, ssrc);  
                   
            buildPacket(headerBuffer, frameImage, packetBuffer, chunks,
                    frameBytesPerPacket);

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
            std::this_thread::sleep_for(std::chrono::nanoseconds(packetDelay_ns));
        }
        timestamp++;
        framesSent++;
        chunks = 0;
        endtp = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::microseconds>(endtp - begintp).count();
        if(duration < framePeriod) {
            std::this_thread::sleep_for(std::chrono::microseconds(framePeriod-duration));
        } else {
            underspeedEvents++;
            printf("WARNING, not meeting frame rate. Effective rate (this frame): %3.1f FPS. Can't keep up. Event: %d\n",
                    1E6*(1.0/duration), 
                    underspeedEvents); fflush(stdout);
        }

    }
    int duration = std::chrono::duration_cast<std::chrono::microseconds>(endtp - startMaintp).count();

    printf("Number of underspeed events: %d\n", underspeedEvents);
    printf("Average frame rate: %3.3f FPS\n",
        (1E6*(1.0/duration)*framesSent));

    float gigabitsPerSec = 8*bytesSentTotal*(1E6*(1.0/duration))/1024/1024/1024;

    printf("Average Datarate: %0.3f gigabits/sec\n", gigabitsPerSec);
    printf("\n");

    printf("Done.\n");
    return 0; 
}

