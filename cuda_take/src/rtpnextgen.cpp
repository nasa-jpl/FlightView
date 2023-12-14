#include "rtpnextgen.hpp"

rtpnextgen::rtpnextgen(takeOptionsType opts) {
    coutbuf = std::cout.rdbuf(); // grab the std out buffer so that we can enforce its use later

    this->options = opts;
    LOG << "Starting RTP NextGen camera with width: " << options.rtpWidth << ", height: " << options.rtpHeight
        << ", UDP port: " << options.rtpPort;

    if(options.havertpInterface) {
        LOG << "network interface:    " << options.rtpInterface;
    }

    if(options.havertpAddress) {
        LOG << "Address to listen on: " << options.rtpAddress;
        if(options.havertpInterface) {
            LOG << "Note: interface and ip address specified. Will default to specified network interface rather than specified ip address.";
        }
    }

    // ignored LOG << "rgb mode: " << opts.rtprgb;

    haveInitialized = false;
    std::cout.rdbuf(coutbuf);

    this->port = options.rtpPort;
    this->rtprgb = options.rtprgb;
    this->frHeight = options.rtpHeight;
    this->frame_height = frHeight;
    this->data_height = frHeight;
    this->frWidth = options.rtpWidth;
    this->frame_width = options.rtpWidth;

    this->interface = options.rtpInterface;

    if (initialize())
    {
        LOG << "RTP NextGen initialize successful";
    } else {
        LOG << "ERROR, RTP NextGen initialize fail";
    }

    std::cout.rdbuf(coutbuf);

    // TODO, remove
    if(false) {
        // This test is here because sometimes the stdout gets stolen.
        // We need to know when that happens.
        // Test the logging capability:
        // LOG TEST:
        std::cout << "std::cout Testing std out logging capability from rtpnextgen:" << std::endl;
        std::cerr << "std::cerr Testing std err output" << std::endl;
        LOG << "Testing the log.";
        LOG << "Current log level: " << cuda_logginglevel; // Change in cudalog.h
        for(int n=0; n < 10; n++)
        {
            LL(n) << "Testing log level " << n;
        }
        LOG << "Done testing log.";
    }
}

rtpnextgen::~rtpnextgen() {
    destructorRunning = true;
    LL(4) << "Running RTP NextGen camera destructor.";

    // Mark the exit event:
    g_bRunning = false;
    if(camcontrol != NULL)
        camcontrol->exit = true;

    // close socket
    if((rtp.m_nHostSocket != -1 ) && (rtp.m_nHostSocket != 0) ) {
        LL(3) << "Closing RTP NextGen socket";
        int close_rtn = close(rtp.m_nHostSocket);
        if(close_rtn != 0) {
            LOG << "ERROR, closing RTP socket resulted in non-zero return value of " << close_rtn;
        }
    } else {
        LL(5) << "Note, RTP socket was already closed or was null.";
    }

    // deallocate buffers
    LL(3) << "Deleting RTP Packet buffer";
    if( rtp.m_pPacketBuffer != nullptr ) {
        delete[] rtp.m_pPacketBuffer;
    }
    LL(3) << "Done deleting RTP Packet buffer";

    LL(3) << "Freeing RTP Frame buffer:";
    for(int b =0; b < guaranteedBufferFramesCount_rtpng; b++) {
        if(guaranteedBufferFrames[b] != NULL) {
            // EHL TODO: CAREFUL
            free(guaranteedBufferFrames[b]);
        }
    }
    LL(3) << "Done freeing RTP Frame buffer";

    LL(3) << "Freeing RTP LargePacketBuffer:";
    for(int b =0; b < guaranteedBufferFramesCount_rtpng; b++) {
        if(largePacketBuffer[b] != NULL) {
            // EHL TODO: CAREFUL
            free(largePacketBuffer[b]);
        }
    }
    LL(3) << "Done freeing RTP LPB.";


    LL(4) << "Done with RTP NextGen destructor";
}

bool rtpnextgen::initialize() {
    std::cout.rdbuf(coutbuf);
    LL(4) << "Starting RTP NextGen init";

    if(haveInitialized)
    {
        LOG << "Warning, running RTP NextGen initializing function after initialization...";
        // continue for now.
    }

    rtp.m_bInitOK = false;
    rtp.m_uPortNumber = port;
    rtp.m_nHostSocket = -1;

    // m_uPacketBuffer is for the raw packets,
    // which contain fragments of full frames of data.
    rtp.m_uPacketBufferSize = 65527;
    rtp.m_pPacketBuffer = new uint8_t[rtp.m_uPacketBufferSize];

    // m_pOutputBuffer is for holding completed frames.
    // We set it to point at buffer later.
    rtp.m_pOutputBuffer = nullptr;
    rtp.m_uOutputBufferSize = 0;
    rtp.m_uOutputBufferUsed = 0;

    // Allocate frame buffer for completed frames, as well as timeout frame
    LL(5) << "Allocating TF and GBF";
    frameBufferSizeBytes = frame_width*data_height*sizeof(uint16_t);
    timeoutFrame = (uint16_t*)calloc(frameBufferSizeBytes, 1);
    timeoutFrame[0] = 0x0045; timeoutFrame[1] = 0x0084; timeoutFrame[2] = 0x004C;
    for(int f = 0; f < guaranteedBufferFramesCount_rtpng; f++)
    {
        guaranteedBufferFrames[f] = (uint16_t*)calloc(frameBufferSizeBytes, 1);
        if(guaranteedBufferFrames[f] == NULL) {
            LOG << "ERROR, cannot allocate memory for RTP NextGen frame buffer. Asked for " << frameBufferSizeBytes << " bytes.";
            LOG << "ERROR, calling abort(). Program will crash.";
            abort();
        }
    }

    // Allocate large packet buffer, into which packets are received.
    // At this stage in the game, we have no idea how many chunks will
    // be needed per frame, so we make a generous estimate and live with it.
    // We are going to allocate 2x the frame size for the packets.
    // Generally we are using 12 bytes for the packet header, and perhaps 500 packets per frame
    // worst case, so perhaps 600kbyte of overhead is actually needed. Oh well, memory is cheap
    LL(5) << "Allocating LPB";
    for(int f = 0; f < guaranteedBufferFramesCount_rtpng; f++)
    {
        largePacketBuffer[f] = (uint8_t*)calloc(frameBufferSizeBytes*2, 1); // 2x overhead allowed
        if(largePacketBuffer[f] == NULL) {
            LOG << "ERROR, cannot allocate memory for RTP NextGen Large Packet Buffer. Asked for " << frameBufferSizeBytes*2 << " bytes.";
            LOG << "ERROR, calling abort(). Program will crash.";
            abort();
        }
    }

    // Here we prepare the secondary buffer which stores only the size of each packet:
    LL(5) << "Preparing PSB";
    for(int f = 0; f < guaranteedBufferFramesCount_rtpng; f++)
    {
        for(int n = 0; n < 1024; n++) {
            packetSizeBuffer[f][n] = 0;
        }
    }



    rtp.m_uRTPChunkSize = 0;
    rtp.m_uRTPChunkCnt = 0;
    rtp.m_uSequenceNumber = 0;
    rtp.m_uFrameStartSeq = 0;
    rtp.m_bFirstPacket = true;
    firstChunk = true;
    LL(3) << "Setting up socket";
    // Set up the network listening socket:
    rtp.m_nHostSocket = socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );

    if( rtp.m_nHostSocket == -1 ) {
        LOG << "ERROR! RTP NextGen Failed to open socket!";
        return false;
    } else {
        LOG << "RTP NextGen socket open success.";
    }
    memset( (void*) &rtp.m_siHost, 0, sizeof(rtp.m_siHost) );
    rtp.m_siHost.sin_family = AF_INET;
    rtp.m_siHost.sin_port = htons(options.rtpPort);
    struct in_addr addr;

    // Priority goes to the interface name.
    // If the interface cannot be found, then INADDR_ANY is used.
    // If no interface name is supplied, but an ip address to listen from is given,
    // then we attempt to use the ip address supplied. If that ip address fails,
    // then we will fall back to INADDR_ANY.

    if(options.havertpInterface && (options.rtpInterface != NULL)) {
        bool ok = this->getIfAddr(options.rtpInterface, &addr);
        if(ok) {
            rtp.m_siHost.sin_addr.s_addr = addr.s_addr;
            LOG << "Using user-specified RTP network interface.";
        } else {
            LOG << "ERROR, cannot make use of supplied interface name [" << options.rtpInterface << "]. Listening on ALL interfaces as a fallback.";
            rtp.m_siHost.sin_addr.s_addr = htonl(INADDR_ANY);
        }
    } else if(options.havertpAddress && (options.rtpAddress != NULL)) {
        bool success_inet_conv = (bool)inet_aton(options.rtpAddress, &addr);
        if(success_inet_conv) {
            rtp.m_siHost.sin_addr.s_addr = addr.s_addr;
            LL(2) << "Using user-supplied RTP listening address of [" << options.rtpAddress << "], hex: 0x" << std::hex << addr.s_addr;
        } else {
            LOG << "ERROR, cannot make use of supplied interface address [" << options.rtpAddress << "]. Listening on ALL interfaces as a fallback.";
            rtp.m_siHost.sin_addr.s_addr = htonl(INADDR_ANY);
        }
    } else {
        LOG << "WARNING, Listening for RTP traffic on ALL addresses and ALL interfaces.";
        rtp.m_siHost.sin_addr.s_addr = htonl(INADDR_ANY);
    }

    int nBinding = bind( rtp.m_nHostSocket, (const sockaddr*)&rtp.m_siHost, sizeof(rtp.m_siHost) );
    if( nBinding == -1 )
    {
        LOG << "ERROR! RTP NextGen Failed to bind socket!";
        return false;
    } else {
        LOG << "RTP NextGen bind to UDP socket success. Network ready.";
    }

    // Prepare buffer:
    currentFrameNumber = 0;
    frameCounter = 0;
    doneFrameNumber = guaranteedBufferFramesCount_rtpng-1; // last position. Data not valid anyway.
    lastFrameDelivered = doneFrameNumber;
    //rtp.m_pOutputBuffer = (uint8_t *)guaranteedBufferFrames[0];
    rtp.m_uOutputBufferSize = frameBufferSizeBytes;

    // RTPGetNextOutputBuffer( rtp, false );
    rtp.m_bInitOK = true;
    lpbPos = 0;
    psbPos = 0;
    psbFramePos = 0;
    lpbFramePos = 0;

    haveInitialized = true;
    LL(4) << "Completed RTP NextGen init";

    return true;
}

bool rtpnextgen::getIfAddr(const char *ifString, in_addr *addr) {
    // This mostly comes from the man (3) page for getifaddrs
    struct ifaddrs *ifaddr;
    int family;
    char addr_dotted[INET_ADDRSTRLEN];
    bool foundInterface = false;

    if(ifString==NULL) {
        LOG << "ERROR, cannot find RTP network interface with null name.";
        return false;
    }

    if(getifaddrs(&ifaddr)) {
        LOG << "ERROR, cannot determine the addresses of your network interfaces.";
        return false;
    }

    for (struct ifaddrs *ifa = ifaddr; ifa != NULL;
         ifa = ifa->ifa_next)
    {
        if(ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;
        if(family == AF_INET) {

            if( strncmp(ifa->ifa_name, ifString, IFNAMSIZ-1)==0 ) {
                foundInterface = true;
                addr->s_addr =  ((sockaddr_in*)ifa->ifa_addr)->sin_addr.s_addr;
                inet_ntop(AF_INET, addr, addr_dotted, INET_ADDRSTRLEN);
                LOG << "Found user-specified interface " << ifa->ifa_name << " with ipv4 address 0x" << std::hex << addr->s_addr << std::dec << ", which is: " << addr_dotted << " in dotted-decimal form.";
                break; // bump out of for loop, we're done here!
            }
        } else if (family == AF_INET6) {
            LL(2) << "Note, ipv6 not supported yet in RTP NextGen. ifname=[" << ifa->ifa_name << "]";
        } else {
            LL(1) << "Note, don't have handler for interface address family type=" << family;
        }
    }
    if(foundInterface) {
        LL(1) << "Found interface requested [" << ifString << "]";
    } else {
        LOG << "Warning, failed to find address for requested interface [" << ifString << "]";
    }

    freeifaddrs(ifaddr);
    return foundInterface;
}

void rtpnextgen::RTPGetNextOutputBuffer( SRTPData& rtp, bool bLastBufferReady ) {
    // This is called at the completion of each frame.
    // The largePacketBuffer has the frame data,
    // we merely need to note the new buffer position for the next frame,
    // which is also picked up as a "new frame event" by the watching
    // thread when the number changes.

    doneFrameNumber = currentFrameNumber; // position
    currentFrameNumber = (currentFrameNumber+1) % (guaranteedBufferFramesCount_rtpng);
    frameCounter++;
}

bool rtpnextgen::RTPExtract( uint8_t* pBuffer, size_t uSize, bool& bMarker,
                             uint8_t** ppData, size_t& uChunkSize, uint16_t& uSeqNumber,
                             uint8_t &uVer, bool& bPadding, bool& bExtension, uint8_t& uCRSCCount,
                             uint8_t& uPayloadType, uint32_t& uTimeStamp, uint32_t& uSource ) {
    // Examine the pBuffer and extract useful information, including
    // the payload (payload is ppData)

    // No allocations take place here.

    if( uSize < 12 )
    {
        return false;
    }
    uVer = 0x3 & ( pBuffer[0] >> 6 );
    bPadding = ( pBuffer[0] & 0x20 ) != 0;
    bExtension = ( pBuffer[0] & 0x10 ) != 0;
    uCRSCCount = 0xF & pBuffer[0];
    bMarker = ( pBuffer[1] & 0x80 ) != 0;
    uPayloadType = 0x7F & pBuffer[0];
    uSeqNumber = (((uint16_t)pBuffer[2]) << 8 ) | (uint16_t)pBuffer[3];
    uTimeStamp = (((uint32_t)pBuffer[4]) << 24 ) | (((uint32_t)pBuffer[5]) << 16 ) | (((uint32_t)pBuffer[6]) << 8 ) | (uint32_t)pBuffer[7];
    uSource    = (((uint32_t)pBuffer[8]) << 24 ) | (((uint32_t)pBuffer[9]) << 16 ) | (((uint32_t)pBuffer[10]) << 8 ) | (uint32_t)pBuffer[11];
    uChunkSize = uSize - 12;
    // Here, we have assumed that the frame data is the
    // entire packet size, minus the 12 bytes of header.
    // We are thus assuming that uCRSCCount is zero.

//    if( ppData != nullptr )
//    {
//        *ppData = pBuffer + 12;
//    }
    return true;
}

void rtpnextgen::RTPPump(SRTPData& rtp ) {
    // This function is called over and over again.
    // The function requests to receive network data,
    // processes the data to extract the payload,
    // and then memcpys the data to an output buffer.
    //
    // This function will hang here waiting for data without timeout.
    // Thus, it should be watched externally to see what is happening.
    bool bMarker = false;
    std::chrono::steady_clock::time_point starttp;
    std::chrono::steady_clock::time_point endtp;

    if((size_t)(lpbPos+rtp.m_uPacketBufferSize) > frameBufferSizeBytes*2) {
        LOG << "Error, cannot store this much data. Likely the end of frame was missed.";
        // TODO: goto cleanup;
        lpbFramePos = (lpbFramePos+1)%guaranteedBufferFramesCount_rtpng;
        psbFramePos = (psbFramePos+1)%guaranteedBufferFramesCount_rtpng;
        psbPos = 0;
        lpbPos = 0;
        return;
    }


    // Receive from network into rtp.m_pPacketBuffer:
    starttp = std::chrono::steady_clock::now();
    receiveFromWaiting = true; // for debug readout
    // Receive directly into the large packet buffer
    // at an offset:
    ssize_t uRxSize = recvfrom(
                rtp.m_nHostSocket, (void*)(largePacketBuffer[lpbFramePos]+lpbPos), rtp.m_uPacketBufferSize,
                0, nullptr, nullptr
                );
    receiveFromWaiting = false;
    endtp = std::chrono::steady_clock::now();
    frameReceive_microSec[lpbFramePos%guaranteedBufferFramesCount_rtpng] = std::chrono::duration_cast<std::chrono::microseconds>(endtp - starttp).count();


    if( uRxSize == -1 )
    {
        // Handle error or cleanup.
        LOG << "ERROR, Received size -1 from RTP UDP socket.";
        //g_bRunning = false;
        return;
    }
    if( uRxSize == 0) {
        // I believe that since we do not timeout, this should generally not happen.
        LOG << "ERROR, Received size 0 from RTP UDP socket.";
        //g_bRunning = false;
        return;
    }

    // The number of non-zero members at each primary position's sub entries
    // tells how many chunks per frame.
    // The number stored in each tells how large each chunk is.
    packetSizeBuffer[psbFramePos][psbPos] = uRxSize;

    uint8_t* pData = nullptr;
    size_t uChunkSize = 0;
    uint16_t uSeqNumber = 0;
    bool bPadding = false; bool bExtension = false; uint8_t uCRSCCount = 0;
    uint8_t uPayloadType = 0; uint32_t uTimeStamp = 0; uint32_t uSource = 0; uint8_t uVer = 0;

    // Examine the packet:
    // Essentially from m_pPacketBuffer to the payload, &pData. By reference of course.
    bool bChunkOK = RTPExtract(
                largePacketBuffer[lpbFramePos]+lpbPos,  uRxSize, bMarker, &pData, uChunkSize, uSeqNumber,
                uVer, bPadding, bExtension, uCRSCCount, uPayloadType, uTimeStamp, uSource
                );

    lpbPos = lpbPos+uRxSize;
    psbPos++;

    if( !bChunkOK )
    {
        LOG << "ERROR, bad RTP packet!";
        return;
    }
    rtp.m_uSource = uSource;
    rtp.m_timestamp = uTimeStamp;

    // We have to be careful here. The FPIE-D could reboot and we will not be
    // ready for the new source number if it is selected at random.
    // On the other hand, it seems to always be set to zero.
    if(firstChunk) {
        this->sourceNumber = rtp.m_uSource;
        LL(3) << "Message source set to: [" << std::setfill('0') << std::setw(8) << std::right << std::hex << rtp.m_uSource << "]." << std::dec;
        firstChunk = false;
    } else {
        if(rtp.m_uSource != this->sourceNumber) {
            LOG << "Warning, rejecting chunk. Message source does not match. Initial: [" << std::setfill('0') << std::setw(8) << std::right << std::hex << this->sourceNumber << "], this chunk: [" << rtp.m_uSource << "]." << std::dec;
            return;
        }
    }

    if( !rtp.m_bFirstPacket )
    {
        uint16_t uNext = rtp.m_uSequenceNumber + 1;
        if( uNext != uSeqNumber )
        {
            // There is no point notifying when the drop is so large
            // that is is likely simply an indication of a reboot
            if(uNext-uSeqNumber < 600)
                LOG << "ERROR, RTP sequence number. Got: " << std::dec << uSeqNumber << ", expected: " << uNext << ", missed: " << uSeqNumber-uNext << " chunks.";
        }
    }
    rtp.m_bFirstPacket = false;
    rtp.m_uSequenceNumber = uSeqNumber;

    if( rtp.m_uOutputBufferUsed == 0 ) // Make a note of chunk size on first packet of frame so we can data that is missing in the right place
    {
        // First packet of this frame
        rtp.m_uRTPChunkSize = uChunkSize; // size of first packet minus 12 bytes header.
        rtp.m_uFrameStartSeq = uSeqNumber; // sequence number from first packet.
    }
    size_t uChunkIndex;
    if( uSeqNumber >= rtp.m_uFrameStartSeq ) {
        uChunkIndex = uSeqNumber - rtp.m_uFrameStartSeq;
    } else {
        uChunkIndex = 0x10000 - ((size_t)rtp.m_uFrameStartSeq - (size_t)uSeqNumber);
    }
    size_t uOffset = uChunkIndex * rtp.m_uRTPChunkSize;
    // Offset is how far into the frame data we are.
    // The offset must not exceed the size of a frame!
    if( ( uOffset + uChunkSize ) > rtp.m_uOutputBufferSize ) {
        LOG << "An end of frame marker was missed, or the frame being received is larger than expected, or the chunks are not all the same size. Not keeping this chunk: " << rtp.m_uRTPChunkCnt+1;
        LOG << "  Chunk count for last good frame was " << chunksPerFramePrior << ". Size (bytes) of this chunk: " << rtp.m_uRTPChunkSize << ", offset into frame: " << uOffset << ", size of buffer for single frame storage: " << rtp.m_uOutputBufferSize;
        LOG << "  Forcing end-of-frame MARK.";
        if(uRxSize-uRxSizePrior != 0) {
            LOG << "  Size (bytes) of this UDP transaction: " << uRxSize << ", size of prior transaction: " << uRxSizePrior << ". Delta: " << uRxSize-uRxSizePrior;
        }
        // At this point, we've received more data than we expected in this frame. We probably missed a MARK packet,
        // and to recover, we will fake the MARK and take the data at the quantity we expected, and move on.
        // There is no need to memcpy since we're dealing with corrupted data anyway. How would we know where to put it?
        bMarker = true;
        // We also don't need to record the size of this packet since it may be in error anyway.
        // A consequence of this is that the next frame received may be offset by the error,
        // but it will all get straightened out once the next MARK is received.
    } else {
        // VALID data for a frame!! Let's keep it!
        // TODO
        //memcpy( rtp.m_pOutputBuffer + uOffset, pData, uChunkSize );
        uRxSizePrior = uRxSize;
    }
    rtp.m_uRTPChunkCnt++;
    rtp.m_uOutputBufferUsed += uChunkSize;
    if( bMarker ) // EoF (Frame complete)
    {
        if(options.debug) {
            LL(3) << "MARK end of frame #" << frameCounter << ", Buffer used: " << rtp.m_uOutputBufferUsed << " bytes, buffer size allocated: " << rtp.m_uOutputBufferSize << " bytes, chunk count: " << rtp.m_uRTPChunkCnt << " chunks.";
            // This will only work if the packets are at least 40 bytes each.
            // This is because it is a quick hack and looks at the packet buffer, not the actual assembled frame.
            for(int b=0; b < 40; b++) {
                std::cout << std::setfill('0') << std::setw(2) << std::right << std::hex << (int)(largePacketBuffer[lpbFramePos][12+b]) << std::dec << " ";
            }
            std::cout << std::endl;
        }
        chunksPerFramePrior = rtp.m_uRTPChunkCnt;

        if((rtp.m_timestamp < lastTimeStamp) && (lastTimeStamp != 0) && (rtp.m_timestamp != 0)) {
            LOG << "Error, frame timestamp decreased. Prior frame: " << lastTimeStamp << ", this frame: " << rtp.m_timestamp << ", keeping anyway.";
            // keep the frame anyway.
            // Also, there is a minor issue that the timestamp is only compared from the last packet of the frame versus all packets of a frame, etc.
        }
        lastTimeStamp = rtp.m_timestamp;
        RTPGetNextOutputBuffer( rtp, true ); // This is where the frame is advanced.
        // Reset buffer
        rtp.m_uRTPChunkCnt = 0;
        rtp.m_uOutputBufferUsed = 0;
        // Mark the next spot as zero:
        packetSizeBuffer[psbFramePos][psbPos] = 0; // psbPos has been ++ already.
        // Advance to next slot of large packet buffer, and reset sub index
        lpbFramePos = (lpbFramePos+1)%guaranteedBufferFramesCount_rtpng;
        psbFramePos = (psbFramePos+1)%guaranteedBufferFramesCount_rtpng;

        psbPos = 0;
        lpbPos = 0;
    }
}

void rtpnextgen::debugFrame(uint8_t *buf, int start, int end) {
    LOG << "Debugging frame from " << start << " to " << end;
    unsigned char c = 0;
    int ccount=0;
    for(int b=start; b < end; b++) {
        c = buf[b];
        std::cout << std::setfill('0') << std::setw(2) << std::right << std::hex << (int)c << std::dec << " ";
        ccount++;
        if( (ccount % 40) == 0) {
            std::cout << std::endl;
        }
    }

    return;
}

bool rtpnextgen::buildFrameFromPackets(int pos) {

    // For each chunk stored,
    // there are 12 bytes of header
    // and N bytes of frame, where N
    // is the packetSize, as stored in packetSizeBuffer,
    // minus the 12 bytes of header.
    // Indeed, we are not even needing to read the header.

    // The data are stored in largePacketBuffer[pos].
    // we pass the position variable because it is possible
    // that the frame advances significantly while we are here.

    // We must be much faster than 1/FPS to keep up.

    // Keeping some of these volatile for debug purposes.
    std::chrono::steady_clock::time_point starttp;
    std::chrono::steady_clock::time_point endtp;
    volatile size_t frameBytesMoved = 0;
    volatile int chunk = 0;
    int headerOffsetBytes = 12;
    volatile int startOffset = 0;
    starttp = std::chrono::steady_clock::now();

    for(; packetSizeBuffer[pos][chunk] !=0; chunk++) {
        if(packetSizeBuffer[pos][chunk] > 65535) {
            LOG << "ERROR, packet size recorded is too large. Corruption likely. packetSizeBuffer[" << pos << "][" << chunk << "]: " << packetSizeBuffer[pos][chunk];
            return false;
        } else {
            memcpy(((uint8_t *)guaranteedBufferFrames[pos])+frameBytesMoved,
                   largePacketBuffer[pos]+startOffset+headerOffsetBytes,
                   packetSizeBuffer[pos][chunk]-headerOffsetBytes);
        }
        startOffset += packetSizeBuffer[pos][chunk];
        frameBytesMoved += packetSizeBuffer[pos][chunk]-headerOffsetBytes;
    }

    // Capture the time spent copying for benchmark purposes:
    endtp = std::chrono::steady_clock::now();
    durationOfMemoryCopy_microSec[pos] = std::chrono::duration_cast<std::chrono::microseconds>(endtp - starttp).count();
    return true;
}

void rtpnextgen::streamLoop() {
    // This will run until we are closing.

    LL(3) << "Starting RTPPump()";
    volatile uint64_t pumpCount=0;
    g_bRunning = true;
    while(g_bRunning) {
        // Function returns once bytes are received.
        RTPPump(rtp);
        pumpCount++;
        if(camcontrol != NULL) {
            if(camcontrol->exit)
                g_bRunning = false;
            if(camcontrol->pause) {
                // debug opportunity:
                if(options.debug) {
                    for(int n=0; n < guaranteedBufferFramesCount_rtpng; n++) {
                        LOG << "[" << n << "]: " << frameReceive_microSec[n];
                    }
                }
            }
        }

    }
    LL(3) << "Finished RTPPump() with pumpCount = " << pumpCount;
}

uint16_t* rtpnextgen::getFrameWait(unsigned int lastFrameNumber, camStatusEnum *stat) {
    // These volatiles are merely for debug and testing.
    volatile uint64_t tap = 0;
    volatile int lastFrameNumber_local_debug = lastFrameNumber;
    int pos = 0;
    pos = doneFrameNumber; // doneFrameNumber is a number that is the most recent frame finished.
    if(camcontrol->pause)
    {
        *stat = CameraModel::camPaused;
        LL(4) << "RTP NextGen Camera paused";
        LOG << "Frames delivered: " << framesDeliveredCounter << ", frames received: " << frameCounter;
        usleep(1E6);
        return timeoutFrame;
    }
    if(camcontrol->exit)
    {
        *stat = CameraModel::camDone;
        LL(3) << "Closing down RTP NextGen stream";
        this->g_bRunning = false;
        return timeoutFrame;
    }
    while(lastFrameDelivered==(unsigned int)pos) {
        *stat = camWaiting;
        tap++;
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
        pos = doneFrameNumber;
        if(camcontrol->exit)
            return timeoutFrame;
    }

    *stat = camPlaying;

    // This is where to reconstruct the frame into the gbf[pos].
    int frameToDeliver = (lastFrameDelivered+1)%guaranteedBufferFramesCount_rtpng;
    bool successBuilding = buildFrameFromPackets(frameToDeliver);
    lastFrameDelivered = frameToDeliver;

    framesDeliveredCounter++;

    if((unsigned int)frameToDeliver != doneFrameNumber) {
        lagEventCounter++;
        if(options.debug)
            LL(4) << "Frame delivery is LAGGING the buffer. Delivering frame " << frameToDeliver << ", fresh ready frame is " << doneFrameNumber << ", frame count: " << framesDeliveredCounter;
    } else {
        if(options.debug)
            LL(4) << "Frame delivery is sync w/ the buffer. Delivering frame " << frameToDeliver << ", fresh ready frame is " << doneFrameNumber << ", frame count: " << framesDeliveredCounter;
    }

    return guaranteedBufferFrames[frameToDeliver];

    (void)lastFrameNumber_local_debug;
    (void)successBuilding;
}

uint16_t* rtpnextgen::getFrame(CameraModel::camStatusEnum *stat) {
    // DO NOT USE
    (void)stat;
    LOG << "ERROR, incorrect getFrame function called for RTP stream.";
    return timeoutFrame;
}

camControlType* rtpnextgen::getCamControlPtr()
{
    return this->camcontrol;
}

void rtpnextgen::setCamControlPtr(camControlType* p)
{
    this->camcontrol = p;
}

