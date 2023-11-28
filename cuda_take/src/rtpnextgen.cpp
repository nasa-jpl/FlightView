#include "rtpnextgen.hpp"

rtpnextgen::rtpnextgen(takeOptionsType opts) {
    this->options = opts;
    LOG << "Starting RTP NextGen camera with width: " << options.rtpWidth << ", height: " << options.rtpHeight
        << ", port: " << options.rtpPort <<", network interface: " << options.rtpInterface << ", multicast-group: " << options.rtpAddress
        << ", rgb mode: " << opts.rtprgb;
    haveInitialized = false;

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
        LOG << "initialize successful";
    } else {
        LOG << "initialize fail";
    }

}

rtpnextgen::~rtpnextgen() {
    destructorRunning = true;
    LOG << "Running RTP NextGen camera destructor.";


    // close socket

    if((rtp.m_nHostSocket != -1 ) && (rtp.m_nHostSocket != 0) ) {
        LOG << "Closing RTP NextGen socket";
        close(rtp.m_nHostSocket);
    }

    // deallocate buffers
    LOG << "Deleting RTP Packet buffer";
    if( rtp.m_pPacketBuffer != nullptr ) {
        delete[] rtp.m_pPacketBuffer;
    }
    LOG << "Done deleting RTP Packet buffer";

    LOG << "Freeing RTP Frame buffer:";
    for(int b =0; b < guaranteedBufferFramesCount_rtpng; b++)
    {
        if(guaranteedBufferFrames[b] != NULL)
        {
            free(guaranteedBufferFrames[b]);
        }
    }
    LOG << "Done freeing RTP Frame buffer";

    LOG << "Done with RTP NextGen destructor";
}

bool rtpnextgen::initialize() {
    LOG << "Starting RTP NextGen init";

    if(haveInitialized)
    {
        LOG << "Warning, running initializing function after initialization...";
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
    frameBufferSizeBytes = frame_width*data_height*sizeof(uint16_t);
    timeoutFrame = (uint16_t*)calloc(frameBufferSizeBytes, 1);
    for(int f = 0; f < guaranteedBufferFramesCount_rtpng; f++)
    {
        guaranteedBufferFrames[f] = (uint16_t*)calloc(frameBufferSizeBytes, 1);
        if(guaranteedBufferFrames[f] == NULL) {
            LOG << "ERROR, cannot allocate memory for RTP NextGen frame buffer.";
            LOG << "Calling abort()";
            abort();
        }
    }

    // Generate static table for buffer position:
    // Write order is: 5,6,7,8,9,0,1,2,3,4 (sequential plus offset)
    // Read order is:  0,1,2,3,4,5,6,7,8,9 (sequential)
    // ftab[10] = {5, 6, 7, 8, 9, 0, 1, 2, 3, 4};
    int halfPt = guaranteedBufferFramesCount_rtpng/2;
    for(int b=0; b < guaranteedBufferFramesCount_rtpng; b++) {
        ftab[b] = (b-halfPt)%guaranteedBufferFramesCount_rtpng;
        LOG << "ftab[" << b << "] = " << ftab[b]; // EHL DEBUG remove later
    }

    rtp.m_uRTPChunkSize = 0;
    rtp.m_uRTPChunkCnt = 0;
    rtp.m_uSequenceNumber = 0;
    rtp.m_uFrameStartSeq = 0;
    rtp.m_bFirstPacket = true;

    // Set up the network listening socket:
    rtp.m_nHostSocket = socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );

    if( rtp.m_nHostSocket == -1 ) {
        LOG << "RTP NextGen Failed to open socket!";
        return false;
    } else {
        LOG << "RTP NextGen socket open success.";
    }
    memset( (void*) &rtp.m_siHost, 0, sizeof(rtp.m_siHost) );
    rtp.m_siHost.sin_family = AF_INET;
    rtp.m_siHost.sin_port = htons(options.rtpPort);
    rtp.m_siHost.sin_addr.s_addr = htonl(INADDR_ANY); // TODO EHL, allow for setting specific addresses to listen on, thereby isolating data to a specific network interface.
    int nBinding = bind( rtp.m_nHostSocket, (const sockaddr*)&rtp.m_siHost, sizeof(rtp.m_siHost) );
    if( nBinding == -1 )
    {
        LOG << "RTP NextGen Failed to bind socket!";
        return false;
    } else {
        LOG << "RTP NextGen bind to socket success.";
    }

    // Prepare buffer:
    currentFrameNumber = 0;
    frameCounter = 0;
    doneFrameNumber = guaranteedBufferFramesCount_rtpng-1; // last position. Data not valid anyway.
    lastFrameDelivered = doneFrameNumber;
    rtp.m_pOutputBuffer = (uint8_t *)guaranteedBufferFrames[0];
    rtp.m_uOutputBufferSize = frameBufferSizeBytes;

    RTPGetNextOutputBuffer( rtp, false );
    rtp.m_bInitOK = true;


    LOG << "Completed RTP NextGen init";
    return true;
}

void rtpnextgen::RTPGetNextOutputBuffer( SRTPData& rtp, bool bLastBufferReady ) {
    // This is called at the very start of the program,
    // and at the completion of each frame.
    // rtp.m_pOutputBuffer holds the completed frame.
    // bLastBufferReady is true when there is a complete frame ready for copy
    // bLastBufferReady is false when we are merely getting ready for that first frame.


    // The frame has been copied already into rtp.m_pOutputBuffer
    // All we have to do here is point the buffer to prepare for the next frame.

    if( bLastBufferReady )
    {
        // At this point, a new complete frame is available.
        doneFrameNumber = currentFrameNumber; // position
        currentFrameNumber = (currentFrameNumber+1) % (guaranteedBufferFramesCount_rtpng);
        rtp.m_pOutputBuffer = (uint8_t *)guaranteedBufferFrames[currentFrameNumber]; // buffer for NEXT frame
        frameCounter++;
    } else {
        LOG << "Error, was asked to get RTP NextGen buffer ready for next frame but frame was not ready!";
    }
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
    if( ppData != nullptr )
    {
        *ppData = pBuffer + 12;
    }
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

    // Receive from network into rtp.m_pPacketBuffer:
    receiveFromWaiting = true; // for debug readout
    ssize_t uRxSize = recvfrom(
                rtp.m_nHostSocket, rtp.m_pPacketBuffer, rtp.m_uPacketBufferSize,
                0, nullptr, nullptr
                );
    receiveFromWaiting = false;

    if( uRxSize == -1 )
    {
        // Handle error or cleanup.
        std::cout << "recvfrom returned -1" << std::endl;
        g_bRunning = false;
        return;
    }
    if( uRxSize == 0) {
        // I believe that since we do not timeout, this should generally not happen.
        LOG << "Received size 0 from UDP socket.";
        g_bRunning = false;
        return;
    }
    bool bMarker = false;
    uint8_t* pData = nullptr;
    size_t uChunkSize = 0;
    uint16_t uSeqNumber = 0;
    bool bPadding = false; bool bExtension = false; uint8_t uCRSCCount = 0;
    uint8_t uPayloadType = 0; uint32_t uTimeStamp = 0; uint32_t uSource = 0; uint8_t uVer = 0;

    // Examine the packet:
    // Essentially from m_pPacketBuffer to the payload, &pData. By reference of course.
    bool bChunkOK = RTPExtract(
                rtp.m_pPacketBuffer, uRxSize, bMarker, &pData, uChunkSize, uSeqNumber,
                uVer, bPadding, bExtension, uCRSCCount, uPayloadType, uTimeStamp, uSource
                );
    if( !bChunkOK )
    {
        LOG << "bad RTP packet!";
        return;
    }
    if( !rtp.m_bFirstPacket )
    {
        uint16_t uNext = rtp.m_uSequenceNumber + 1;
        if( uNext != uSeqNumber )
        {
            LOG << "RTP sequence number error. Got: " << std::dec << uSeqNumber << ", expected: " << uNext;
        }
    }
    rtp.m_bFirstPacket = false;
    rtp.m_uSequenceNumber = uSeqNumber;
    if( rtp.m_uOutputBufferUsed == 0 ) // Make a note of chunk size on first packet of frame so we can data that is missing in the right place
    {
        // First packet
        rtp.m_uRTPChunkSize = uChunkSize;
        rtp.m_uFrameStartSeq = uSeqNumber;
    }
    size_t uChunkIndex;
    if( uSeqNumber >= rtp.m_uFrameStartSeq ) {
        uChunkIndex = uSeqNumber - rtp.m_uFrameStartSeq;
    } else {
        uChunkIndex = 0x10000 - ((size_t)rtp.m_uFrameStartSeq - (size_t)uSeqNumber);
    }
    size_t uOffset = uChunkIndex * rtp.m_uRTPChunkSize;
    if( ( uOffset + uChunkSize ) > rtp.m_uOutputBufferSize ) {
        LOG << "A end of frame marker was missed!";
    } else {
        // VALID data for a frame!! Let's keep it!
        memcpy( rtp.m_pOutputBuffer + uOffset, pData, uChunkSize );
    }
    rtp.m_uRTPChunkCnt++;
    rtp.m_uOutputBufferUsed += uChunkSize;
    if( bMarker ) // EoF (Frame complete)
    {
        LL(2) << "MARK end of frame";
        RTPGetNextOutputBuffer( rtp, true );
        // Reset buffer
        rtp.m_uRTPChunkCnt = 0;
        rtp.m_uOutputBufferUsed = 0;
    }
}

void rtpnextgen::streamLoop() {
    // This will run until we are closing.

    LOG << "Starting RTPPump()";
    volatile uint64_t pumpCount=0;
    g_bRunning = true;
    while(g_bRunning) {
        // Function returns once bytes are received.
        RTPPump(rtp);
        pumpCount++;
    }
    LOG << "Finished RTPPump() with pumpCount = " << pumpCount;
}

uint16_t* rtpnextgen::getFrameWait(unsigned int lastFrameNumber, camStatusEnum *stat) {
    volatile uint64_t tap = 0;
    volatile int lastFrameNumber_local_debug = lastFrameNumber;
    int pos = 0;
    pos = doneFrameNumber; // doneFrameNumber is a number that is the most recent frame finished.
    if(camcontrol->pause)
    {
        *stat = CameraModel::camPaused;
        LL(4) << "RTP NextGen Camera paused";
        return timeoutFrame;
    }
    if(camcontrol->exit)
    {
        *stat = CameraModel::camDone;
        LOG << "Closing down RTP NextGen stream";
        this->g_bRunning = false;
        return timeoutFrame;
    }
    while(lastFrameDelivered==(unsigned int)pos) {
        *stat = camWaiting;
        usleep(NG_FRAME_WAIT_MIN_DELAY_US);
        pos = doneFrameNumber;
    }

    *stat = camPlaying;
    lastFrameDelivered = pos; // keep a copy around
    return guaranteedBufferFrames[pos];
    (void)lastFrameNumber_local_debug;
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

