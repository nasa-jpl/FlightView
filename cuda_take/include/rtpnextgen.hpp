#ifndef RTPNESTGEN_HPP
#define RTPNESTGEN_HPP

// System:
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <pthread.h>

#include <cstdint>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <sys/types.h>
#include <ifaddrs.h>

#ifndef IFNAMSIZ
#define IFNAMSIZ (16)
#endif

// cuda_take:
#include "cameramodel.h"
#include "constants.h"
#include "cudalog.h"
#include "takeoptions.h"


#define RTPNG_TIMEOUT_DURATION 100
#define guaranteedBufferFramesCount_rtpng (10)

#define NG_FRAME_WAIT_MIN_DELAY_US (1)
#define MAX_FRAME_WAIT_TAPS (100000)

using namespace std::chrono;

using std::cout;
using std::endl;

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


class rtpnextgen : public CameraModel
{
public:


    rtpnextgen(takeOptionsType opts);

    ~rtpnextgen();

    uint16_t* getFrameWait(unsigned int lastFrameNumber, CameraModel::camStatusEnum *stat);
    uint16_t* getFrame(CameraModel::camStatusEnum *stat);
    void streamLoop(); // This should be its own thread and is effectivly the producer of image data.
    virtual camControlType* getCamControlPtr();
    virtual void setCamControlPtr(camControlType* p);

private:
    bool initialize(); // all setup functions
    std::streambuf *coutbuf;
    takeOptionsType options;
    const char* interface;
    bool rtprgb = true;
    bool firstChunk = true;
    uint32_t sourceNumber = 0;
    uint32_t lastTimeStamp = 0;
    int port;
    int frWidth;
    int frHeight;
    unsigned int currentFrameNumber = 0;
    unsigned int doneFrameNumber = 0;
    unsigned int lastFrameDelivered = 0;
    uint64_t frameCounter = 0;
    bool haveInitialized = false;
    bool loopRunning = false;
    bool destructorRunning = false;

    bool receiveFromWaiting = false;

    camControlType *camcontrol = NULL;
    uint16_t *guaranteedBufferFrames[guaranteedBufferFramesCount_rtpng] = {NULL};
    uint16_t *timeoutFrame = NULL;
    int ftab[guaranteedBufferFramesCount_rtpng] = {0};
    size_t frameBufferSizeBytes = 0;
    SRTPData rtp;
    bool g_bRunning = false;
    void RTPGetNextOutputBuffer( SRTPData& rtp, bool bLastBufferReady );
    void RTPPump( SRTPData& rtp );
    bool RTPExtract( uint8_t* pBuffer, size_t uSize, bool& bMarker, uint8_t** ppData, size_t& uChunkSize, uint16_t& uSeqNumber,
        uint8_t &uVer, bool& bPadding, bool& bExtension, uint8_t& uCRSCCount, uint8_t& uPayloadType, uint32_t& uTimeStamp, uint32_t& uSource
        );

    bool getIfAddr(const char* ifString, in_addr *addr);

    void debugMessage(const char* msg);
    void debugMessage(const std::string msg);
};











#endif
