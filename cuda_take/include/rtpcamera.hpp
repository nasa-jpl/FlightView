#ifndef RTPCAMERA_HPP
#define RTPCAMERA_HPP

// System:
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <chrono>

// GST:
extern "C" {
#include <gst/gst.h>
#include <string.h>
#include <glib/gprintf.h>

#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
}

// cuda_take:
#include "cameramodel.h"
#include "constants.h"
#include "cudalog.h"

// define FPS_MEAS_ACQ

#ifdef FPS_MEAS_ACQ
#include <sys/time.h>
#endif

#define TIMEOUT_DURATION 100
#define guaranteedBufferFramesCount (3)

#undef GST_HAS_GRAY

#define FRAME_WAIT_MIN_DELAY_US (1)
#define MAX_FRAME_WAIT_TAPS (100000)

using namespace std::chrono;

using std::cout;
using std::endl;

// This struct is shared among many of the static functions.
typedef struct
{
    GMainLoop *loop;
    GstElement *sourcePipe;
    GstElement *sinkPipe;
    unsigned int currentFrameNumber = 0; // being written to
    unsigned int doneFrameNumber = 0; // good frame to copy out
    uint64_t frameCounter = 0;
    uint16_t **buffer;
} ProgramData;


static GstFlowReturn on_new_sample_from_sink(GstElement * elt, ProgramData * data);
static gboolean on_source_message (GstBus * bus, GstMessage * message, ProgramData * data);
static void siphonData (GstMapInfo* map, ProgramData *data);


class RTPCamera : public CameraModel
{
public:
    RTPCamera(int frWidth,
              int frHeight,
              int port,
              const char* interface);

    ~RTPCamera();

    uint16_t* getFrameWait(unsigned int lastFrameNumber, CameraModel::camStatusEnum *stat);
    uint16_t* getFrame(CameraModel::camStatusEnum *stat);
    void streamLoop(); // This should be its own thread and is effectivly the producer of image data.
    virtual camControlType* getCamControlPtr();
    virtual void setCamControlPtr(camControlType* p);

private:
    bool initialize(); // all setup functions

    const char* interface;
    int payload = 90;
    int clockRate = 90000;
    int port;
    int frWidth;
    int frHeight;
    unsigned int *currentFrameNumber = 0;
    unsigned int *doneFrameNumber = 0;
    unsigned int lastFrameDelivered = 0;
    uint64_t *frameCounter = 0;
    bool haveInitialized = false;

    camControlType *camcontrol = NULL;
    uint16_t *guaranteedBufferFrames[guaranteedBufferFramesCount] = {NULL};
    uint16_t *timeoutFrame = NULL;

    // GST:
    // move to static void modify_in_data (GstMapInfo * map);
    //static GstFlowReturn on_new_sample_from_sink(GstElement * elt, ProgramData * data);

    // moved to static gboolean on_source_message (GstBus * bus, GstMessage * message, ProgramData * data);
    //gboolean on_sink_message (GstBus * bus, GstMessage * message, ProgramData *);

    // GST Variables:
    GstElement *sourcePipe, *source, *rtp, *appSink;
    GstBus *busSourcePipe;
    GstMessage *msg;
    GstStateChangeReturn ret;
    ProgramData *data = NULL;


    void debugMessage(const char* msg);
    void debugMessage(const std::string msg);
};











#endif
