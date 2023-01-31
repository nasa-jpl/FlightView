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

#define TIMEOUT_DURATION 100
#define guaranteedBufferFramesCount (3)

#undef GST_HAS_GRAY

using namespace std::chrono;
struct timeval tval_before, tval_after, tval_result;

using std::cout;
using std::endl;

// This struct is shared among many of the static functions.
typedef struct
{
    GMainLoop *loop;
    GstElement *sourcePipe;
    GstElement *sinkPipe;
    int *currentFrame = 0;
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
              char* interface);

    ~RTPCamera();

    virtual uint16_t* getFrame(CameraModel::camStatusEnum *stat);
    void streamLoop();
    virtual camControlType* getCamControlPtr();
    virtual void setCamControlPtr(camControlType* p);

private:
    bool initialize(); // all setup functions

    char* interface;
    int payload = 90;
    int clockRate = 90000;
    int port;
    int frWidth;
    int frHeight;
    int currentFrame;
    bool haveInitialized = false;

    camControlType *camcontrol = NULL;
    uint16_t *guaranteedBufferFrames[guaranteedBufferFramesCount] = {NULL};

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
