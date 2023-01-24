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

#include <gst/gst.h>
#include <string.h>
#include <glib/gprintf.h>

#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>


// cuda_take:
#include "cameramodel.h"
#include "constants.h"
#include "cudalog.h"

#define TIMEOUT_DURATION 100
#define guaranteedBufferFramesCount (3)

using namespace std::chrono;

using std::cout;
using std::endl;

typedef struct
{
  GMainLoop *loop;
  GstElement *sourcePipe;
  GstElement *sinkPipe;
} ProgramData;




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
    char* interface;
    int payload = 90;
    int clockRate = 90000;
    int port;
    int frWidth;
    int frHeight;
    bool haveInitialized = false;

    camControlType *camcontrol = NULL;
    uint16_t *guaranteedBufferFrames[guaranteedBufferFramesCount] = {NULL};

    // GST:
    void modify_in_data (GstMapInfo * map);
    GstFlowReturn on_new_sample_from_sink(GstElement * elt, ProgramData * data);

    gboolean on_source_message (GstBus * bus, GstMessage * message, ProgramData * data);
    gboolean on_sink_message (GstBus * bus, GstMessage * message, ProgramData *);

    bool initialize(); // all setup functions


    void debugMessage(const char* msg);
    void debugMessage(const std::string msg);
};











#endif
