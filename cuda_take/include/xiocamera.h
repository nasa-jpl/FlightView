#ifndef XIOCAMERA_H
#define XIOCAMERA_H

#include <stdlib.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <fstream>
#include <deque>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>

#ifdef FAIL_PLZ

#include <qt5/QtCore/qdebug.h>
#include <qt5/QtCore/QDir>
#include <QTime>

#include <qt5/QtCore/QtConcurrent/QtConcurrent>
#include <QFuture>

#include "lvframe.h"
#endif

#include "osutils.h"
#include "alphanum.hpp"

#include "cameramodel.h"
#include "constants.h"

#include "cudalog.h"

#define TIMEOUT_DURATION 100

using namespace std::chrono;

using std::cout;
using std::endl;

class XIOCamera : public CameraModel
{

public:
    XIOCamera(int frWidth = 640,
              int frHeight = 480,
              int dataHeight = 480);
    ~XIOCamera();

    virtual void setDir(const char *dirname);

    virtual uint16_t* getFrame();
    void readLoop();

private:
    std::string getFname();
    void readFile();

    bool is_reading; // Flag that is true while reading from a directory
    std::ifstream dev_p;
    std::string ifname;
    std::string data_dir;
    std::streampos bufsize;
    const int nFrames;
    size_t framesize;
    const int headsize;
    volatile int dummycounttotal=0;
    volatile int dummyrepeats=0;

    size_t image_no;
    std::vector<std::string> xio_files;
    std::deque< std::vector<uint16_t> > frame_buf;
    std::vector<unsigned char> header;
    std::vector<uint16_t> dummy;
    std::vector<uint16_t> temp_frame;

    // atomic bool vectorLocked = false;
    std::atomic_bool frameVecLocked;
    std::atomic_bool fileListVecLocked;

    //QFuture<void> readLoopFuture;
    int tmoutPeriod;
    unsigned int getFrameCounter = 0;

    void debugMessage(const char* msg);
    void debugMessage(const std::string msg);
};

#endif // XIOCAMERA_H
