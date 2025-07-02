#ifndef TAKEOPTIONS_H
#define TAKEOPTIONS_H

#include <string>

// This struct is similar to the one in startupOptions.h,
// but it does not require any qt libraries.
// It is used by take_object and the cuda back-end.
// Not all attributes are currently used.

// Make sure when updating this file to also update
// void frameWorker::convertOptions() in frame_worker.cpp
// as well as the struct in startupOptions.h

struct takeOptionsType
{
    bool debug = false;
    bool laggy = false;
    bool flightMode = false;
    bool disableGPS = false;
    bool disableCamera = false;
    bool runStdDevCalculation = true;
    bool dataLocationSet = false;
    //QString dataLocation = QString("/data");
    bool gpsIPSet = false;
    //QString gpsIP = QString("10.0.0.6");
    bool gpsPortSet = false;
    uint16_t gpsPort = 8111;
    bool deviceFPIEDSet = false;
    bool deviceIHESet = false;
    bool xioCam = false;
    bool heightWidthSet = false;
    uint16_t xioHeight;
    uint16_t xioWidth;

    const char* rtpInterface = NULL;
    const char* rtpAddress = NULL;
    bool havertpAddress = false;
    bool havertpInterface = false;
    uint16_t rtpHeight;
    uint16_t rtpWidth;
    int rtpPort = 5004;
    bool rtpCam = false;
    bool rtprgb = true;
    bool rtpNextGen = false;

    bool er2mode = false;
    bool headless = false;
    bool noGPU = false; // experimental
    bool rotate = false;
    bool swapSpatialSpectral = false;
    bool remapPixels = false;
    char* instrumentPrefix = NULL;
    bool haveInstrumentPrefix = false;

    bool useSHM = false;

    uint16_t height;
    uint16_t width;
    float targetFPS = 100.00;
    bool xioDirSet = false;
    std::string *xioDirectory = NULL;

    bool theseAreDefault = false;
};


#endif // TAKEOPTIONS_H
