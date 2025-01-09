#ifndef STARTUPOPTIONS_H
#define STARTUPOPTIONS_H
#include <QString>

// Make sure when updating this file to also update
// void frameWorker::convertOptions() in frame_worker.cpp
// as well as the struct in takeoptions.h

struct startupOptionsType
{
    bool debug = false;
    bool laggy = false;
    bool flightMode = false;
    bool disableGPS = false;
    bool disableCamera = false;
    bool runStdDevCalculation = true;
    bool dataLocationSet = false;
    QString dataLocation = QString("/data");
    bool gpsIPSet = false;
    QString gpsIP = QString("10.0.0.6");
    bool gpsPortSet = false;
    uint16_t gpsPort = 8111;
    bool deviceFPIEDSet = false;
    bool deviceIHESet = false;
    bool xioCam = false;
    bool heightWidthSet = false;
    //QString *xioDirectory = NULL;
    char* xioDirectoryArray = NULL;
    uint16_t xioHeight;
    uint16_t xioWidth;
    uint16_t height;
    uint16_t width;
    float targetFPS;

    const char* rtpInterface = NULL;
    const char* rtpAddress = NULL;
    bool havertpAddress = false;
    bool havertpInterface = false;
    uint16_t rtpHeight;
    uint16_t rtpWidth;
    int rtpPort = 5004;
    bool rtpCam = false;
    bool rtpNextGen = false;
    bool rtprgb = true;

    bool er2mode = false;
    bool headless = false;
    bool noGPU = false; // experimental
    bool rotate = false;
    bool remapPixels = false;

    bool useSHM = false;

    bool wfPreviewEnabled = false;
    bool wfPreviewContinuousMode = false;
    bool wfPreviewlocationset = false;
    QString wfPreviewLocation = QString("/tmp");
    int wfCompQuality = 75;
};


#endif // STARTUPOPTIONS_H
