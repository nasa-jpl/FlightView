#ifndef TAKEOPTIONS_H
#define TAKEOPTIONS_H

#include <string>

// This struct is similar to the one in startupOptions.h,
// but it does not require any qt libraries.
// It is used by take_object and the cuda back-end.
// Not all attributes are currently used.
//
// The attributes are copied inside frame_worker.cpp
//

struct takeOptionsType
{
    bool debug = false;
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

    const char* rtpInterface;
    const char* rtpAddress;
    uint16_t rtpHeight;
    uint16_t rtpWidth;
    int rtpPort = 5004;
    bool rtpCam = false;

    uint16_t height;
    uint16_t width;
    float targetFPS = 100.00;
    bool xioDirSet = false;
    std::string *xioDirectory = NULL;
};


#endif // TAKEOPTIONS_H
