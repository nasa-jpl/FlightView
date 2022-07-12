#ifndef STARTUPOPTIONS_H
#define STARTUPOPTIONS_H
#include <QString>

struct startupOptionsType
{
    bool debug = false;
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
    QString xioDirectory = QString("/");
    uint16_t xioHeight;
    uint16_t xioWidth;
    uint16_t height;
    uint16_t width;
    float targetFPS;

};


#endif // STARTUPOPTIONS_H
