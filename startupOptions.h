#ifndef STARTUPOPTIONS_H
#define STARTUPOPTIONS_H
#include <QString>

struct startupOptionsType
{
    bool debug = false;
    bool flightMode = false;
    bool disableGPS = false;
    bool disableCamera = false;
    bool dataLocationSet = false;
    QString dataLocation = QString("/data");
    bool gpsIPSet = false;
    QString gpsIP = QString("10.0.0.6");
    bool deviceFPIEDSet = false;
    QString deviceFPIED = QString("/dev/fpied");
    bool deviceIHESet = false;
    QString deviceIHE = QString("/dev/ihe");

};


#endif // STARTUPOPTIONS_H
