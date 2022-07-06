#ifndef TAKEOPTIONS_H
#define TAKEOPTIONS_H

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
    uint16_t height;
    uint16_t width;
    float targetFPS = 96.00;
};


#endif // TAKEOPTIONS_H
