#ifndef FLIGHTAPPSTATUSTYPES_H
#define FLIGHTAPPSTATUSTYPES_H

// Default values are to indicate success.
// False success is ok here, false failure will
// lead to a reboot of the electronics

struct flightAppStatus_t {
    uint16_t stat_diskOk = 1;
    uint16_t stat_gpsLinkOk = 1;
    uint16_t stat_gpsReady = 1;
    uint16_t stat_cameraReady = 1;
    uint16_t stat_headerOk = 1;
    uint16_t stat_framesCaptured = 1;
    int fps = 100;
};

#endif // FLIGHTAPPSTATUSTYPES_H
