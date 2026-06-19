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
    bool continuousRecording = false;
    int fps = 100;

    // Frame header health checks (written by backend acquire, read by frontend).
    // "Ok" fields reflect the most recent frame only (true = expected, false = unexpected).
    // "Sticky" fields latch false on the first failure and must be manually reset by the GUI.
    bool fh_magicOk = true;           // Check 1: offset 0x50 == 0xDEAD or 0xBABE
    bool fh_magicOkSticky = true;
    bool fh_frameCountOk = true;      // Check 2: frame count increments by 1 (or laps)
    bool fh_frameCountOkSticky = true;
    bool fh_ppsCountOk = true;        // Check 3: PPS count non-decreasing (or laps)
    bool fh_ppsCountOkSticky = true;
};

#endif // FLIGHTAPPSTATUSTYPES_H
