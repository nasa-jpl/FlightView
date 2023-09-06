#ifndef SHM_GPS_H
#define SHM_GPS_H

#include <stdint.h>
#include "gpsGUI/gpsbinaryreader.h"

// SHM statusByte:
#define SHM_STATUS_READY (31)
#define SHM_STATUS_WAITING (28)
#define SHM_STATUS_INITALIZING (26)
#define SHM_STATUS_CLOSED (24)
#define SHM_STATUS_ERROR (13)

#define shmGPSMessageBufferSize (10)

struct shmGPSDataStruct {
    char statusByte;
    bool connected; // true with working GNSS connection
    uint16_t counter;
    int writingMessageNumber;
    int messageBufferSize = shmGPSMessageBufferSize;
    gpsMessage message[shmGPSMessageBufferSize];
};


#endif // SHM_GPS_H
