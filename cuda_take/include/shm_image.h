#ifndef SHM_IMAGE_H
#define SHM_IMAGE_H

#include <stdint.h>
#include <stdbool.h>

// Note: This is a static allocation for a specific geometry.
// A static allocation is easier to deal with for the SHM.


#define shmHeight (480)
#define shmWidth (1280)

// Time between frames, ie, 1/FPS, in microseconds
#define frameDwellTime_us (10000)

#define shmFrameBufferSize (10)
#define shmFilenameBufferSize (256)

#define SHM_STATUS_READY (31)
#define SHM_STATUS_WAITING (28)

#define SHM_STATUS_INITALIZING (26)

#define SHM_STATUS_CLOSED (24)
#define SHM_STATUS_ERROR (13)


struct shmSharedDataStruct {
    char statusByte; // 1+3 bytes, shall be set to "31" when in-use.
    float fps; // monitor this for an indication of if the data are flowing as expected
    bool recordingDataToFile; // will indicate if a file happens to be being written to at the moment
    uint16_t counter; // +1 each time a frame is written in. Used to track missing frames and overall progress

    int writingFrameNum; // indicates the frame number being written to. Do not read that frame, always read one behind.
    int bufferSizeFrames;
    int frameWidth;
    int frameHeight;

    bool takingDark;
    uint64_t frameTime[shmFrameBufferSize]; // system time since epoch, used to verify FPS and overall progress
    uint16_t frameBuffer[shmFrameBufferSize][shmWidth*shmHeight];
    char lastFilename[shmFilenameBufferSize]; // I might send this just for reference, it could be handy for anyone looking at the data.
};

union shmDataCombiner {
        char* c;
        uint16_t *u;
};



#endif // SHM_IMAGE_H
