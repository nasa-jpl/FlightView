#ifndef SHM_IMAGE_H
#define SHM_IMAGE_H

#include <stdint.h>
#include <stdbool.h>

// This file describes the shared memory segment
// used for images and image meta-data.
// The segment is named "/liveview_image"
// Data are processed for 2s compliment and "invertedness" and then
// memcpy'd to the shared memory segment.
// Data are written here whenever liveview is open.

// Note on allocation: This is a static allocation for a specific frame geometry.
// A static allocation is easier to deal with for the SHM.
// Future versions might make this more dynamic.
// It is, of course, possible to write smaller frames into a buffer designed
// for larger frames.
#define shmHeight (480)
#define shmWidth (1280)
#define shmFrameBufferSize (10)
#define shmFilenameBufferSize (256)

// Shared Memory Segment statusByte:
#define SHM_STATUS_READY (31)
#define SHM_STATUS_WAITING (28)
#define SHM_STATUS_INITALIZING (26)
#define SHM_STATUS_CLOSED (24)
#define SHM_STATUS_ERROR (13)

// Shared Memory Segment Data Structure:
struct shmSharedDataStruct {
    char statusByte; // Packed as four bytes, see above
    float fps; // monitor this for an indication of if the data are flowing as expected
    bool recordingDataToFile; // True while data is being saved to disk
    uint16_t counter; // +1 each time a frame is written in. Used to track missing frames and overall progress

    int writingFrameNum; // indicates the frame number being written to. Do not read that frame, always read one behind.
    int bufferSizeFrames; // number of frames in the buffer. Set equal to shmFrameBufferSize during runtime.
    int frameWidth; // pixels wide
    int frameHeight; // pixels high, includes "line header"

    bool takingDark; // indicates if, at the moment, we are recording darks.
    uint64_t frameTime[shmFrameBufferSize]; // system (computer) time since epoch, in milliseconds. Use to monitor "freshness" of data.
    uint16_t frameBuffer[shmFrameBufferSize][shmWidth*shmHeight]; // Buffer of frames. Read into the buffer by offsetting how many bytes-of-frame are needed.
    char lastFilename[shmFilenameBufferSize]; // Last used filename for saving data out. Is not cleared or reset after saving.
};

// Union for manipulating the buffers as either pixels or bytes:
union shmDataCombiner {
        char* c;
        uint16_t *u;
};



#endif // SHM_IMAGE_H
