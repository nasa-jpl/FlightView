#ifndef CONSTANTS_H_
#define CONSTANTS_H_

/*! \file
 * \brief Specifies the constants for hardware and memory allocation.
 *
 * These settings are tuned for the hardware used by the AVIRIS lab. New frame geometry must be specified here. Additionally,
 * depending on system requirements, the memory allocation sizes may need to be adjusted.
 */

const static unsigned int MAX_WIDTH = 1280;
const static unsigned int MAX_HEIGHT = 480;
const static unsigned int TAP_WIDTH = 160;
const static unsigned int MAX_SIZE = MAX_WIDTH*MAX_HEIGHT;
const static unsigned int FFT_MEAN_BUFFER_LENGTH = 500;
const static unsigned int FFT_INPUT_LENGTH = 256; // Must be 256, will fail silently otherwise
static const unsigned int MAX_FFT_SIZE = 4096;
static const unsigned int MAX_N = 500;
static const unsigned int CPU_FRAME_BUFFER_SIZE = 1500; // The frame ring buffer size in number of frame_c structs
static const unsigned int GPU_FRAME_BUFFER_SIZE = MAX_N*3/2; //1500
static const unsigned int BLOCK_SIZE = 20; // This is not used by default.

static const unsigned int NUMBER_OF_BINS = 1024; // For histograms

#endif /* CONSTANTS_H_ */
