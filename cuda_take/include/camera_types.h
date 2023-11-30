#ifndef CAMERA_TYPES_H_
#define CAMERA_TYPES_H_

/*! \file
 * \brief Specifies camera types specific to the AVIRIS lab. */

//enum camera_t {CL_6604A, CL_6604B, FPGA};

enum camera_t {SSD_ENVI, SSD_XIO, CL_6604A, CL_6604B};

// EHL TODO: Expand these to include four options each with a null termination.
const static unsigned int number_of_taps[] = {4,8,4};
const static unsigned int max_val[] = {0x3fff,0xffff,0x3fff};
const static unsigned int height[] = {481,480,480};
const static unsigned int width[] = {640,1280,640};

#endif /* CAMERA_TYPES_H_ */
