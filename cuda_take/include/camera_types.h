#ifndef CAMERA_TYPES_H_
#define CAMERA_TYPES_H_

/*! \file
 * \brief Specifies camera types specific to the AVIRIS lab. */

// Note, this file will be depreciated soon.

enum camera_t {SSD_ENVI=0, SSD_XIO=1, CL_6604A=2, CL_6604B=3};

const static unsigned int number_of_taps[] = {4,8,4,4,4,(unsigned int)-1};
const static unsigned int max_val[] = {0x3fff,0xffff,0x3fff, 0xffff, 0xffff, (unsigned int)-1};
const static unsigned int height[] = {481,480,480,480,480, (unsigned int)-1};
const static unsigned int width[] = {640,1280,640,480,480, (unsigned int)-1};

#endif /* CAMERA_TYPES_H_ */
