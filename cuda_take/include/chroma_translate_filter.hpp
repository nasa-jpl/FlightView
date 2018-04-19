/*
 * chroma_translate_filter.h
 *
 *  Created on: Jun 3, 2014
 *      Author: nlevy
 */

#ifndef CHROMA_TRANSLATE_FILTER_H_
#define CHROMA_TRANSLATE_FILTER_H_
#include "edtinc.h"
#include "camera_types.h"
#include "constants.h"
#include <stdint.h>

/*! \file
 * \brief A filter which converts parallel data from the camera link to a corrected image.
 * \paragraph
 *
 * This function takes the pixel data from the chroma detector, which comes through as eight parallel pixels from each tap,
 * and distributes them evenly among the taps to re-create the actual image. Additionally, the pixels are inverted in magnitude
 * based on the raw image. 0xffff represents the maximum pixel value for the 16-bit data.
 */

static uint16_t pic_buffer[MAX_SIZE];
static int hardware;
static unsigned int frHeight;
static unsigned int frWidth;
static unsigned int num_taps;
static unsigned int MAX_VAL;

void setup_filter(camera_t camera_type);
uint16_t * apply_chroma_translate_filter(uint16_t * picture);

#endif /* CHROMA_TRANSLATE_FILTER_H_ */
