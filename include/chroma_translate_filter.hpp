/*
 * chroma_translate_filter.h
 *
 *  Created on: Jun 3, 2014
 *      Author: nlevy
 */

#ifndef CHROMA_TRANSLATE_FILTER_H_
#define CHROMA_TRANSLATE_FILTER_H_
#include "edtinc.h"
#include <stdint.h>


static const unsigned int WIDTH = 1280;
static const unsigned int  HEIGHT = 480;
static const unsigned int  PIC_SIZE = WIDTH*HEIGHT*sizeof(uint16_t);

uint16_t * apply_chroma_translate_filter(uint16_t * picture);



#endif /* CHROMA_TRANSLATE_FILTER_H_ */
