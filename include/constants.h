/*
 * constants.h
 *
 *  Created on: Jul 15, 2014
 *      Author: nlevy
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_
const static int MAX_WIDTH = 1280;
const static int MAX_HEIGHT = 480;
const static int MAX_SIZE = MAX_WIDTH*MAX_HEIGHT;
const static unsigned int MEAN_BUFFER_LENGTH = 256; //must be power of 2; will fail silently otherwise
static const unsigned int MAX_FFT_SIZE = 4096;
static const int MAX_N = 1000;
static const int BLOCK_SIDE = 20;
static const int NUMBER_OF_BINS = 1024;



#endif /* CONSTANTS_H_ */
