/*
 * constants.h
 *
 *  Created on: Jul 15, 2014
 *      Author: nlevy
 */
#ifndef CONSTANTS_H_
#define CONSTANTS_H_
const static unsigned int MAX_WIDTH = 1280;
const static unsigned int MAX_HEIGHT = 480;
const static unsigned int MAX_SIZE = MAX_WIDTH*MAX_HEIGHT;
const static unsigned int FFT_MEAN_BUFFER_LENGTH = 1000;
const static unsigned int FFT_INPUT_LENGTH = 256; //must be power of 2; will fail silently otherwise
static const unsigned int MAX_FFT_SIZE = 4096;
//static const unsigned int MAX_N = 200;
static const unsigned int MAX_N = 1000;
static const unsigned int CPU_FRAME_BUFFER_SIZE = 1500;
static const unsigned int GPU_FRAME_BUFFER_SIZE = MAX_N*3/2; //1500
static const unsigned int BLOCK_SIDE = 20;
static const unsigned int NUMBER_OF_BINS = 1024;



#endif /* CONSTANTS_H_ */
