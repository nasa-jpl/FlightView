#include "chroma_translate_filter.hpp"
#include <iostream>
#include "camera_types.h"

void setup_filter(camera_t camera_type)
{
	hardware = camera_type;
	frHeight = height[hardware];
	frWidth = width[hardware];
	num_taps = number_of_taps[hardware];
	MAX_VAL = max_val[hardware];
    //std::cout << "------------------ Completed setup_filter(camType) with height: " << frHeight << ", width: " << frWidth << std::endl;
    std::cout << "Completed camera_type setup." << std::endl;

}

void setup_filter(unsigned int h, unsigned int w)
{
    frHeight = h;
    frWidth = w;
    std::cout << "Setting camera geometry:\n";
    std::cout << "  Completed setup_filter(h,w) with height: " << frHeight << ", width: " << frWidth << std::endl;
}

void apply_teledyne_translation_filter(uint16_t *source, uint16_t *dest) {
    unsigned int row, col;
    //unsigned int div, mod;
    unsigned int destCol = 0;
    const unsigned int tapWidth = 512;

    for(row = 0; row < frHeight; row++)
    {
#pragma omp parallel for num_threads(4)
        for(col = 0; col< frWidth; col++)
        {
            destCol = ((col%4)*tapWidth) + (col/4);
            dest[destCol + row*frWidth] = source[col + row*frWidth ];
        }
    }
}

void apply_teledyne_translation_filter_and_rotate(uint16_t *input, uint16_t* output,
                                                  int origHeight, int origWidth) {

    // TODO: Complete this function
    // DO NOT USE AT THIS TIME
    abort();
    int p=0;
    int outPos = 0;
    int c = 0;
    unsigned int destCol = 0;
    const unsigned int tapWidth = 512;
    // height and width reference the original (input) matrix dims

#pragma omp parallel for num_threads(8)
    for(p = 0; p < origWidth; p++) {
        outPos = origHeight*p;
        for(c=0; c < origHeight*origWidth; c=c+origWidth) {
            // Remap:
           // destCol = ((col%4)*tapWidth) + (col/4);
           // dest[destCol + row*frWidth] = source[col + row*frWidth ];

            // Rotate:
            output[outPos+(c/origWidth)] = input[c+p]; // rotate only


        }
    }

    // Single thread method, which may be slightly faster for single thread only:
    for(p = 0; p < origWidth; p++) {
        for(c=0; c < origHeight*origWidth; c=c+origWidth) {
            output[outPos] = input[c+p]; outPos++; // rotated

           // destCol = ((col%4)*tapWidth) + (col/4);
           // dest[destCol + row*frWidth] = source[col + row*frWidth ];
        }
    }

}

uint16_t* apply_2sComp_translate_filter(uint16_t *picture_in)
{
    // There are no current instruments which need this function.
    unsigned int row, col;
    //unsigned int div, mod;
    for(row = 0; row < frHeight; row++)
	{
        for(col = 0; col< frWidth; col++)
		{
            //div = col/num_taps;
            //mod = col%num_taps;
            //pic_buffer[div + TAP_WIDTH * mod + row * frWidth] = (MAX_VAL - picture_in[col + row * frWidth]);
            pic_buffer[col + row * frWidth] = (picture_in[col + row * frWidth] ^ (1<<15)); // normal 2s compliment

            //pic_buffer[col + row * frWidth] = (picture_in[col + row * frWidth] ^ (1<<15)) - (1<<14); // for EMIT data
            //pic_buffer[col + row * frWidth] = picture_in[col + row * frWidth];
            //pic_buffer[col + row * frWidth] = col + row * frWidth;
            //picture_in[col + row * frWidth] = col + row * frWidth;

		}
	}
    // memcpy(picture_in,pic_buffer,MAX_SIZE*sizeof(uint16_t));
    memcpy(picture_in,pic_buffer,frHeight*frWidth*sizeof(uint16_t));
	return picture_in;
}

