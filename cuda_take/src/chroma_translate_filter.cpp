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
    std::cout << "------------------ Completed setup_filter(camType) with height: " << frHeight << ", width: " << frWidth << std::endl;
}

void setup_filter(unsigned int h, unsigned int w)
{
    frHeight = h;
    frWidth = w;
    std::cout << "Overriding camera geometry:\n";
    std::cout << "------------------ Completed setup_filter(h,w) with height: " << frHeight << ", width: " << frWidth << std::endl;
    std::cout << "------------------ h: " << h << ", w: " << w << std::endl;
}

uint16_t* apply_chroma_translate_filter(uint16_t *picture_in)
{
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

