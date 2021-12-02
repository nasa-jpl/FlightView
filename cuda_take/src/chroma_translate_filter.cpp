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
}
uint16_t* apply_chroma_translate_filter(uint16_t *picture_in)
{
	unsigned int row, col, div, mod;
    for(row = 0; row < frHeight; row++)
	{
        for(col = 0; col< frWidth; col++)
		{
            //div = col/num_taps;
            //mod = col%num_taps;
            //pic_buffer[div + TAP_WIDTH * mod + row * frWidth] = (MAX_VAL - picture_in[col + row * frWidth]);
            pic_buffer[col + row * frWidth] = picture_in[col + row * frWidth] ^ (1<<15);
		}
	}
	memcpy(picture_in,pic_buffer,MAX_SIZE*sizeof(uint16_t));
	return picture_in;
}

