#include "chroma_translate_filter.hpp"
#include <iostream>
#include "constants.h"

//MAKING ASSUMPTIONS BECAUSE CHROMA!


#define UINT16_MAX_NEW 0xffff


static uint16_t pic_buffer[MAX_SIZE];

uint16_t * apply_chroma_translate_filter(uint16_t * picture_in)
{
    for(unsigned int row = 0; row < HEIGHT; row++)
	{
        for(unsigned int col = 0; col<WIDTH; col++)
		{
            unsigned int div = col/8;
            unsigned int mod = col%8;
            pic_buffer[div + TAP_WIDTH*mod + row*WIDTH] = (UINT16_MAX_NEW - picture_in[col + row*WIDTH]);
		}
	}
	memcpy(picture_in,pic_buffer,MAX_SIZE*sizeof(uint16_t));
	return picture_in;

}

