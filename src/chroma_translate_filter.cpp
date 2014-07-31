#include "chroma_translate_filter.hpp"
#include <iostream>
#include "constants.h"

//MAKING ASSUMPTIONS BECAUSE CHROMA!

#define UINT16_MAX = 0xFFFF
static uint16_t pic_buffer[MAX_SIZE];

uint16_t * apply_chroma_translate_filter(uint16_t * picture_in)
{
	unsigned int width_eigth = WIDTH/8;
	for(unsigned int r = 0; r < HEIGHT; r++)
	{
		for(unsigned int col = 0; col<WIDTH; col++)
		{
			unsigned int c = col/8;
			unsigned int i = col % 8;
			pic_buffer[c+width_eigth*i + r*WIDTH] = (0xFFFF - picture_in[c*8+i + r*WIDTH]);
		}
	}
	memcpy(picture_in,pic_buffer,MAX_SIZE*sizeof(uint16_t));
	return picture_in;

}

