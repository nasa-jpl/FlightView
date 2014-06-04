#include <stdint.h>
#include "chroma_translate_filter.cuh"

#ifndef BYTES_PER_PIXEL
#define BYTES_PER_PIXEL 2
#endif
u_char * chroma_translate(u_char * in,int height, int width)
{

	uint16_t * in_r = reinterpret_cast<uint16_t *>(in);


	static uint16_t * out_r = new uint16_t[width*height]; //only create this once...
	//int length = height * width;

	int width_eigth = width/8;
	for(int r = 0; r < height; r++)
	{
		for(int c = 0; c < width_eigth; c++)
		{
			for(int i = 0; i < 8; i++)
			{
				out_r[c+width_eigth*i + r*width] = in_r[c*8+i + r*width];
			}


		}
	}
	//Copy back into original pointer
	memcpy(in, out_r, width*height*BYTES_PER_PIXEL);
	//delete [] out_r;
	//in[0] = 0;
	return in;

}
