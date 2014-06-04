#include "frame.hpp"
#include <iostream>

frame::frame(u_char * data_in, int size, int h, int w, bool isChroma)
{
	this->height = h;
	this->width = w;
	raw_data = new u_char[size];
	if(raw_data == NULL)
	{
		//TODO: Throw exception
		std::cout << "couldn't allocate frame data" << std::endl;
	}


	if(isChroma)
	{

		uint16_t * in_r = reinterpret_cast<uint16_t *>(data_in);


		uint16_t * out_r = reinterpret_cast<uint16_t *>(raw_data);

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
	//	memcpy(in, out_r, width*height*BYTES_PER_PIXEL);
		//delete [] out_r;
		//in[0] = 0;
		//return in;
	}

	else
	{
	memcpy(raw_data, data_in, size); //This could probably be replaced with std::copy
	}
	image_data_ptr = raw_data + width*BYTES_PER_PIXEL;
	this->cmTime = *(uint64_t *) raw_data;
	this->framecount = *((uint16_t *) (raw_data) + 160);
	//image2d =
}
frame::~frame()
{
	delete [] raw_data;

}


