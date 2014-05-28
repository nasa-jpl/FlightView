#include "frame.hpp"
#include <iostream>

frame::frame(u_char * data_in, int size, int h, int w)
{
	this->height = h;
	this->width = w;
	raw_data = new u_char[size];
	if(raw_data == NULL)
	{
		//TODO: Throw exception
		std::cout << "couldn't allocate frame data" << std::endl;
	}
	memcpy(raw_data, data_in, size); //This could probably be replaced with std::copy
	image_data_ptr = raw_data + width*BYTES_PER_PIXEL;
	this->cmTime = *(uint64_t *) raw_data;
	this->framecount = *((uint16_t *) (raw_data) + 160);
	//image2d =
}
frame::~frame()
{
	delete [] raw_data;

}


