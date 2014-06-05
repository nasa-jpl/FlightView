#include "frame.hpp"
#include <iostream>

frame::frame(uint16_t * data_in, int size, int h, int w, bool isChroma)
{
	this->height = h;
	this->width = w;
	raw_data = new uint16_t[size];
	if(raw_data == NULL)
	{
		//TODO: Throw exception
		std::cout << "couldn't allocate frame data" << std::endl;
	}

	memcpy(raw_data, data_in, size); //This could probably be replaced with std::copy
	if(isChroma)
	{
		image_data_ptr = raw_data; //The Chroma has no header
	}
	else
	{
		image_data_ptr = raw_data + width;
	}
	this->cmTime = *(uint64_t *) raw_data;
	this->framecount = *((uint16_t *) (raw_data) + 160);
	//image2d =
}
frame::~frame()
{
	delete [] raw_data;

}


