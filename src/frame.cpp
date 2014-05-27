#include "frame.hpp"
#include <iostream>

frame::frame(u_char * data_in, int size, int height, int width)
{
	raw_data = new u_char[size];
	if(raw_data == NULL)
	{
		//TODO: Throw exception
		std::cout << "couldn't allocate frame data" << std::endl;
	}
	memcpy(raw_data, data_in, size);
	image_data_ptr = raw_data + width*BYTES_PER_PIXEL;
}
frame::~frame()
{

}


