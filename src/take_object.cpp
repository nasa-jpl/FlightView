/*
 * takeobject.cpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#include "take_object.hpp"
#include <iostream>
take_object::take_object(int channel_num, int number_of_buffers, int fmsize)
{
	this->channel = channel_num;
	this->numbufs = number_of_buffers;
	this->frame_history_size = fmsize;
	this->frame_buffer = boost::circular_buffer<frame>(fmsize); //Initialize the ring buffer of frames
}
take_object::~take_object()
{
	pdv_close(pdv_p);
	//Apparently if I use pdv_open, it implicitly calls malloc and I should use free, not delete
	//delete this->pdv_p;
	free(this->pdv_p);
}


void take_object::start()
{
	this->pdv_p = NULL;
	this->pdv_p = pdv_open_channel(EDT_INTERFACE,0,this->channel);
	if(pdv_p == NULL)
	{
		std::cout << "Oh gawd, couldn't open channel" << std::endl;
		//TODO: throw exception
		return;
	}
	//TODO: Sizing stuff
	int size = pdv_get_dmasize(pdv_p); //Not using this at the moment
	/*I realize im allocating a ring buffer here and I am using boost to make a totally different one as well.
	 *It seems dumb to do this, but the pdv library has scary warning about making the ring buffer to big, so I decided not to risk it.
	 *
	 */
	pdv_multibuf(pdv_p,this->numbufs);
	//The internet says that I need to pass a pointer to the threadable function, and a reference to the calling instance of take_object (this)
	pdv_thread = boost::thread(&take_object::pdv_init, this);
}
void take_object::pdv_init()
{

	//Going to try naively using pdv_image, if this doesn't work TODO: use pdv_start_image and pdv_wait_image
	while(1)
	{
		u_char * new_image_address = pdv_image(pdv_p);

	}
}
void take_object::append_to_frame_buffer(u_char * data_in, int framesize)
{

}

