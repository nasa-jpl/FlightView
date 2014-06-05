/*
 * takeobject.cpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#include "chroma_translate_filter.cuh"
#include "take_object.hpp"
#include <iostream>
take_object::take_object(int channel_num, int number_of_buffers, int fmsize, int frf)
{
	this->channel = channel_num;
	this->numbufs = number_of_buffers;
	this->frame_history_size = fmsize;
	this->filter_refresh_rate =frf;
	this->frame_buffer = boost::circular_buffer<boost::shared_ptr<frame>>(fmsize); //Initialize the ring buffer of frames
	this->count = 0;
}
take_object::~take_object()
{
	int dummy;
	//pdv_thread.interrupt();
	pdv_thread_run = 0;
	pdv_thread.join(); //Wait for thread to end
	pdv_wait_last_image(pdv_p,&dummy); //Collect the last frame to avoid core dump
	pdv_close(pdv_p);
	//Apparently if I use pdv_open, it implicitly calls malloc and therefore I should use free, not delete.
	//free(this->pdv_p);
}

void take_object::initFilters(int history_size)
{
	sdvf=new std_dev_filter(width,height,history_size);
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
	size = pdv_get_dmasize(pdv_p); //Not using this at the moment
	isChroma = size > 481*640*BYTES_PER_PIXEL ? true : false;

	width = pdv_get_width(pdv_p);

	//Our version of height should not include the header size
	height = pdv_get_height(pdv_p);
	if(!isChroma)
	{
		height = height - 1;
	}

	/*I realize im allocating a ring buffer here and I am using boost to make a totally different one as well.
	 *It seems dumb to do this, but the pdv library has scary warning about making the ring buffer to big, so I decided not to risk it.
	 *
	 */
	std::cout << "about to start threads" << std::endl;
	pdv_multibuf(pdv_p,this->numbufs);
	//The internet says that I need to pass a pointer to the threadable function, and a reference to the calling instance of take_object (this)
	pdv_thread_run = 1;
	pdv_thread = boost::thread(&take_object::pdv_init, this);
}
void take_object::pdv_init()
{

	uint16_t * new_image_address;
	pdv_start_images(pdv_p,numbufs); //Before looping, emit requests to fill the pdv ring buffer
	//sdvf->start_std_dev_filter(frame_buffer);
	while(pdv_thread_run == 1)
	{


		new_image_address = reinterpret_cast<uint16_t *>(pdv_wait_image(pdv_p)); //We're never going to deal with u_char *, ever again.
		//std::cout << "a good outcome" << std::endl;

		pdv_start_image(pdv_p); //Start another
		//if chroma, translate new image
		//std::cout << "a gooder outcome" << std::endl;

		boost::unique_lock< boost::mutex > lock(framebuffer_mutex); //Grab the lock so that ppl won't be reading as you try to write the frame
		//std::cout << "a goodest outcome" << std::endl;

		if(isChroma)
		{
		new_image_address = ctf.apply_chroma_translate_filter(new_image_address);
		}
		append_to_frame_buffer(new_image_address);

		if(count % filter_refresh_rate == 0)
		{
			/*
			std_dev_data = sdvf->wait_std_dev_filter();


			sdvf->start_std_dev_filter();
			 */
			//std_dev_data = boost::shared_array < float > (sdvf->wait_std_dev_filter()); //Use copy constructor
		}

		count++;
		lock.unlock();


		newFrameAvailable.notify_one(); //Tells everyone waiting on newFrame available that they can now go.

	}
}
void take_object::append_to_frame_buffer(uint16_t * data_in)
{
	boost::shared_ptr<frame> frame_sp(new frame(data_in, size, height,width, isChroma));
	frame_buffer.push_front(frame_sp);

}
boost::shared_ptr<frame> take_object::getFrontFrame()
{
	boost::unique_lock< boost::mutex > lock(framebuffer_mutex); //Grab the lock so that ppl won't be reading as you try to write the frame
	newFrameAvailable.wait(lock);
	return frame_buffer[0];

}
boost::shared_array<float>  take_object::getStdDevData()
{
	return std_dev_data;
}

