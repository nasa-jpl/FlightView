/*
 * takeobject.cpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#include "chroma_translate_filter.cuh"
#include "take_object.hpp"
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

	if(!isChroma) //This strips the header from the height on the 6604A
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
	dsf = new dark_subtraction_filter(width,height);
	sdvf = new std_dev_filter(width,height);
	pdv_start_images(pdv_p,numbufs); //Before looping, emit requests to fill the pdv ring buffer
	pdv_thread = boost::thread(&take_object::pdv_init, this);
	save_thread = boost::thread(&take_object::savingLoop, this);

}
void take_object::pdv_init()
{

	uint16_t * new_image_address;

	//sdvf->start_std_dev_filter(frame_buffer);
	count = 0;
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

		//dsf->update(new_image_address); //let's let this chug we'll we set up the rest of the frame

		boost::shared_ptr<frame> frame_sp(new frame(new_image_address, size, height,width, isChroma));
		frame_sp->dsf_data = dsf->wait_dark_subtraction(); //Add framebuffer[2] frames dark subtracted version (dsf lags by 2 frames)
		frame_buffer.push_front(frame_sp);


		if(count % filter_refresh_rate == 0)
		{
			std_dev_data = sdvf->wait_std_dev_filter();
		}
		if(frame_buffer.size() > 1)
		{
			dsf->update(frame_buffer[1]->image_data_ptr);
			sdvf->update_GPU_buffer(frame_buffer[1]->image_data_ptr);
			if(count % 10 == 0)
			{
				sdvf->start_std_dev_filter(400);
			}
		}
		lock.unlock();
		count++;
		newFrameAvailable.notify_one(); //Tells everyone waiting on newFrame available that they can now go.




		//std_dev_data = boost::shared_array < float > (sdvf->wait_std_dev_filter()); //Use copy constructor
	}
}
void take_object::savingLoop()
{
	while(pdv_thread_run)
	{
		if(raw_save_ptr && raw_save_file != NULL) //Apparently this returns false if null, true otherwise
		{
			//	raw_save_file->write(raw_save_ptr.get(), width*height*sizeof(uint16_t));
			if(width*height != fwrite(raw_save_ptr.get(), sizeof(uint16_t), width*height, raw_save_file))
			{
				printf("Writing raw has an error.");
			}
		}


	}
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
boost::shared_array<float> take_object::getDarkSubtractedData()
{
	return dsf->wait_dark_subtraction();
}
void take_object::startCapturingDSFMask()
{
	dsf->start_mask_collection();

}
void take_object::finishCapturingDSFMask()
{
	dsf->finish_mask_collection();
}
void take_object::startSavingRaws(const char * raw_file_name)
{
	if(raw_save_file != NULL)
	{
		stopSavingRaws();
	}
	raw_save_file = fopen(raw_file_name, "wb");
}
void take_object::stopSavingRaws()
{
	if(raw_save_file != NULL)
	{
		fclose(raw_save_file);
	}
}
void take_object::startSavingDSFs(const char * dsf_file_name)
{

}
void take_object::stopSavingDSFs()
{

}
void take_object::startSavingSTD_DEVs(const char * std_dev_file_name)
{

}
void take_object::stopSavingSTD_DEVs()
{

}
void take_object::loadDSFMask(const char * file_name)
{
	boost::shared_array < float > mask_in(new float[width*height]);
	FILE * pFile;
	unsigned long size = 0;
	pFile  = fopen(file_name, "rb");
	if(pFile==NULL) std::cerr << "error opening raw file" << std::endl;
	else
	{
		fseek (pFile, 0, SEEK_END);   // non-portable
		size=ftell (pFile);
		if(size != (width*height*sizeof(float)))
		{
			std::cerr << "error mask file does not match image size" << std::endl;
			fclose (pFile);
			return;
		}
		rewind(pFile);   // go back to beginning
		fread(mask_in.get(),sizeof(float),width*height,pFile);
		fclose (pFile);
		std::cout << file_name << " read in "<< size << " bytes successfully " <<  std::endl;



	}
	dsf->load_mask(mask_in);
}


