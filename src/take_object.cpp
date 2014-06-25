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

	this->std_dev_filter_N = 400;

	this->do_raw_save = false;
	this->dsf_save_available = false;
	this->std_dev_save_available = false;
	this->raw_save_file = NULL;
	this->dsf_save_file = NULL;
	this->std_dev_save_file = NULL;

}
take_object::~take_object()
{
	int dummy;
	pdv_thread_run = 0;
	pdv_thread.join(); //Wait for thread to end
	pdv_wait_last_image(pdv_p,&dummy); //Collect the last frame to avoid core dump
	pdv_close(pdv_p);
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
		pdv_start_image(pdv_p); //Start another
		if(isChroma)
		{
			new_image_address = ctf.apply_chroma_translate_filter(new_image_address);
		}
		boost::unique_lock< boost::mutex > lock(framebuffer_mutex); //Grab the lock so that ppl won't be reading as you try to write the frame
		boost::shared_ptr<frame> frame_sp(new frame(new_image_address, size, height,width, isChroma));
		dark_subtraction_data = dsf->wait_dark_subtraction(); //Add framebuffer[2] frames dark subtracted version (dsf lags by 2 frames)
		frame_buffer.push_front(frame_sp);
		if(count % filter_refresh_rate == 0)
		{
			std_dev_data = sdvf->wait_std_dev_filter();
			std_dev_histogram_data = sdvf->wait_std_dev_histogram();
			std_dev_save_ptr = std_dev_data;
			std_dev_save_available = true;
		}
		//update saving thread
		raw_save_ptr = frame_sp->raw_data;
		if(frame_buffer.size() > 1)
		{
			dsf->update(frame_buffer[1]->image_data_ptr);
			sdvf->update_GPU_buffer(frame_buffer[1]->image_data_ptr);
			if(count % 10 == 0)
			{
				sdvf->start_std_dev_filter(std_dev_filter_N);
			}
		}
		lock.unlock();
		count++;
		newFrameAvailable.notify_one(); //Tells everyone waiting on newFrame available that they can now go.
		if(CHECK_FOR_MISSED_FRAMES_6604A && !isChroma && frame_buffer.size() >= 2)
		{
			if( frame_sp->framecount -1 != frame_buffer[1]->framecount)
			{
				std::cerr << "WARN MISSED FRAME" << frame_sp->framecount << " " << lastfc << std::endl;
			}
		}
		lastfc = frame_sp->framecount;
	}
}
void take_object::savingLoop()
{
	printf("saving loop started!");
	uint16_t * old_raw_save_ptr = NULL;
	int true_height = (isChroma ? height : height+1);
	while(pdv_thread_run)
	{

		if((old_raw_save_ptr != raw_save_ptr) && raw_save_file != NULL) //Apparently this returns false if null, true otherwise
		{
			//boost::unique_lock<boost::mutex> lock(saving_mutex); //Lock for writing

			if(width*true_height != fwrite(raw_save_ptr, sizeof(uint16_t), width*true_height, raw_save_file))
			{
				printf("Writing raw has an error.\n");
			}
			old_raw_save_ptr = raw_save_ptr; //Hoping this is an atomic operations
		}
		//TODO: Insert DSF saving code
		if(std_dev_save_available && std_dev_save_file != NULL)
		{
			if(width*height != fwrite(std_dev_save_ptr.get(),sizeof(float),width*height,std_dev_save_file))
			{
				printf("Writing std_dev has an error.\n");
				perror("");
			}
			std_dev_save_available = false;
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
	return dark_subtraction_data;
	//return dsf->wait_dark_subtraction();
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
	printf("Begin frame save! @ %s",raw_file_name);
	if(raw_save_file != NULL)
	{
		stopSavingRaws();
	}
	raw_save_file = fopen(raw_file_name, "wb");
	//do_raw_save = true;
}
void take_object::stopSavingRaws()
{
	printf("stop saving raws!");
	//do_raw_save = false;
	if(raw_save_file != NULL)
	{
		FILE * ptr_copy = raw_save_file;
		raw_save_file = NULL;  //Since raw_save_file = NULL should be an atomic operation, this ensures that it won't save while we're calling fclose (boost locks were quite laggy here)


		fclose(ptr_copy);
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
	printf("Begin std_dev save! @ %s",std_dev_file_name);
	if(std_dev_save_file != NULL)
	{
		stopSavingSTD_DEVs();
	}
	std_dev_save_file = fopen(std_dev_file_name, "wb");
}
void take_object::stopSavingSTD_DEVs()
{
	printf("stop saving std_dev!");
	//do_raw_save = false;
	if(std_dev_save_file != NULL)
	{
		FILE * ptr_copy = std_dev_save_file;
		std_dev_save_file = NULL;  //Since raw_save_file = NULL should be an atomic operation, this ensures that it won't save while we're calling fclose (boost locks were quite laggy here)

		fclose(ptr_copy);
	}
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
void take_object::setStdDev_N(int s)
{
	this->std_dev_filter_N = s;
}
boost::shared_array<uint32_t> take_object::getHistogramData()
{
	return std_dev_histogram_data;
}

std::vector<float> * take_object::getHistogramBins()
{
	return sdvf->getHistogramBins();
}
bool take_object::std_dev_ready()
{
	return sdvf->outputReady();
}
