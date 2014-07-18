/*
 * takeobject.cpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#include "chroma_translate_filter.cuh"
#include "take_object.hpp"
#include "fft.hpp"
take_object::take_object(int channel_num, int number_of_buffers, int fmsize, int frf)
{
	this->channel = channel_num;
	this->numbufs = number_of_buffers;
	this->filter_refresh_rate =frf;
	this->count = 0;

	this->std_dev_filter_N = 400;

	this->do_raw_save = false;

	this->raw_save_file = NULL;

	dsfMaskCollected = false;


}
take_object::~take_object()
{
	int dummy;
	pdv_thread_run = 0;
	pdv_thread.join(); //Wait for thread to end
	pdv_wait_last_image(pdv_p,&dummy); //Collect the last frame to avoid core dump
	pdv_close(pdv_p);

	delete dsf;
	delete sdvf;
	//delete mf;
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
	size = pdv_get_dmasize(pdv_p); //Not using this at the moment
	switch(size)
	{
	case 481*640*sizeof(uint16_t): cam_type = CL_6604A; break;
	default: cam_type = CL_6604B; break;
	}
	printf("frame period:%i\n", pdv_get_frame_period(pdv_p));
	frWidth = pdv_get_width(pdv_p);
	//Our version of height should not include the header size
	dataHeight = pdv_get_height(pdv_p);
	frHeight = cam_type == CL_6604A ? dataHeight -1 : dataHeight;
	printf("cam type: %u. Width: %u Height %u frame height %u \n", cam_type, frWidth, dataHeight, frHeight);

	std::cout << "about to start threads" << std::endl;
	pdv_multibuf(pdv_p,this->numbufs);

	pdv_thread_run = 1;
	dsf = new dark_subtraction_filter(frWidth,frHeight);
	sdvf = new std_dev_filter(frWidth,frHeight);
	//numbufs = 16;
	pdv_start_images(pdv_p,numbufs); //Before looping, emit requests to fill the pdv ring buffer
	pdv_thread = boost::thread(&take_object::pdv_loop, this);
	//save_thread = boost::thread(&take_object::savingLoop, this);

}
void take_object::pdv_loop() //Producer Thread
{

	uint16_t framecount = 1;
	uint16_t last_framecount = 0;
	count = 0;

	u_char * wait_ptr;

	while(pdv_thread_run == 1)
	{
		//curFrame = std::shared_ptr<frame_c>(new frame_c());
		curFrame = new frame_c();
		wait_ptr = pdv_wait_image(pdv_p);
		memcpy(curFrame->raw_data_ptr,wait_ptr,frWidth*dataHeight*sizeof(uint16_t));

		if(cam_type == CL_6604B)
		{
			ctf.apply_chroma_translate_filter(curFrame->raw_data_ptr, curFrame->raw_data_ptr);
			curFrame->image_data_ptr = curFrame->raw_data_ptr;
		}

		else
		{
			curFrame->image_data_ptr = curFrame->raw_data_ptr + frWidth;
		}

		mean_filter * mf = new mean_filter(curFrame, count, frWidth, frHeight); //This will deallocate itself when it is done.
		mf->start_mean();
		dsf->update(curFrame->raw_data_ptr,curFrame->dark_subtracted_data);
		sdvf->update_GPU_buffer(curFrame->image_data_ptr);

		if(count % filter_refresh_rate == 0)
		{
			//sdvf->start_std_dev_filter(std_dev_filter_N,curFrame->std_dev_data,curFrame->std_dev_histogram);
		}

		if(count % filter_refresh_rate == filter_refresh_rate-1)
		{
			//sdvf->wait_std_dev();
		}
		//mf->wait_mean();
		dsf->wait_dark_subtraction();
		frame_list.push_front(curFrame);
		count++;
		saveFrameAvailable = true;
		//doSave();
		pdv_start_image(pdv_p); //Start another

		//newFrameAvailable.notify_one(); //Tells everyone waiting on newFrame available that they can now go.
		framecount = *(curFrame->raw_data_ptr + 160);
		if(CHECK_FOR_MISSED_FRAMES_6604A && cam_type == CL_6604A)
		{
			if( framecount -1 != last_framecount)
			{
				std::cerr << "WARN MISSED FRAME" << framecount << " " << lastfc << std::endl;
			}
		}
		last_framecount = framecount;
		//while(newFrameAvailable)
		{
			; //do nothing
		}
	}
}

void take_object::doSave()
{
	if(raw_save_file != NULL) //Apparently this returns false if null, true otherwise
	{
		if(frWidth*dataHeight != fwrite(raw_save_ptr, sizeof(uint16_t), frWidth*dataHeight, raw_save_file))
		{
			printf("Writing raw has an error.\n");
		}
		//old_raw_save_ptr = raw_save_ptr; //Hoping this is an atomic operations
		saveFrameAvailable = false;
		save_framenum--;
		if(save_framenum == 0)
		{
			stopSavingRaws();
		}
	}
}

unsigned int take_object::getFrameHeight()
{
	return frHeight;
}
unsigned int take_object::getFrameWidth()
{
	return frWidth;
}
unsigned int take_object::getDataHeight()
{
	return dataHeight;
}

void take_object::startCapturingDSFMask()
{
	dsfMaskCollected = false;
	dsf->start_mask_collection();

}
void take_object::finishCapturingDSFMask()
{
	dsf->finish_mask_collection();
	dsfMaskCollected = true;
}
void take_object::startSavingRaws(std::string raw_file_name, unsigned int frames_to_save)
{
	save_framenum = frames_to_save;
	save_count = 0;
	printf("Begin frame save! @ %s", raw_file_name.c_str());
	if(raw_save_file != NULL)
	{
		stopSavingRaws();
	}
	raw_save_file = fopen(raw_file_name.c_str(), "wb");

	setvbuf(raw_save_file,NULL,_IOFBF,NUMBER_OF_FRAMES_TO_BUFFER*size);

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

void take_object::loadDSFMask(std::string file_name)
{
	//boost::shared_array < float > mask_in(new float[frWidth*frHeight]);
	float * mask_in = new float[frWidth*frHeight];
	FILE * pFile;
	unsigned long size = 0;
	pFile  = fopen(file_name.c_str(), "rb");
	if(pFile==NULL) std::cerr << "error opening raw file" << std::endl;
	else
	{
		fseek (pFile, 0, SEEK_END);   // non-portable
		size=ftell (pFile);
		if(size != (frWidth*frHeight*sizeof(float)))
		{
			std::cerr << "error mask file does not match image size" << std::endl;
			fclose (pFile);
			return;
		}
		rewind(pFile);   // go back to beginning
		fread(mask_in,sizeof(float),frWidth*frHeight,pFile);
		fclose (pFile);
		std::cout << file_name << " read in "<< size << " bytes successfully " <<  std::endl;



	}
	dsf->load_mask(mask_in);
}
void take_object::setStdDev_N(int s)
{
	this->std_dev_filter_N = s;
}


std::vector<float> * take_object::getHistogramBins()
{
	return sdvf->getHistogramBins();
}
bool take_object::std_dev_ready()
{
	return sdvf->outputReady();
}

