/*
 * takeobject.cpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#include "chroma_translate_filter.cuh"
#include "take_object.hpp"
#include "fft.hpp"
const static int NUMBER_OF_FRAMES_TO_BUFFER = 100;
take_object::take_object(int channel_num, int number_of_buffers, int fmsize, int frf)
{
	this->channel = channel_num;
	this->numbufs = number_of_buffers;
	this->frame_history_size = fmsize;
	this->filter_refresh_rate =frf;
	this->count = 0;

	this->std_dev_filter_N = 400;

	this->do_raw_save = false;

	this->raw_save_file = NULL;

	newFrameAvailable = false;
	dsfMaskCollected = false;


}
take_object::~take_object()
{
	int dummy;
	pdv_thread_run = 0;
	pdv_thread.join(); //Wait for thread to end
	save_thread.join();
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

	switch(size)
	{

	case 481*640*sizeof(uint16_t): cam_type = CL_6604A; break;
	default: cam_type = CL_6604B; break;
	}
	printf("frame period:%i\n", pdv_get_frame_period(pdv_p));

	frWidth = pdv_get_width(pdv_p);

	//Our version of height should not include the header size
	dataHeight = pdv_get_height(pdv_p);

	if(cam_type == CL_6604A) //This strips the header from the height on the 6604A
	{
		frHeight = dataHeight - 1;
	}
	else
	{
		frHeight = dataHeight;
	}
	printf("cam type: %u. Width: %u Height %u frame height %u \n", cam_type, frWidth, dataHeight, frHeight);

	raw_data_ptr = new uint16_t[frWidth*dataHeight];

	fftReal_mean_data = new float[MEAN_BUFFER_LENGTH/2];

	dark_subtraction_data = new float[frWidth*frHeight];
	read_lock = new boost::unique_lock<boost::mutex>(data_mutex,boost::defer_lock);

	//this->dark_subtraction_data = boost::shared_array < float >(new float[frWidth*frHeight]);
	//this->std_dev_data = boost::shared_array < float >(new float[frWidth*frHeight]);
	/*I realize im allocating a ring buffer here and I am using boost to make a totally different one as well.
	 *It seems dumb to do this, but the pdv library has scary warning about making the ring buffer to big, so I decided not to risk it.
	 *
	 */
	std::cout << "about to start threads" << std::endl;
	pdv_multibuf(pdv_p,this->numbufs);
	//The internet says that I need to pass a pointer to the threadable function, and a reference to the calling instance of take_object (this)
	pdv_thread_run = 1;
	dsf = new dark_subtraction_filter(frWidth,frHeight);
	sdvf = new std_dev_filter(frWidth,frHeight);
	mf = new mean_filter(frWidth,frHeight);
	//numbufs = 16;
	pdv_start_images(pdv_p,numbufs); //Before looping, emit requests to fill the pdv ring buffer
	pdv_thread = boost::thread(&take_object::pdv_loop, this);
	//save_thread = boost::thread(&take_object::savingLoop, this);

}
void take_object::pdv_loop()
{

	//uint16_t * image_data_ptr;
	uint16_t framecount = 1;
	uint16_t last_framecount = 0;
	//sdvf->start_std_dev_filter(frame_buffer);
	count = 0;

	while(pdv_thread_run == 1)
	{
		//printf("pdv thread\n");
		//raw_data_ptr = reinterpret_cast<uint16_t *>(pdv_wait_image(pdv_p));
		boost::unique_lock<boost::mutex> exclusive_lock(data_mutex);
		memcpy(raw_data_ptr,pdv_wait_image(pdv_p),frWidth*dataHeight*sizeof(uint16_t));

		if(cam_type == CL_6604B)
		{
			raw_data_ptr = ctf.apply_chroma_translate_filter(raw_data_ptr);

			image_data_ptr = raw_data_ptr;
		}
		else
		{
			image_data_ptr = raw_data_ptr + frWidth;
		}
		mf->start_mean(image_data_ptr);


		if(dsfMaskCollected)
		{
			dark_subtraction_data = dsf->wait_dark_subtraction(); //Add framebuffer[2] frames dark subtracted version (dsf lags by 2 frames)
		}

		if(count % filter_refresh_rate == 0)
		{
			std_dev_data = sdvf->wait_std_dev_filter();
			std_dev_histogram_data = sdvf->wait_std_dev_histogram();

		}

		//update saving thread
		raw_save_ptr = raw_data_ptr;


		dsf->update( image_data_ptr);
		//sdvf->update_GPU_buffer( image_data_ptr);
		if(count % filter_refresh_rate == 0)
		{
			sdvf->start_std_dev_filter(std_dev_filter_N);
		}
		horizontal_mean_data = mf->wait_horizontal_mean();
		vertical_mean_data = mf->wait_vertical_mean();
		fftReal_mean_data = mf->wait_mean_fft();
		count++;
		newFrameAvailable = true;
		saveFrameAvailable = true;
		doSave();
		exclusive_lock.unlock();
		pdv_start_image(pdv_p); //Start another

		//newFrameAvailable.notify_one(); //Tells everyone waiting on newFrame available that they can now go.
		framecount = *(raw_data_ptr + 160);
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
void take_object::savingLoop()
{

	printf("saving loop started!");
	while(pdv_thread_run)
	{

		while(!saveFrameAvailable)
		{
			usleep(500);
		}
		if(raw_save_file != NULL) //Apparently this returns false if null, true otherwise
		{
			//boost::unique_lock<boost::mutex> lock(saving_mutex); //Lock for writing
			//boost::shared_lock<boost::shared_mutex> lock(data_mutex);
			if(frWidth*dataHeight != fwrite(raw_save_ptr, sizeof(uint16_t), frWidth*dataHeight, raw_save_file))
			{
				printf("Writing raw has an error.\n");
			}
			//old_raw_save_ptr = raw_save_ptr; //Hoping this is an atomic operations
			saveFrameAvailable = false;
			save_framenum--;
			//lock.unlock();
			if(save_framenum == 0)
			{
				stopSavingRaws();
			}
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
uint16_t * take_object::getImagePtr()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return image_data_ptr;

}
uint16_t * take_object::getRawPtr()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return raw_data_ptr;
}

void take_object::waitForReadLock()
{
	//boost::unique_lock< boost::shared_mutex > wait_for_new_write_lock(data_mutex); //Grab the lock so that we won't write a new frame while trying to read
	//printf("wait thread\n");

	while(1) //No matter if we have the lock or not, spin.
	{
		usleep(500);
		//printf("new frame unavailable");
		if(newFrameAvailable)
		{
			if(!read_lock->owns_lock())
			{
				read_lock->try_lock(); //Spin until we have the lock
				if(read_lock->owns_lock())
				{
					newFrameAvailable = false;
					//printf("nfa\n");
					break;
				}
			}
		}


	}
}
void take_object::releaseReadLock()
{
	read_lock->unlock();
}
unsigned int take_object::getFrameHeight()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return frHeight;
}
unsigned int take_object::getFrameWidth()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return frWidth;
}
unsigned int take_object::getDataHeight()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return dataHeight;
}
float *  take_object::getStdDevData()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return std_dev_data;
}
float * take_object::getDarkSubtractedData()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return dark_subtraction_data;
}

float* take_object::getHorizontalMean()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return horizontal_mean_data;
}
float* take_object::getVerticalMean()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return vertical_mean_data;
}
float* take_object::getRealFFTMagnitude()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return fftReal_mean_data;
}
uint32_t * take_object::getHistogramData()
{
	while(!read_lock->owns_lock())
	{
		; //DO NOTHING IF NOT LOCKED!
	}
	return std_dev_histogram_data;
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

