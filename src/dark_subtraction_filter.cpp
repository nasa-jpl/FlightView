#include "dark_subtraction_filter.hpp"

//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
//Kernel code, this runs on the GPU (device)

void dark_subtraction_filter::start_mask_collection()
{
	mask_collected = false;
	averaged_samples = 0; //Synchronous
//#pragma omp parallel for
	for(unsigned int i = 0; i < width*height; i++)
	{
		mask[i]=0;
	}
}


void dark_subtraction_filter::finish_mask_collection()
{
//#pragma omp parallel for
	for(unsigned int i = 0; i < width*height; i++)
	{
		mask[i]/=averaged_samples;
	}
	mask_collected = true;
	std::cout << "mask collected: " << std::endl;
}
void dark_subtraction_filter::update(uint16_t * pic_in, float * pic_out)
{
	if(mask_collected)
	{
		update_dark_subtraction(pic_in, pic_out);
	}
	else
	{
		update_mask_collection(pic_in);
	}
}

void dark_subtraction_filter::load_mask(float * mask_arr)
{
	memcpy(mask,mask_arr,width*height*sizeof(float));
	mask_collected = true;
	std::cout << "mask loaded" << std::endl;
}
float * dark_subtraction_filter::get_mask()
{
	return mask;
}
void dark_subtraction_filter::update_dark_subtraction(uint16_t * pic_in, float * pic_out)
{

	//Synchronous

	//Asynchronous
//#pragma omp parallel for
	for(unsigned int i = 0; i < width*height; i++)
	{
		pic_out[i] = pic_in[i] - mask[i];
	}
}

uint32_t dark_subtraction_filter::update_mask_collection(uint16_t * pic_in)
{
	// Synchronous
//#pragma omp parallel for
	for(unsigned int i=0; i<width*height;i++)
	{
		mask[i] = pic_in[i] + mask[i];

	}
	averaged_samples++;
	return averaged_samples;
}

dark_subtraction_filter::dark_subtraction_filter(int nWidth, int nHeight)
{


	mask_collected = false;
	width=nWidth;
	height=nHeight;

}
dark_subtraction_filter::~dark_subtraction_filter()
{
	mask_collected = false; //Do this to prevent reading after object has been killed
}


