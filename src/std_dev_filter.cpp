#include "std_dev_filter.hpp"
#include "cuda_utils.cuh"
#include "constants.h"
#include <cuda_profiler_api.h>
#include <math.h>
#include <iostream>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

std_dev_filter::std_dev_filter(int nWidth, int nHeight)
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	width = nWidth; //Making the assumption that all frames in a frame buffer are the same size
	height = nHeight;
	gpu_buffer_head = 0;
	currentN = 0;


	printf("\ncreated logarithmic bins\n");

	HANDLE_ERROR(cudaStreamCreate(&std_dev_stream));
	HANDLE_ERROR(cudaMalloc( (void **)&pictures_device, width*height*sizeof(uint16_t)*GPU_FRAME_BUFFER_SIZE)); //Allocate a huge amount of memory on the GPU (N times the size of each frame stored as a u_char)
	HANDLE_ERROR(cudaMalloc( (void **)&picture_out_device, width*height*sizeof(float))); //Allocate memory on GPU for reduce target



	HANDLE_ERROR(cudaMalloc( (void **)&histogram_out_device, NUMBER_OF_BINS*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMalloc( (void **)&histogram_bins_device, NUMBER_OF_BINS*sizeof(float)));
	memcpy(histogram_bins,getHistogramBinValues().data(),NUMBER_OF_BINS*sizeof(float));
	//Calculate logarithmic bins
	//float increment = (UINT16_MAX - 0)/NUMBER_OF_BINS;

	HANDLE_ERROR(cudaMemcpyAsync(histogram_bins_device ,histogram_bins,NUMBER_OF_BINS*sizeof(float),cudaMemcpyHostToDevice,std_dev_stream)); 	//Incrementally copies data to device (as each frame comes in it gets copied
}

std_dev_filter::~std_dev_filter()
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	HANDLE_ERROR(cudaFree(pictures_device)); //do not free current picture because it poitns to a location inside pictures_device
	HANDLE_ERROR(cudaFree(picture_out_device));

	HANDLE_ERROR(cudaFree(histogram_out_device));
	HANDLE_ERROR(cudaFree(histogram_bins_device));

	HANDLE_ERROR(cudaStreamDestroy(std_dev_stream));
}

void std_dev_filter::update_GPU_buffer(frame_c * frame, unsigned int N)
{
	static int count = 0;
	//Synchronous Part
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));

	cudaError std_dev_stream_status = cudaStreamQuery(std_dev_stream);

    char *device_ptr = ((char *)(pictures_device)) + (gpu_buffer_head*width*height*sizeof(uint16_t));
	//uint16_t *device_ptr = pictures_device + (gpu_buffer_head*width*height);
	//Asynchronous Part

    HANDLE_ERROR(cudaMemcpyAsync(device_ptr ,frame->image_data_ptr,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,std_dev_stream)); 	//Incrementally copies data to device (as each frame comes in it gets copied
	if(cudaSuccess == cudaStreamQuery(std_dev_stream))
	{
		printf("really weird\n");
	}

	if(cudaSuccess == std_dev_stream_status)
	{
		if(prevFrame != NULL)
		{
			prevFrame->has_valid_std_dev = 2; //Ready to display
		}

		frame->has_valid_std_dev = 1; //is processing
		prevFrame = frame;
		dim3 blockDims(BLOCK_SIDE,BLOCK_SIDE,1);
		dim3 gridDims(width/blockDims.x, height/blockDims.y,1);

		HANDLE_ERROR(cudaMemsetAsync(histogram_out_device,0,NUMBER_OF_BINS*sizeof(uint32_t),std_dev_stream));
		//std_dev_filter_kernel <<<gridDims,blockDims,0,std_dev_stream>>> (pictures_device, picture_out_device, histogram_bins_device, histogram_out_device, width, height, gpu_buffer_head, N);
		//printf("launching new std_dev kernel @ count:%i\n",count);
		std_dev_filter_kernel_wrapper(gridDims,blockDims,0,std_dev_stream,pictures_device, picture_out_device, histogram_bins_device, histogram_out_device, width, height, gpu_buffer_head, N);
		HANDLE_ERROR( cudaPeekAtLastError() );
		HANDLE_ERROR(cudaMemcpyAsync(frame->std_dev_data,picture_out_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,std_dev_stream)); //Despite the name, these calls are synchronous w/ respect to the CPU
		HANDLE_ERROR(cudaMemcpyAsync(frame->std_dev_histogram,histogram_out_device,NUMBER_OF_BINS*sizeof(uint32_t),cudaMemcpyDeviceToHost,std_dev_stream));

	}
	//Synchronous again
	if(++gpu_buffer_head == GPU_FRAME_BUFFER_SIZE) //Increment and test for ring buffer overflow
		gpu_buffer_head = 0; //If overflow, than start overwriting the front
	if(currentN < MAX_N) //If the frame buffer has not been fully populated
	{
		currentN++; //Increment how much history is available
	}
	count++;



}
uint16_t * std_dev_filter::getEntireRingBuffer() //For testing only
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	uint16_t * out = new uint16_t[width*height*MAX_N];
	HANDLE_ERROR(cudaMemcpy(out,pictures_device,width*height*sizeof(uint16_t)*MAX_N,cudaMemcpyDeviceToHost));
	return out;
}

std::vector <float> * std_dev_filter::getHistogramBins()
{

	shb.assign(histogram_bins,histogram_bins+NUMBER_OF_BINS);
	return &shb;
}
bool std_dev_filter::outputReady()
{
	return !(currentN < lastN);
}
