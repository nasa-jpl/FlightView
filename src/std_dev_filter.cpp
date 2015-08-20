#include "std_dev_filter.hpp"
#include "cuda_utils.cuh"
#include "constants.h"
#include <cuda_profiler_api.h>
#include <math.h>
#include <iostream>

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

std_dev_filter::std_dev_filter(int nWidth, int nHeight)
{
    /*! \brief Allocate memory and specify device.
     * \param nWidth The frame width. This is specified initially and cannot be changed during operation.
     * \param nHeight The frame height. This is speccified initially and cannot be changed during operation.
     *
     * To set up the kernel, we have to first allocate memory on the device to which we can copy incoming frames. As the standard
     * deviation calculation is split into two components (The std. dev. image itself and the histogram), there are two separate
     * allocation steps. Additionally, within these steps, any memory specified for input must also have an associated array for output.
     *
     * Finally, the memory for the histogram is specified to copied from the device to the host asynchronously as frames come in.
     */
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
    width = nWidth; // Making the assumption that all frames in a frame buffer are the same size
    height = nHeight;
    gpu_buffer_head = 0; // read point for the GPU ring buffer data structure
    currentN = 0; // number of complete frames in the GPU ring buffer

	HANDLE_ERROR(cudaStreamCreate(&std_dev_stream));
    HANDLE_ERROR(cudaMalloc( (void **)&pictures_device, width*height*sizeof(uint16_t)*GPU_FRAME_BUFFER_SIZE)); // Allocate a huge amount of memory on the GPU (N times the size of each frame stored as a u_char)
    HANDLE_ERROR(cudaMalloc( (void **)&picture_out_device, width*height*sizeof(float))); // Allocate memory on GPU for reduce target

	HANDLE_ERROR(cudaMalloc( (void **)&histogram_bins_device, NUMBER_OF_BINS*sizeof(float)));
    HANDLE_ERROR(cudaMalloc( (void **)&histogram_out_device, NUMBER_OF_BINS*sizeof(uint32_t)));
	memcpy(histogram_bins,getHistogramBinValues().data(),NUMBER_OF_BINS*sizeof(float));

    HANDLE_ERROR(cudaMemcpyAsync(histogram_bins_device,histogram_bins,NUMBER_OF_BINS*sizeof(float),cudaMemcpyHostToDevice,std_dev_stream)); // Incrementally copies data to device (as each frame comes in it gets copied)
}
std_dev_filter::~std_dev_filter()
{
    /*! Free all devices and allocated memory (except the current and picture, and set the device strame to be destroyed. */
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
    HANDLE_ERROR(cudaFree(pictures_device)); // Do not free current picture because it points to a location inside pictures_device
	HANDLE_ERROR(cudaFree(picture_out_device));
	HANDLE_ERROR(cudaFree(histogram_out_device));
	HANDLE_ERROR(cudaFree(histogram_bins_device));
	HANDLE_ERROR(cudaStreamDestroy(std_dev_stream));
}

void std_dev_filter::update_GPU_buffer(frame_c * frame, unsigned int N)
{
    /*! \brief CPU code for launching the kernel and copying over the result of the standard deviation calculation.
     * \param frame The current frame to be worked on.
     * \param N The number of frames to use in the buffer, or the integration length of the calculation.
     */
	static int count = 0;

    // Synchronous
    /*! Step 1: Set the device, get the status, and create a pointer to the current position on the device ring buffer. */
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	cudaError std_dev_stream_status = cudaStreamQuery(std_dev_stream);
    char *device_ptr = ((char *)(pictures_device)) + (gpu_buffer_head*width*height*sizeof(uint16_t));

    // Asynchronous
    /*! Step 2: Copy the current image on the host to the device ring buffer. */
    HANDLE_ERROR(cudaMemcpyAsync(device_ptr,frame->image_data_ptr,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,std_dev_stream)); // Incrementally copies data to device (as each frame comes in it gets copied)

    if(cudaSuccess == cudaStreamQuery(std_dev_stream) && DEBUG)
	{
        printf("really weird\n"); // Noah wrote this debug line. I'm not sure when or why it triggers...
	}

	if(cudaSuccess == std_dev_stream_status)
	{
        /*! Step 3: If there are no errors, check that there are std. dev. frames ready to be displayed */
		if(prevFrame != NULL)
		{
            prevFrame->has_valid_std_dev = 2; // Ready to display
		}

        frame->has_valid_std_dev = 1; // is processing
		prevFrame = frame;

        /*! Step 4: Set the number of blocks and the number of threads per block */
        dim3 blockDims(BLOCK_SIZE,BLOCK_SIZE,1); // We have 2-dimensional blocks of 20x20 threads... These threads will share their "block_histogram" array on the device
        dim3 gridDims(width/blockDims.x, height/blockDims.y,1); // Determine the number of blocks needed for the image

        /*! Step 5: Initialize the histogram output array. */
		HANDLE_ERROR(cudaMemsetAsync(histogram_out_device,0,NUMBER_OF_BINS*sizeof(uint32_t),std_dev_stream));

        /*! Step 6: Launch the kernel using the wrapper function defined in the device code. */
		std_dev_filter_kernel_wrapper(gridDims,blockDims,0,std_dev_stream,pictures_device, picture_out_device, histogram_bins_device, histogram_out_device, width, height, gpu_buffer_head, N);

        /*! Step 7: Check for errors and copy the output arrays off the device. */
        HANDLE_ERROR(cudaPeekAtLastError());
		HANDLE_ERROR(cudaMemcpyAsync(frame->std_dev_data,picture_out_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,std_dev_stream)); //Despite the name, these calls are synchronous w/ respect to the CPU
		HANDLE_ERROR(cudaMemcpyAsync(frame->std_dev_histogram,histogram_out_device,NUMBER_OF_BINS*sizeof(uint32_t),cudaMemcpyDeviceToHost,std_dev_stream));
	}

    // Synchronous
    /*! Step 8: Increment the ring buffer and the current number of frames in the buffer, if applicable. As this is a ring buffer, the
     * gpu_buffer_head will return to the beginning of the array when it reaches the end of the allocated space.
     */
	if(++gpu_buffer_head == GPU_FRAME_BUFFER_SIZE) //Increment and test for ring buffer overflow
        gpu_buffer_head = 0; // If overflow, than start overwriting the front
    if(currentN < MAX_N) // If the frame buffer has not been fully populated
	{
		currentN++; //Increment how much history is available
	}
	count++;
}
uint16_t * std_dev_filter::getEntireRingBuffer() //For testing only
{
    /*! Captures the ring buffer of standard deviation frames. */
    HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	uint16_t * out = new uint16_t[width*height*MAX_N];
	HANDLE_ERROR(cudaMemcpy(out,pictures_device,width*height*sizeof(uint16_t)*MAX_N,cudaMemcpyDeviceToHost));
	return out;
}
std::vector <float> * std_dev_filter::getHistogramBins()
{
    /*! Captures all current histogram bins. */
	shb.assign(histogram_bins,histogram_bins+NUMBER_OF_BINS);
	return &shb;
}
bool std_dev_filter::outputReady()
{
    /*! Returns true if std. dev. frames are ready to be plotted. */
	return !(currentN < lastN);
}
