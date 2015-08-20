/*
 * frame_c.hpp
 *
 *  Created on: Jul 15, 2014
 *      Author: nlevy
 */
#include <atomic>
#include "constants.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_utils.cuh"
#ifndef FRAME_C_HPP_
#define FRAME_C_HPP_
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

#define USE_PINNED_MEMORY

/*! \brief The data structure which contains all data for a frame.
 *
 * The memory for a frame is page-locked at the host to save time during memory transfers to the device. By defining the macro
 * USE_PINNED_MEMORY, we are specifying to use heap arrays for the raw data and standard deviation data (which is filtered on the
 * device) using a page-locked format. The GPU uses this format by default. The other memory can be allocated as static arrays.
 * This procedure is standard as defined by the CUDA manual.
 */

struct frame_c{
#ifdef USE_PINNED_MEMORY
	uint16_t * raw_data_ptr;

	float * std_dev_data;
	uint32_t * std_dev_histogram;
#else
	uint16_t raw_data_ptr[MAX_SIZE];
	float std_dev_data[MAX_SIZE];
	uint32_t std_dev_histogram[NUMBER_OF_BINS];
#endif
	uint16_t * image_data_ptr;

	float dark_subtracted_data[MAX_SIZE];
    float vertical_mean_profile[MAX_HEIGHT]; //These can use regular C++ allocation because they do not have to deal w/cuda
    float horizontal_mean_profile[MAX_WIDTH];
    float fftMagnitude[FFT_INPUT_LENGTH/2];
	std::atomic_int_least8_t async_filtering_done;
    std::atomic_int_least8_t has_valid_std_dev; //1 indicates doing std. dev, 2 indicates done with std. dev

	frame_c() {
		reset();
#ifdef USE_PINNED_MEMORY
		HANDLE_ERROR(cudaMallocHost( (void **)&raw_data_ptr, MAX_SIZE*sizeof(uint16_t), cudaHostAllocPortable));
		HANDLE_ERROR(cudaMallocHost( (void **)&std_dev_data, MAX_SIZE*sizeof(float), cudaHostAllocPortable));
		HANDLE_ERROR(cudaMallocHost( (void **)&std_dev_histogram, NUMBER_OF_BINS*sizeof(uint32_t), cudaHostAllocPortable));
#endif
	}
	void reset()
	{
		async_filtering_done = 0;
        has_valid_std_dev = 0;
	}


	~frame_c()
	{
#ifdef USE_PINNED_MEMORY
		HANDLE_ERROR(cudaFreeHost(raw_data_ptr));
		HANDLE_ERROR(cudaFreeHost(std_dev_data));
		HANDLE_ERROR(cudaFreeHost(std_dev_histogram));
#endif

	}
};

#endif /* FRAME_C_HPP_ */
