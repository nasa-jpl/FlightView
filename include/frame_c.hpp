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


struct frame_c{
#ifdef USE_PINNED_MEMORY
	uint16_t * raw_data_ptr;
	float * dark_subtracted_data;

	float * std_dev_data;
	uint32_t * std_dev_histogram;
#else
	uint16_t raw_data_ptr[MAX_SIZE];
	float dark_subtracted_data[MAX_SIZE];
	float std_dev_data[MAX_SIZE];
	uint32_t std_dev_histogram[NUMBER_OF_BINS];
#endif
	uint16_t * image_data_ptr;

	float vertical_mean_profile[MAX_HEIGHT]; //These can use regular C++ allocation because they do not have to deal w/cuda
	float horizontal_mean_profile[MAX_WIDTH];
	float fftMagnitude[FFT_INPUT_LENGTH/2];
	std::atomic_int_least8_t async_filtering_done;
	std::atomic_int_least8_t has_valid_std_dev; //1 indicates doing std. dev, 2 indicates done with std. dev

	frame_c() {
		reset();
#ifdef USE_PINNED_MEMORY
		HANDLE_ERROR(cudaMallocHost( (void **)&raw_data_ptr, MAX_SIZE*sizeof(uint16_t), cudaHostAllocPortable));
		HANDLE_ERROR(cudaMallocHost( (void **)&dark_subtracted_data, MAX_SIZE*sizeof(float), cudaHostAllocPortable));
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
		HANDLE_ERROR(cudaFreeHost(dark_subtracted_data));
		HANDLE_ERROR(cudaFreeHost(std_dev_data));
		HANDLE_ERROR(cudaFreeHost(std_dev_histogram));
#endif

	}
};


#endif /* FRAME_C_HPP_ */
