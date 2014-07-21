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


struct frame_c{
	uint16_t * raw_data_ptr;
	uint16_t * image_data_ptr;
	float * dark_subtracted_data;

	float * std_dev_data;
	uint32_t * std_dev_histogram;
	float vertical_mean_profile[MAX_HEIGHT]; //These can use regular C++ allocation because they do not have to deal w/cuda
	float horizontal_mean_profile[MAX_WIDTH];
	float fftMagnitude[FFT_INPUT_LENGTH/2];
	std::atomic_int_least8_t delete_counter; //NOTE!! It is incredibly critical that this number is equal to the number of frameview_widgets that are drawing + the number of mean profiles, too many and memory will leak; too few and invalid data will be displayed.
	std::atomic_int_least8_t async_filtering_done;
	std::atomic_int_least8_t has_valid_std_dev;

	frame_c() {
		delete_counter = 5;
		async_filtering_done = 0;
		has_valid_std_dev = 0;
		HANDLE_ERROR(cudaMallocHost( (void **)&raw_data_ptr, MAX_SIZE*sizeof(uint16_t), cudaHostAllocPortable));
		HANDLE_ERROR(cudaMallocHost( (void **)&dark_subtracted_data, MAX_SIZE*sizeof(float), cudaHostAllocPortable));
		HANDLE_ERROR(cudaMallocHost( (void **)&std_dev_data, MAX_SIZE*sizeof(float), cudaHostAllocPortable));
		HANDLE_ERROR(cudaMallocHost( (void **)&std_dev_histogram, NUMBER_OF_BINS*sizeof(uint32_t), cudaHostAllocPortable));
	};
	~frame_c()
	{
		HANDLE_ERROR(cudaFreeHost(raw_data_ptr));
		HANDLE_ERROR(cudaFreeHost(dark_subtracted_data));
		HANDLE_ERROR(cudaFreeHost(std_dev_data));
		HANDLE_ERROR(cudaFreeHost(std_dev_histogram));

	}
};


#endif /* FRAME_C_HPP_ */
