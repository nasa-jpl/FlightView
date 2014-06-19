/*
 * std_dev_filter.cuh
 *
 *  Created on: May 29, 2014
 *      Author: nlevy
 */

#ifndef STD_DEV_FILTER_CUH_
#define STD_DEV_FILTER_CUH_


#include <stdint.h>
#include <boost/shared_ptr.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/shared_array.hpp>
#include "edtinc.h"
#include "cuda.h"
#include "frame.hpp"
#include "cuda_runtime.h"
static const int STD_DEV_DEVICE_NUM = 5;
static const int MAX_N = 1000;
static const int BLOCK_SIDE = 20;
static const bool DEBUG = false;

class std_dev_filter
{
public:
	std_dev_filter(int nWidth, int nHeight);
	virtual ~std_dev_filter();
	void start_std_dev_filter(int N);
	void update_GPU_buffer(uint16_t * image_ptr);
	boost::shared_array< float > wait_std_dev_filter();
	uint16_t * getEntireRingBuffer(); //For testing only
	cudaStream_t std_dev_stream;
private:
	std_dev_filter() {}//Private defauklt constructor
	boost::shared_array<float> picture_out;
	int width;
	int height;
	int gpu_buffer_head;
	int currentN;
	uint16_t * pictures_device;
	uint16_t * current_picture_device;
	float * picture_out_device;
	float * picture_out_host;
	uint16_t * picture_in_host;

};
#endif /* STD_DEV_FILTER_CUH_ */
