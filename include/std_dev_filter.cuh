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
#define STD_DEV_DEVICE_NUM 5

class std_dev_filter
{
public:
	std_dev_filter(int nWidth, int nHeight);
	virtual ~std_dev_filter();
	void start_std_dev_filter(int N);
	void update_GPU_buffer(uint16_t * image_ptr);
	boost::shared_array< float > wait_std_dev_filter();
	cudaStream_t std_dev_stream;
private:
	std_dev_filter() {}//Private defauklt constructor
	boost::shared_array<float> picture_out;
	int width;
	int height;
	int gpu_buffer_head;
	int gpu_buffer_tail;
	uint16_t * pictures_device;
	uint16_t * current_picture_device;
	float * picture_out_device;

};
#endif /* STD_DEV_FILTER_CUH_ */
