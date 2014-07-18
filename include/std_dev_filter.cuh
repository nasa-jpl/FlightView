/*
 * std_dev_filter.cuh
 *
 *  Created on: May 29, 2014
 *      Author: nlevy
 */

#ifndef STD_DEV_FILTER_CUH_
#define STD_DEV_FILTER_CUH_


#include <stdint.h>
#include <vector>
#include "constants.h"
#include "edtinc.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_utils.cuh"

static const int STD_DEV_DEVICE_NUM = (1 % getDeviceCount());


static const bool DEBUG = false;
class std_dev_filter
{
public:
	std_dev_filter(int nWidth, int nHeight);
	virtual ~std_dev_filter();
	void start_std_dev_filter(unsigned int N, float * std_dev_out, uint32_t * std_dev_histogram);
	void update_GPU_buffer(uint16_t * image_ptr);
	void wait_std_dev();
	bool outputReady();
	float * wait_std_dev_filter();
	uint32_t * wait_std_dev_histogram();
	std::vector<float> * getHistogramBins();
	uint16_t * getEntireRingBuffer(); //For testing only
	cudaStream_t std_dev_stream;
private:
	std_dev_filter() {}//Private defauklt constructor

	//boost::shared_array<float> picture_out;
	//boost::shared_array<uint32_t> hist_data;
	std::vector <float> shb;
	int width;
	int height;
	int lastN;
	int gpu_buffer_head;
	int currentN;

	uint16_t * picture_in_host;
	uint16_t * pictures_device;
	uint16_t * current_picture_device;

	float * picture_out_device;
	float * picture_out_host;
	float histogram_bins[NUMBER_OF_BINS];
	float * histogram_bins_device;

	uint32_t * histogram_out_host;
	uint32_t * histogram_out_device;


	float * std_dev_result;
	uint32_t * histogram_out;
};
#endif /* STD_DEV_FILTER_CUH_ */
