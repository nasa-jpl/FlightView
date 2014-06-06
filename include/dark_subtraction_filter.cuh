/*
 * dark_subtraction_filter.cuh
 *
 *  Created on: May 22, 2014
 *      Author: nlevy
 */

#ifndef DARK_SUBTRACTION_FILTER_CUH_
#define DARK_SUBTRACTION_FILTER_CUH_

#include <stdint.h>
#include "edtinc.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <boost/shared_array.hpp>

class dark_subtraction_filter
{
public:
	dark_subtraction_filter() {};//Useless default constructor
	dark_subtraction_filter(int nWidth, int nHeight);
	virtual ~dark_subtraction_filter();
	void update_dark_subtraction(uint16_t * pic_in);
	boost::shared_array< float > wait_dark_subtraction();
	void start_mask_collection();
	uint32_t update_mask_collection(uint16_t * pic_in);
	void finish_mask_collection();
	cudaStream_t dark_subtraction_stream;
private:
	boost::shared_array<float> picture_out;
	uint16_t width;
	uint16_t height;
	uint32_t averaged_samples;

	uint16_t * pic_in_host;
	uint16_t * picture_device;
	float * mask_device;
	float * result_device;
	float * pic_out_host;

	cudaStream_t dsf_stream;
};

#endif /* DARK_SUBTRACTION_FILTER_CUH_ */
