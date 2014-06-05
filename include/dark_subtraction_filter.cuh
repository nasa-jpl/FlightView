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
	dark_subtraction_filter(int nWidth, int nHeight, int nN);
	virtual ~dark_subtraction_filter();
	void start_std_dev_filter(uint16_t * pic_in);
	boost::shared_array< float > wait_std_dev_filter();
	cudaStream_t dark_subtraction_stream;
private:
	dark_subtraction_filter() {};//Private defauklt constructor
	boost::shared_array<float> picture_out;
	uint16_t width;
	uint16_t height;
	int pic_size;
	uint16_t * picture_device;
	uint16_t * mask_device;

};

#endif /* DARK_SUBTRACTION_FILTER_CUH_ */
