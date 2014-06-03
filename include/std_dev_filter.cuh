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

__global__ void std_dev_filter_kernel(u_char * pic_d, float * picture_out_device, int width, int height, int N);
class std_dev_filter
{
public:
	std_dev_filter(int nWidth, int nHeight, int nN);
	virtual ~std_dev_filter();
	void start_std_dev_filter(boost::circular_buffer<boost::shared_ptr <frame> > frame_buffer);
	boost::shared_array< float > wait_std_dev_filter();
	cudaStream_t std_dev_stream;
private:
	std_dev_filter() {}//Private defauklt constructor
	boost::shared_array<float> picture_out;
	int width;
	int height;
	int pic_size;
	int N;
	u_char * pictures_device;
	u_char * current_picture_device;
	float * picture_out_device;

};
#endif /* STD_DEV_FILTER_CUH_ */
