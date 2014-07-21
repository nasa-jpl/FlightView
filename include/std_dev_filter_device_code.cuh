/*
 * std_dev_filter_device_code.cuh
 *
 *  Created on: Jul 21, 2014
 *      Author: nlevy
 */
#include <stdint.h>
#include <stdio.h>
#include "constants.h"

static const int STD_DEV_DEBUG = true;
#ifndef STD_DEV_FILTER_DEVICE_CODE_CUH_
#define STD_DEV_FILTER_DEVICE_CODE_CUH_
__global__ void std_dev_filter_kernel(uint16_t * pic_d, float * picture_out_device, float * histogram_bins, uint32_t * histogram_out, int width, int height, int gpu_buffer_head, int N);


void std_dev_filter_kernel_wrapper(dim3 gd, dim3 bd, unsigned int shm_size, cudaStream_t strm, uint16_t * pic_d, float * picture_out_device, float * histogram_bins, uint32_t * histogram_out, int width, int height, int gpu_buffer_head, int N);

#endif /* STD_DEV_FILTER_DEVICE_CODE_CUH_ */
