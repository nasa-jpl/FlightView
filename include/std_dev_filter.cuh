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

#ifdef __cplusplus
extern "C" //Makes this callable from c
#endif

boost::shared_array<float> apply_std_dev_filter(boost::circular_buffer<boost::shared_ptr <frame> > frame_buffer, unsigned int N);


#endif /* STD_DEV_FILTER_CUH_ */
