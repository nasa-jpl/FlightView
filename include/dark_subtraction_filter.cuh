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

#ifdef __cplusplus
extern "C"
#endif

u_char * apply_dark_subtraction_filter(u_char * picture_in, u_char * dark_mask, int width, int height)


#endif /* DARK_SUBTRACTION_FILTER_CUH_ */
