/*
 * constant_filter.h
 *
 *  Created on: May 21, 2014
 *      Author: nlevy
 */


#ifndef CONSTANT_FILTER_H_
#define CONSTANT_FILTER_H_

#include <stdint.h>
#include "edtinc.h"
#include "cuda.h"
#include "cuda_runtime.h"


extern "C"{
u_char * apply_constant_filter(u_char * picture_in, int width, int height, int16_t filter_coeff);
};

#endif /* CONSTANT_FILTER_H_ */
