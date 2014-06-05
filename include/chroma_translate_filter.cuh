/*
 * chroma_translate_filter.h
 *
 *  Created on: Jun 3, 2014
 *      Author: nlevy
 */

#ifndef CHROMA_TRANSLATE_FILTER_H_
#define CHROMA_TRANSLATE_FILTER_H_
#include "edtinc.h"
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

class chroma_translate_filter
{
public:
	chroma_translate_filter();
	virtual ~chroma_translate_filter();
	uint16_t * apply_chroma_translate_filter(uint16_t * in);
	cudaStream_t chroma_translate_stream;

private:
	uint16_t * pic_in_host;
	uint16_t * picture_out;
	uint16_t * picture_device;
	uint16_t * pic_out_d;
};


#endif /* CHROMA_TRANSLATE_FILTER_H_ */
