#include "constant_filter.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef BYTES_PER_PIXEL
#define BYTES_PER_PIXEL 2
#endif

__global__ void pixel_constant_filter(int16_t magnitude, int width, int height)
{
	int x_offset = blockIdx.x*gridIdx.x +threadIdx.x;
	int y_offset = blockIdx.y*gridIdx.y +threadIdx.y;

	if(x_offset < width && y_offset < height)
	{
		picture_device[x_offset + ]
	}

}
u_char * apply_constant_filter(u_char * picture_in, int width, int height, int16_t filter_coeff)
{
	int pic_size = width*height*BYTES_PER_PIXEL;
	int block_length = 20;

	//u_char * picture_in; //Device (GPU) copy of picture in.
	u_char * picture_out = malloc(pic_size); //Create buffer for CPU memory output

	u_char * picture_device;
	cudaMalloc( (void **)&picture_device, pic_size);

	cudaMemcpy(picture_device, picture_in, pic_size, cudaMemcpyHostToDevice);
	dim3 blockDims(block_length,block_length,1);
	dim3 gridDims(ceil((float)width/block_length), ceil((float)height/block_length));


	pixel_constant_filter<<gridDims,blockDims>>(picture_device, picture_device, filter_coeff, width, height);
	cudaMemcpy(picture_out,picture_device,pic_size,cudaMemcpyDeviceToHost);

	cudaFree(picture_device);

}


