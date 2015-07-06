// #include "math.h"
#include "std_dev_filter_device_code.cuh"

void std_dev_filter_kernel_wrapper(dim3 gd, dim3 bd, unsigned int shm_size, cudaStream_t strm, uint16_t * pic_d, float * picture_out_device, float * histogram_bins, uint32_t * histogram_out, int width, int height, int gpu_buffer_head, int N)
{
	std_dev_filter_kernel <<<gd,bd,shm_size,strm>>> (pic_d, picture_out_device, histogram_bins, histogram_out, width, height, gpu_buffer_head, N);

}
__global__ void std_dev_filter_kernel(uint16_t * pic_d, float * picture_out_device, float * histogram_bins, unsigned int * histogram_out, int width, int height, int gpu_buffer_head, int N)
{

	__shared__ unsigned int block_histogram[NUMBER_OF_BINS];
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = col + row*width;
	register float sum = 0; //Should be put in registers
	register double sq_sum = 0;
	double mean = 0;
	double std_dev;
	int value = 0;
	int c = 0;
/*	if(offset == 100*width && STD_DEV_DEBUG)
	{
		printf("sum: %f sq_sum: %f \n",sum,sq_sum);
	} */
	for(int i = 0; i < N; ++i) {
		if((gpu_buffer_head -i) >= 0)
		{
			value = *(pic_d + offset+(width*height*(gpu_buffer_head-i)));
		}
		else
		{
			value = *(pic_d + offset+(width*height*(GPU_FRAME_BUFFER_SIZE - (i-gpu_buffer_head))));
		}
		sum += value;
		sq_sum += (double)value * (double)value;
		if(offset == 100*width && STD_DEV_DEBUG)
		{
			printf("value @ line 100: %i sum: %f sq_sum: %f\n",value,sum,sq_sum);
		}
	}
	mean = (double)sum / (double)N;
	std_dev = sqrt( ((sq_sum - (double)2*mean*(double)sum) / (double)N ) + mean*mean );
    // std_dev = sqrt(sq_sum / N - mean * mean);
	// std_dev = sqrt( (sq_sum- 2*mean*sum)/ N + mean * mean );
	if(offset == 100*width && STD_DEV_DEBUG)
	{
		printf("mean: %f std_dev: %f @ line 100\n",mean, std_dev);
	}
	if(offset == 100*width && STD_DEV_DEBUG)
	{
		printf("value @ line 100: %i sum: %f sq_sum: %d\n",value,sum,sq_sum);
	}
	picture_out_device[offset] = std_dev;
	//__syncthreads(); //unnecessary?
	int blockArea = blockDim.x*blockDim.y;
	for(int shm_offset = 0; shm_offset < NUMBER_OF_BINS; shm_offset+=blockArea)
	{
		if(shm_offset + threadIdx.y * blockDim.x + threadIdx.x < NUMBER_OF_BINS)
		{
			block_histogram[shm_offset + threadIdx.y * blockDim.x + threadIdx.x] = 0; //Zero shared mem initially.
		}
	}

	/*if(offset == 100*width && STD_DEV_DEBUG)
	{
		for(int i = 0; i < NUMBER_OF_BINS;i++)
		{
			printf("%f ,", histogram_bins[i]);
		}
		printf("\n");

	}*/
	while(std_dev > histogram_bins[c] && c < (NUMBER_OF_BINS-1))
	{
		c++;
	}

	__syncthreads();
	atomicAdd(&block_histogram[c], 1); //calculate sub histogram for each block
	__syncthreads();

	/*if(STD_DEV_DEBUG && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y==0)
	{
		sum = 0;
		printf("threads per block %i \n", blockDim.x *blockDim.y);
		for(int i = 0; i < NUMBER_OF_BINS; i++)
		{
			sum += block_histogram[i];
			printf("%i ",block_histogram[i]);
		}
		printf("\n %f",sum);
	}
*/
	if(threadIdx.x == 0 && threadIdx.y == 0) //Only need to do this once per block
	{
		for(c=0; c < NUMBER_OF_BINS; c++)
		{
			atomicAdd(&histogram_out[c],block_histogram[c]);
		}
	}

}
