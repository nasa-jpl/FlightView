#include "std_dev_filter.cuh"
#include "cuda_utils.cuh"
#include "constants.h"
#include <cuda_profiler_api.h>
#include <math.h>
#include <iostream>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
#define DO_HISTOGRAM

#ifdef DO_HISTOGRAM
__global__ void std_dev_filter_kernel(uint16_t * pic_d, float * picture_out_device, float * histogram_bins, uint32_t * histogram_out, int width, int height, int gpu_buffer_head, int N)
#else
__global__ void std_dev_filter_kernel(uint16_t * pic_d, float * picture_out_device, int width, int height, int gpu_buffer_head, int N)
#endif
{
#ifdef DO_HISTOGRAM
	__shared__ int block_histogram[NUMBER_OF_BINS];
#endif
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = col + row*width;
	int c = 0;
	float sum = 0; //Should be put in registers
	float sq_sum = 0;
	float mean = 0;
	float std_dev;
	int value = 0;

	if(offset == 100*width && DEBUG)
	{
		printf("sum: %f sq_sum: %f \n",sum,sq_sum);
	}
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
		sq_sum += value * value;
		if(offset == 100*width && DEBUG)
		{
			printf("value @ line 100: %i sum: %f sq_sum: %f \n",value,sum,sq_sum);
		}
	}
	mean = sum / N;
	std_dev = sqrt(sq_sum / N - mean * mean);
	if(offset == 100*width && DEBUG)
	{
		printf("mean: %f std_dev: %f @ line 100",mean, std_dev);
	}
	picture_out_device[offset] = std_dev;

#ifdef DO_HISTOGRAM
	//__syncthreads(); //unnecessary?
	int blockArea = blockDim.x*blockDim.y;
	for(int shm_offset = 0; shm_offset < NUMBER_OF_BINS; shm_offset+=blockArea)
	{
		if(shm_offset + threadIdx.y * blockDim.x + threadIdx.x < NUMBER_OF_BINS)
		{
			block_histogram[shm_offset + threadIdx.y * blockDim.x + threadIdx.x] = 0; //Zero shared mem initially.
		}
	}
	if(offset == 100*width && DEBUG)
	{
		for(int i = 0; i < NUMBER_OF_BINS;i++)
		{
			printf("%f ,", histogram_bins[i]);
		}
		printf("\n");

	}
	while(std_dev > histogram_bins[c] && c < (NUMBER_OF_BINS-1))
	{
		c++;
	}
	__syncthreads();
	atomicAdd(&block_histogram[c], 1); //calculate sub histogram for each block
	__syncthreads();

	if(DEBUG && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y==0)
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

	if(threadIdx.x == 0 && threadIdx.y == 0) //Only need to do this once per block
	{
		for(c=0; c < NUMBER_OF_BINS; c++)
		{
			atomicAdd(&histogram_out[c],block_histogram[c]);
		}
	}

#endif

}
std_dev_filter::std_dev_filter(int nWidth, int nHeight)
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	width = nWidth; //Making the assumption that all frames in a frame buffer are the same size
	height = nHeight;
	gpu_buffer_head = 0;
	currentN = 0;
	HANDLE_ERROR(cudaStreamCreate(&std_dev_stream));
	HANDLE_ERROR(cudaMalloc( (void **)&pictures_device, width*height*sizeof(uint16_t)*MAX_N)); //Allocate a huge amount of memory on the GPU (N times the size of each frame stored as a u_char)
	HANDLE_ERROR(cudaMalloc( (void **)&picture_out_device, width*height*sizeof(float))); //Allocate memory on GPU for reduce target


	HANDLE_ERROR(cudaMallocHost( (void **)&picture_out_host, width*height*sizeof(float))); //Allocate memory on GPU for reduce target
	HANDLE_ERROR(cudaMallocHost( (void **)&picture_in_host, width*height*sizeof(uint16_t))); //Allocate memory on GPU for reduce target

#ifdef DO_HISTOGRAM
	HANDLE_ERROR(cudaMalloc( (void **)&histogram_out_device, NUMBER_OF_BINS*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMalloc( (void **)&histogram_bins_device, NUMBER_OF_BINS*sizeof(float)));

	HANDLE_ERROR(cudaMallocHost( (void **)&histogram_out_host, NUMBER_OF_BINS*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocHost((void **)&histogram_bins,NUMBER_OF_BINS*sizeof(float)));

	//Calculate logarithmic bins
	//float increment = (UINT16_MAX - 0)/NUMBER_OF_BINS;
	float max = log((1<<16)); //ln(2^16)+.1
	float increment = (max - 0)/(NUMBER_OF_BINS);
	float acc = 0;
	for(unsigned int i = 0; i < NUMBER_OF_BINS; i++)
	{
		histogram_bins[i] = exp(acc)-1;
		acc+=increment;
		//printf("%f, ",histogram_bins[i]);
	}
	printf("\ncreated logarithmic bins\n");
	HANDLE_ERROR(cudaMemcpyAsync(histogram_bins_device ,histogram_bins,NUMBER_OF_BINS*sizeof(float),cudaMemcpyHostToDevice,std_dev_stream)); 	//Incrementally copies data to device (as each frame comes in it gets copied
#endif
}
std_dev_filter::~std_dev_filter()
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	HANDLE_ERROR(cudaFree(pictures_device)); //do not free current picture because it poitns to a location inside pictures_device
	HANDLE_ERROR(cudaFree(picture_out_device));
	HANDLE_ERROR(cudaFreeHost(picture_out_host));
	HANDLE_ERROR(cudaFreeHost(picture_in_host));

#ifdef DO_HISTOGRAM
	HANDLE_ERROR(cudaFree(histogram_out_device));
	HANDLE_ERROR(cudaFree(histogram_bins_device));

	HANDLE_ERROR(cudaFreeHost(histogram_out_host));
	HANDLE_ERROR(cudaFreeHost(histogram_bins));
#endif
	HANDLE_ERROR(cudaStreamDestroy(std_dev_stream));
}

void std_dev_filter::update_GPU_buffer(frame_c * frame, unsigned int N)
{
	//Synchronous Part
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));

	cudaError std_dev_stream_status = cudaStreamQuery(std_dev_stream);

	char *device_ptr = ((char *)(pictures_device)) + (gpu_buffer_head*width*height*sizeof(uint16_t));

	//Asynchronous Part
	HANDLE_ERROR(cudaMemcpyAsync(device_ptr ,frame->image_data_ptr,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,std_dev_stream)); 	//Incrementally copies data to device (as each frame comes in it gets copied
	if(cudaSuccess == std_dev_stream_status)
	{
		dim3 blockDims(BLOCK_SIDE,BLOCK_SIDE,1);
		dim3 gridDims(width/blockDims.x, height/blockDims.y,1);

		HANDLE_ERROR(cudaMemsetAsync(histogram_out_device,0,NUMBER_OF_BINS*sizeof(uint32_t),std_dev_stream));
		std_dev_filter_kernel <<<gridDims,blockDims,0,std_dev_stream>>> (pictures_device, picture_out_device, histogram_bins_device, histogram_out_device, width, height, gpu_buffer_head, N);
		HANDLE_ERROR( cudaPeekAtLastError() );
		HANDLE_ERROR(cudaMemcpyAsync(std_dev_result,picture_out_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,std_dev_stream));

	}
	//Synchronous again
	if(++gpu_buffer_head == GPU_FRAME_BUFFER_SIZE) //Increment and test for ring buffer overflow
		gpu_buffer_head = 0; //If overflow, than start overwriting the front
	if(currentN < MAX_N) //If the frame buffer has not been fully populated
	{
		currentN++; //Increment how much history is available
	}


}
/*
void std_dev_filter::start_std_dev_filter(unsigned int N, float * std_dev_out, uint32_t * std_dev_histogram)
{
	std_dev_result = std_dev_out;
	histogram_out = std_dev_histogram;
	lastN = N;
	//Start thread for doing histogram
	//histogram_thread = boost::thread(&std_dev_filter::doHistogram, this);

	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	if(N < MAX_N && N<= currentN) //We can't calculate the std. dev farther back in time then we are keeping track.
	{


		//Asynchronous Part
#ifdef DO_HISTOGRAM
		HANDLE_ERROR(cudaMemsetAsync(histogram_out_device,0,NUMBER_OF_BINS*sizeof(uint32_t),std_dev_stream));
		std_dev_filter_kernel <<<gridDims,blockDims,0,std_dev_stream>>> (pictures_device, picture_out_device, histogram_bins_device, histogram_out_device, width, height, gpu_buffer_head, N);
		//__global__ void std_dev_filter_kernel(uint16_t * pic_d, float * picture_out_device, float * histogram_bins, uint32_t * histogram_out, int width, int height, int gpu_buffer_head, int N)
		HANDLE_ERROR( cudaPeekAtLastError() );

		HANDLE_ERROR(cudaMemcpyAsync(histogram_out,histogram_out_device,NUMBER_OF_BINS*sizeof(uint32_t),cudaMemcpyDeviceToHost,std_dev_stream));
#else
		std_dev_filter_kernel <<<gridDims,blockDims,0,std_dev_stream>>> (pictures_device, picture_out_device, width, height, gpu_buffer_head, N);
#endif
		HANDLE_ERROR(cudaMemcpyAsync(std_dev_result,picture_out_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,std_dev_stream));
	}
	else
	{
		//std::cerr << "Couldn't take std. dev, N (" << N << " ) exceeded length of history (" <<currentN << ") or maximum alllowed N (" << MAX_N << ")" << std::endl;
		//std::fill_n(picture_out.get(),width*height,-1); //Fill with -1 to indicate fail
	}

}
*/
uint16_t * std_dev_filter::getEntireRingBuffer() //For testing only
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	uint16_t * out = new uint16_t[width*height*MAX_N];
	HANDLE_ERROR(cudaMemcpy(out,pictures_device,width*height*sizeof(uint16_t)*MAX_N,cudaMemcpyDeviceToHost));
	return out;
}

std::vector <float> * std_dev_filter::getHistogramBins()
{

	shb.assign(histogram_bins,histogram_bins+NUMBER_OF_BINS);
	return &shb;
}
bool std_dev_filter::outputReady()
{
	return !(currentN < lastN);
}
