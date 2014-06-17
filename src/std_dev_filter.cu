#include "std_dev_filter.cuh"
#include "cuda_utils.cuh"
#include <cuda_profiler_api.h>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))




__global__ void std_dev_filter_kernel(uint16_t * pic_d, float * picture_out_device, int width, int height, int gpu_buffer_head, int N)
{
	//__shared__ uint16_t cached_block_data [THREADS_PER_BLOCK*MAX_N]; //Should equal 48000Bytes or 24000 uint_16s


	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int offset = col + row*width;
	float sum = 0; //Really hoping all of these get put in registers
	float sq_sum = 0;
	float mean = 0;
	float std_dev;
	int value = 0;
	if(offset < width*height) //Because we needed an interger grid size, we will have a few threads that don't correspond to a location in the image.
	{
		if(offset == 100*width)
		{
			printf("sum: %f sq_sum: %f \n",sum,sq_sum);
		}
		for(int i = 0; i < N; ++i) {
			//index = (gpu_buffer_head - i) < 0 ? (gpu_buffer_head + MAX_N - i)*width*height*sizeof(uint16_t) + offset : (gpu_buffer_head-i)*width*height*sizeof(uint16_t) + offset;
			//index = (gpu_buffer_head-i)*width*height*sizeof(uint16_t) + offset;
			if((gpu_buffer_head -i) >= 0)
			{
				value = *(pic_d + offset+(width*height*(gpu_buffer_head-i)));
			}
			else
			{
				if(offset == 100*width)
				{
					printf("MAX_N: %i GPU_BUF_HEAD: %i i: %i",MAX_N, gpu_buffer_head, i);
				}
				value = *(pic_d + offset+(width*height*(MAX_N - (i-gpu_buffer_head))));
				//value = 0;
			}
			//value = pic_d[index];

			sum += value;
			sq_sum += value * value;
			if(offset == 100*width)
			{
				printf("value @ line 100: %i sum: %f sq_sum: %f \n",value,sum,sq_sum);
			}
			//printf("value %i\n",value);
		}
		mean = sum / N;
		std_dev = sqrt(sq_sum / N - mean * mean);
		if(offset == 100*width)
		{
			printf("mean: %f std_dev: %f @ line 100",mean, std_dev);
		}
		picture_out_device[offset] = std_dev;
	}



}

std_dev_filter::std_dev_filter(int nWidth, int nHeight)
{

	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	width = nWidth; //Making the assumption that all frames in a frame buffer are the same size
	height = nHeight;
	gpu_buffer_head = 0;
	currentN = 0;
	picture_out= boost::shared_array < float >(new float[width*height]);
	HANDLE_ERROR(cudaStreamCreate(&std_dev_stream));
	//std::cout << "threads per block" << THREADS_PER_BLOCK << std::endl;
	HANDLE_ERROR(cudaMalloc( (void **)&pictures_device, width*height*sizeof(uint16_t)*MAX_N)); //Allocate a huge amount of memory on the GPU (N times the size of each frame stored as a u_char)
	HANDLE_ERROR(cudaMalloc( (void **)&picture_out_device, width*height*sizeof(float))); //Allocate memory on GPU for reduce target
	HANDLE_ERROR(cudaMallocHost( (void **)&picture_out_host, width*height*sizeof(float))); //Allocate memory on GPU for reduce target
	HANDLE_ERROR(cudaMallocHost( (void **)&picture_in_host, width*height*sizeof(uint16_t))); //Allocate memory on GPU for reduce target

}
std_dev_filter::~std_dev_filter()
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	//All the memory leaks!
	HANDLE_ERROR(cudaFree(pictures_device)); //do not free current picture because it poitns to a location inside pictures_device
	HANDLE_ERROR(cudaFree(picture_out_device));
	HANDLE_ERROR(cudaFreeHost(picture_out_host));
	HANDLE_ERROR(cudaStreamDestroy(std_dev_stream));
}
void std_dev_filter::update_GPU_buffer(uint16_t * image_ptr)
{
	//Synchronous Part
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	memcpy(picture_in_host, image_ptr,width*height*sizeof(uint16_t));
	char *device_ptr = ((char *)(pictures_device)) + (gpu_buffer_head*width*height*sizeof(uint16_t));
	HANDLE_ERROR(cudaMemcpy(device_ptr ,picture_in_host,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice)); 	//Incrementally copies data to device (as each frame comes in it gets copied
	if(++gpu_buffer_head == MAX_N) //Increment and test for ring buffer overflow
		gpu_buffer_head = 0; //If overflow, than start overwriting the front
	if(currentN < MAX_N) //If the frame buffer has not been fully populated
	{
		currentN++; //Increment how much history is available
	}


}
void std_dev_filter::start_std_dev_filter(int N)
{

	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	if(N < MAX_N && N<= currentN) //We can't calculate the std. dev farther back in time then we are keeping track.
	{

		//Create thread for each pixel
		//dim3 blockDims(THREADS_PER_BLOCK,1,1);
		dim3 blockDims(BLOCK_SIDE,BLOCK_SIDE,1);
		//+1 to account for possible integer-division truncation
		dim3 gridDims(width/blockDims.x, height/blockDims.y,1);
		//std::cout << " kernelN" << kernelN << std::endl;
		//	std_dev_filter_kernel <<<gridDims,blockDims,0,std_dev_stream>>> (pictures_device, picture_out_device, width, height, gpu_buffer_head, gpu_buffer_tail, N);
		std_dev_filter_kernel <<<gridDims,blockDims>>> (pictures_device, picture_out_device, width, height, gpu_buffer_head, N);

		HANDLE_ERROR( cudaPeekAtLastError() );
		//HANDLE_ERROR(cudaMemcpyAsync(picture_out.get(),picture_out_device,width*height*sizeof(uint16_t),cudaMemcpyDeviceToHost,std_dev_stream));
		//HANDLE_ERROR(cudaMemcpy(picture_out.get(),picture_out_device,width*height*sizeof(float),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(picture_out_host,picture_out_device,width*height*sizeof(float),cudaMemcpyDeviceToHost));

		HANDLE_ERROR( cudaPeekAtLastError() );

		//cudaProfilerStop();

		//return std_dev_stream;
		//return result;

	}
	else
	{
		//std::cerr << "Couldn't take std. dev, N exceeded length of history or maximum alllowed N (" << MAX_N << ")" << std::endl;
		std::fill_n(picture_out.get(),width*height,-1); //Fill with -1 to indicate fail
	}

}
uint16_t * std_dev_filter::getEntireRingBuffer() //For testing only
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	uint16_t * out = new uint16_t[width*height*MAX_N];
	HANDLE_ERROR(cudaMemcpy(out,pictures_device,width*height*sizeof(uint16_t)*MAX_N,cudaMemcpyDeviceToHost));
	return out;
}
boost::shared_array <float> std_dev_filter::wait_std_dev_filter()
{

	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	memcpy(picture_out.get(),picture_out_host,width*height*sizeof(float));

	HANDLE_ERROR(cudaStreamSynchronize(std_dev_stream)); //blocks until done
	HANDLE_ERROR( cudaPeekAtLastError() );

	return picture_out;
}


