#include "std_dev_filter.cuh"
#include "cuda_utils.cuh"
#include <cuda_profiler_api.h>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

//This is useful for figuring out how to do caching.
//This number was derived from a "ptxas" error, apparently cuda props lied.
#define SHARED_MEM_PER_BLOCK_GTX590 (0x4000 - 0x10)
#define MAX_N 500

#define THREADS_PER_BLOCK SHARED_MEM_PER_BLOCK_GTX590/(BYTES_PER_PIXEL*MAX_N) //If we want to cache into shared memory, this gives us the maximum number of threads per block should be 24 with truncation
//Kernel code, this runs on the GPU (device) uses shared memory to decrease time


__global__ void std_dev_filter_kernel(uint16_t * pic_d, float * picture_out_device, int width, int height, int gpu_buffer_head, int gpu_buffer_tail, int N)
{
	//__shared__ uint16_t cached_block_data [THREADS_PER_BLOCK*MAX_N]; //Should equal 48000Bytes or 24000 uint_16s
	int offset = blockIdx.x*blockDim.x +threadIdx.x; //This gives us how far we are into the u_char
	uint32_t pic_size = height*width*sizeof(uint16_t); //recalculting this value, integer math is cheaper than I/O
	//Doing allocation outside of if because it reduces forked code size?
	float acc = 0;
	float sum = 0; //Really hoping all of these get put in registers
	float mean = 0;
	uint16_t current_val;
	float std_dev;


	if(offset < width*height*BYTES_PER_PIXEL) //Because we needed an interger grid size, we will have a few threads that don't correspond to a location in the image.
	{

		//Put the device global memory into shared memory (reduces amount of slow memory accesses we need to do), also take the sum for averaging


		//for(int i = 0; i <N; i)
		for(int i = 0; i<N; i++) //Get the sum
		{
			//sum += cached_block_data[i+threadIdx.x*MAX_N];
			sum += pic_d[(offset + ((gpu_buffer_head + i )*width*height*sizeof(uint16_t)) ) % pic_size*MAX_N];
		}

		mean = sum/N;
		for(int i = 0; i<N; i++)
		{
			//acc += pow((cached_block_data[i + threadIdx.x*MAX_N] - mean),2);
			acc += pow(pic_d[(offset + ((gpu_buffer_head + i )*width*height*sizeof(uint16_t)) ) % pic_size*MAX_N] - mean,2);
		}

		std_dev = sqrt(acc/(N-1));

		picture_out_device[offset] = std_dev;
		__syncthreads();
	}



}

std_dev_filter::std_dev_filter(int nWidth, int nHeight)
{

	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	width = nWidth; //Making the assumption that all frames in a frame buffer are the same size
	height = nHeight;
	gpu_buffer_head = 0;
	gpu_buffer_tail = 0;
	picture_out= boost::shared_array < float >(new float[width*height]);
	HANDLE_ERROR(cudaStreamCreate(&std_dev_stream));
	//std::cout << "threads per block" << THREADS_PER_BLOCK << std::endl;
	HANDLE_ERROR(cudaMalloc( (void **)&pictures_device, width*height*sizeof(uint16_t)*MAX_N)); //Allocate a huge amount of memory on the GPU (N times the size of each frame stored as a u_char)
	HANDLE_ERROR(cudaMalloc( (void **)&picture_out_device, width*height*sizeof(float))); //Allocate memory on GPU for reduce target

}
std_dev_filter::~std_dev_filter()
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	//All the memory leaks!
	HANDLE_ERROR(cudaFree(pictures_device)); //do not free current picture because it poitns to a location inside pictures_device
	HANDLE_ERROR(cudaFree(picture_out_device));
	HANDLE_ERROR(cudaStreamDestroy(std_dev_stream));
}
void std_dev_filter::update_GPU_buffer(boost::circular_buffer<boost::shared_ptr <frame> > frame_buffer)
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));

	if(frame_buffer.size() > 1)
	{
		//Incrementally copies data to device (as each frame comes in it gets copied
		HANDLE_ERROR(cudaMemcpyAsync(pictures_device + (gpu_buffer_head*width*height*sizeof(uint16_t)),frame_buffer[1].get()->image_data_ptr,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,std_dev_stream));
		if(++gpu_buffer_head == MAX_N) //Increment and test for ring buffer overflow
			gpu_buffer_head = 0;
		if(frame_buffer.size() == MAX_N) //If the frame buffer has been fully populated
		{
			if(++gpu_buffer_tail == MAX_N) //Increment and test for ring buffer overflow
			{
				gpu_buffer_tail = 0;
			}
		}
	}

}
void std_dev_filter::start_std_dev_filter(int N)
{

	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));

	if(N < MAX_N) //We can't calculate the std. dev farther back in time then we are keeping track.
	{

		//Create thread for each pixel
		dim3 blockDims(THREADS_PER_BLOCK,1,1);

		//+1 to account for possible integer-division truncation
		dim3 gridDims((width*height/blockDims.x +1),1,1);
		uint16_t kernelN = gpu_buffer_head >= gpu_buffer_tail ? gpu_buffer_head - gpu_buffer_tail : gpu_buffer_head - (-gpu_buffer_tail); //account for gpu_buffer_head wraparound
		//std::cout << " kernelN" << kernelN << std::endl;
		std_dev_filter_kernel <<<gridDims,blockDims,0,std_dev_stream>>> (pictures_device, picture_out_device, width, height, gpu_buffer_head, gpu_buffer_tail, N);
		HANDLE_ERROR( cudaPeekAtLastError() );

		HANDLE_ERROR(cudaMemcpyAsync(picture_out.get(),picture_out_device,width*height*sizeof(uint16_t),cudaMemcpyDeviceToHost,std_dev_stream));
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
boost::shared_array <float> std_dev_filter::wait_std_dev_filter()
{
	HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
	HANDLE_ERROR(cudaStreamSynchronize(std_dev_stream)); //blocks until done
	HANDLE_ERROR( cudaPeekAtLastError() );

	return picture_out;
}


