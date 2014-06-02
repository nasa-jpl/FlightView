#include "std_dev_filter.cuh"
#include "cuda_utils.cuh"
#ifndef BYTES_PER_PIXEL
#define BYTES_PER_PIXEL 2
#endif
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

//This is useful for figuring out how to do caching.
//This number was derived from a "ptxas" error, apparently cuda props lied.
#define SHARED_MEM_PER_BLOCK_GTX590 (0x4000 - 0x10)
#define MAX_N 500


#define THREADS_PER_BLOCK SHARED_MEM_PER_BLOCK_GTX590/(BYTES_PER_PIXEL*MAX_N) //If we want to cache into shared memory, this gives us the maximum number of threads per block should be 24 with truncation
//Kernel code, this runs on the GPU (device) uses shared memory to decrease time
__global__ void std_dev_filter(u_char * pic_d, u_char * picture_out_device, int width, int height, int N)
{
	__shared__ uint16_t cached_block_data [THREADS_PER_BLOCK*MAX_N]; //Should equal 48000Bytes or 24000 uint_16s
	int offset = (blockIdx.x*blockDim.x +threadIdx.x)*BYTES_PER_PIXEL; //This gives us how far we are into the u_char
	uint32_t pic_size = height*width*BYTES_PER_PIXEL; //recalculting this value, integer math is cheaper than I/O
	//Doing allocation outside of if because it reduces forked code size?
	float acc = 0;
	uint32_t sum = 0; //Really hoping all of these get put in registers
	uint32_t mean = 0;
	uint16_t current_val;
	uint16_t std_dev;


	if(offset < width*height*BYTES_PER_PIXEL) //Because we needed an interger grid size, we will have a few threads that don't correspond to a location in the image.
	{

		//Put the device global memory into shared memory (reduces amount of slow memory accesses we need to do), also take the sum for averaging
		for(int i = 0; i<N; i++)
		{
			current_val = pic_d[offset] | (pic_d[offset+1] << 8);
			cached_block_data[i + threadIdx.x*MAX_N] = current_val;
			offset += pic_size;
			sum += current_val;
		}

		mean = sum/N;
		for(int i = 0; i<N; i++)
		{
			acc+=(cached_block_data[i + threadIdx.x*MAX_N] - mean)^2;
		}

		std_dev = (uint16_t) (sqrt(acc/(N-1)));
		//Reset offset
		offset = (blockIdx.x*blockDim.x +threadIdx.x)*BYTES_PER_PIXEL;
//std_dev = 0;
		picture_out_device[offset] =(u_char) std_dev; //We want the LSB here
		picture_out_device[offset + 1] =(u_char) (std_dev >> 8); //We want the MSB here

	}



}
boost::shared_array<u_char> apply_std_dev_filter(boost::circular_buffer<boost::shared_ptr <frame> > frame_buffer, unsigned int N)
				{

	int width = frame_buffer[0]->width; //Making the assumption that all frames in a frame buffer are the same size
	int height = frame_buffer[0]->height;
	int pic_size = width*height*BYTES_PER_PIXEL;
	//std::cout << "threads per block" << THREADS_PER_BLOCK << std::endl;

	u_char * pictures_device;
	u_char * current_picture_device;
	//u_char * picture_out = (u_char * )malloc(pic_size); //Create buffer for CPU memory output

	boost::shared_array<u_char> picture_out(new u_char[pic_size]);

	u_char * picture_out_device;
	if(N <= frame_buffer.size() && N <= MAX_N) //We can't calculate the std. dev farther back in time then we are keeping track.
	{
		HANDLE_ERROR(cudaMalloc( (void **)&pictures_device, pic_size*N)); //Allocate a huge amount of memory on the GPU (N times the size of each frame stored as a u_char)
		current_picture_device = pictures_device;

		HANDLE_ERROR(cudaMalloc( (void **)&picture_out_device, pic_size)); //Allocate memory on GPU for reduce target
		//Copy raw image data to device
		for(int i = 0; i < N; i++)
		{
			HANDLE_ERROR(cudaMemcpy(current_picture_device, frame_buffer[i].get()->image_data_ptr, pic_size,cudaMemcpyHostToDevice));
			current_picture_device += pic_size; //Increment the pointer by the size we just filled
		}

		//Create thread for each pixel
		dim3 blockDims(THREADS_PER_BLOCK,1,1);

		//+1 to account for possible integer-division truncation
		dim3 gridDims((width*height/blockDims.x +1),1,1);

		std_dev_filter<<<gridDims,blockDims>>>(pictures_device, picture_out_device, width, height, N);
		HANDLE_ERROR( cudaPeekAtLastError() );
		HANDLE_ERROR( cudaDeviceSynchronize() );

		HANDLE_ERROR(cudaMemcpy(picture_out.get(),picture_out_device,pic_size,cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaFree(pictures_device)); //do not free current picture because it poitns to a location inside pictures_device
		HANDLE_ERROR(cudaFree(picture_out_device));
		//return picture_out;
		//TODO: Fix this
		//boost::shared_ptr<frame> * result = new boost::shared_ptr<frame>(picture_out,  height, width);
		return picture_out;
		//return result;
	}
	std::cerr << "Couldn't take std. dev, N exceeded length of history or maximum alllowed N (" << MAX_N << ")" << std::endl;

	return picture_out;

				}


