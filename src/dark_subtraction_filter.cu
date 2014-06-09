#include "dark_subtraction_filter.cuh"

//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include "cuda_utils.cuh"
#include <stdlib.h>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
#define DSF_DEVICE_NUM 2
#define BLOCK_SIDE 20
//Kernel code, this runs on the GPU (device)
__global__ void apply_mask(uint16_t * pic_d, float * mask_d, float * result_d, uint16_t width, uint16_t height)
{
	unsigned short col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned short row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned short offset = col + row*width;
	result_d[offset] = pic_d[offset] - mask_d[offset];

}
__global__ void sum_mask(uint16_t * pic_d, float * mask_d,uint16_t width, uint16_t height)
{
	unsigned short col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned short row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned short offset = col + row*width;
	mask_d[offset] = mask_d[offset] + (float) pic_d[offset];

}
__global__ void avg_mask(float * mask_d, uint32_t num_samples, uint16_t width, uint16_t height)
{
	unsigned short col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned short row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned short offset = col + row*width;
	mask_d[offset] = mask_d[offset]/num_samples;

}

void dark_subtraction_filter::start_mask_collection()
{
	//HANDLE_ERROR(cudaSetDevice(4));

	averaged_samples = 0; //Synchronous
	HANDLE_ERROR(cudaPeekAtLastError());
	//HANDLE_ERROR(cudaMemset(mask_device,(char) 0,width*height*sizeof(float)));
	HANDLE_ERROR(cudaMemsetAsync(mask_device,(char) 0,width*height*sizeof(float),dsf_stream)); //Asynchronous
}

uint32_t dark_subtraction_filter::update_mask_collection(uint16_t * pic_in)
{
	// Synchronous
	HANDLE_ERROR(cudaSetDevice(4));

	//std::cout << pic_in[100000] << std::endl;
	//std::cout << pic_in_host[100000] << std::endl;
	HANDLE_ERROR(cudaMemcpy(pic_in_host,pic_in,width*height*sizeof(uint16_t),cudaMemcpyHostToHost));
	averaged_samples++;

	//Asynchronous
	HANDLE_ERROR(cudaMemcpyAsync(picture_device,pic_in_host,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,dsf_stream));
	dim3 blockDims(BLOCK_SIDE,BLOCK_SIDE,1);
	dim3 gridDims(width/BLOCK_SIDE, height/BLOCK_SIDE,1);
	sum_mask<<< gridDims, blockDims,0,dsf_stream>>>(picture_device, mask_device,width,height);
	return averaged_samples;
}
void dark_subtraction_filter::finish_mask_collection()
{
	HANDLE_ERROR(cudaSetDevice(4));
	cudaStreamSynchronize(dsf_stream);
	dim3 blockDims(BLOCK_SIDE,BLOCK_SIDE,1);
	dim3 gridDims(width/BLOCK_SIDE, height/BLOCK_SIDE,1);
	avg_mask<<< gridDims, blockDims,0,dsf_stream>>>(mask_device, averaged_samples,width,height);
}

void dark_subtraction_filter::update_dark_subtraction(uint16_t * pic_in)
{
	HANDLE_ERROR(cudaSetDevice(4));

	//Synchronous
	memcpy(pic_in_host,pic_in,width*height*sizeof(uint16_t));

	//Asynchronous
	HANDLE_ERROR(cudaMemcpyAsync(picture_device,pic_in_host,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,dsf_stream));
	dim3 blockDims(BLOCK_SIDE,BLOCK_SIDE,1);
	dim3 gridDims(width/BLOCK_SIDE, height/BLOCK_SIDE,1);
	apply_mask<<< gridDims, blockDims,0,dsf_stream>>>(picture_device, mask_device, result_device, width,height);
	HANDLE_ERROR(cudaMemcpyAsync(pic_out_host,result_device,width*height*sizeof(uint16_t),cudaMemcpyDeviceToHost,dsf_stream));

}
boost::shared_array< float > dark_subtraction_filter::wait_dark_subtraction()
{
	HANDLE_ERROR(cudaSetDevice(4));

	// Synchronous
	HANDLE_ERROR(cudaStreamSynchronize(dsf_stream));
	memcpy(picture_out.get(),pic_out_host,width*height*sizeof(float));
	return picture_out;
}
/*
	dark_subtraction_filter() {};//Private defauklt constructor
	boost::shared_array<float> picture_out;
	uint16_t width;
	uint16_t height;
	int pic_size;
	uint16_t * pic_in_host;
	uint16_t * picture_device;
	float * mask_device;
 */

dark_subtraction_filter::dark_subtraction_filter(int nWidth, int nHeight)
{
	HANDLE_ERROR(cudaSetDevice(4));

	width=nWidth;
	height=nHeight;

	picture_out = boost::shared_array<float>(new float[width*height]);
	HANDLE_ERROR(cudaStreamCreate(&dsf_stream));

	//cudaThreadExit(); // clears all the runtime state for the current thread
	//HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM)); // explicit set the current device for the other calls
	//HANDLE_ERROR(cudaMallocHost( (void **)&pic_in_host,width*height*sizeof(uint16_t))); //cudaHostAllocPortable??
	HANDLE_ERROR(cudaHostAlloc( (void **)&pic_in_host,width*height*sizeof(uint16_t),cudaHostAllocPortable)); //cudaHostAllocPortable??
	//HANDLE_ERROR(cudaMallocHost( (void **)&pic_out_host,width*height*sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc( (void **)&pic_out_host,width*height*sizeof(float),cudaHostAllocPortable)); //cudaHostAllocPortable??

	HANDLE_ERROR(cudaMalloc( (void **)&picture_device, width*height*sizeof(uint16_t)));
	HANDLE_ERROR(cudaMalloc( (void **)&mask_device, width*height*sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void **)&result_device, width*height*sizeof(float)));


	HANDLE_ERROR(cudaPeekAtLastError());
	std::cout << "done alloc" << std::endl;
}
dark_subtraction_filter::~dark_subtraction_filter()
{
	HANDLE_ERROR(cudaSetDevice(4));

	HANDLE_ERROR(cudaStreamDestroy(dsf_stream));
	HANDLE_ERROR(cudaFree(picture_device));
	HANDLE_ERROR(cudaFree(mask_device));
	HANDLE_ERROR(cudaFree(result_device));
	HANDLE_ERROR(cudaFreeHost(pic_in_host));
	HANDLE_ERROR(cudaFreeHost(pic_out_host));

}
/*
u_char * apply_dark_subtraction_filter(uint16_t * picture_in,int width, int height)
{
	int pic_size = width*height*BYTES_PER_PIXEL;

	//u_char * picture_in; //Device (GPU) copy of picture in.
	u_char * picture_out = (u_char * )malloc(pic_size); //Create buffer for CPU memory output

	u_char * picture_device;
	u_char * dark_mask_device;

	cudaMalloc( (void **)&picture_device, pic_size);
	cudaMalloc( (void **)&dark_mask_device, pic_size);


	cudaMemcpy(picture_device, picture_in, pic_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dark_mask_device, dark_mask, pic_size, cudaMemcpyHostToDevice);

	//dim3 blockDims(block_length,block_length,1);
	dim3 blockDims(512,1,1);
	dim3 gridDims(ceil((float)width*height/blockDims.x),1,1);


	pixel_dark_subtraction_filter<<<gridDims,blockDims>>>(picture_device, dark_mask_device, width, height);
	cudaMemcpy(picture_out,picture_device,pic_size,cudaMemcpyDeviceToHost);

	cudaFree(picture_device);
	return picture_out;
}
 */

