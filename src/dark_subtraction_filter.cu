#include "dark_subtraction_filter.cuh"

//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include "cuda_utils.cuh"
#include <stdlib.h>
#include <iostream>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
//Kernel code, this runs on the GPU (device)
__global__ void apply_mask(uint16_t * pic_d, float * mask_d, float * result_d, uint16_t width, uint16_t height)
{
	uint32_t col = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t row = blockIdx.y*blockDim.y + threadIdx.y;
	uint32_t offset = col + row*width;
	//result_d[offset] = (float)(pic_d[offset]);// - mask_d[offset];
	result_d[offset] = (float)(pic_d[offset]) - mask_d[offset];


}
__global__ void sum_mask(uint16_t * pic_d, float * mask_d,uint16_t width, uint16_t height)
{
	uint32_t col = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t row = blockIdx.y*blockDim.y + threadIdx.y;
	uint32_t offset = col + row*width;

	mask_d[offset] = mask_d[offset] + (float)(pic_d[offset]);

}
__global__ void avg_mask(float * mask_d, float num_samples, uint16_t width, uint16_t height)
{
	uint32_t col = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t row = blockIdx.y*blockDim.y + threadIdx.y;
	uint32_t offset = col + row*width;
	//mask_d[offset] = __fdividef(mask_d[offset], num_samples);
	mask_d[offset] = mask_d[offset]/ num_samples;

}
__global__ void floatMemset(float * mask_d, float val, uint16_t width, uint16_t height)
{
	uint32_t col = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t row = blockIdx.y*blockDim.y + threadIdx.y;
	uint32_t offset = col + row*width;
	mask_d[offset] = val;
}
void dark_subtraction_filter::start_mask_collection()
{
	HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));
	mask_collected = false;
	averaged_samples = 0; //Synchronous
	HANDLE_ERROR(cudaPeekAtLastError());

	floatMemset<<< gridDims, blockDims,0,dsf_stream>>>(mask_device, 0.0f,width,height);

	//HANDLE_ERROR(cudaMemcpyAsync(pic_out_host,mask_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,dsf_stream));

	HANDLE_ERROR(cudaStreamSynchronize(dsf_stream));

}


void dark_subtraction_filter::finish_mask_collection()
{
	HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));

	avg_mask<<< gridDims, blockDims,0,dsf_stream>>>(mask_device, (float)averaged_samples,width,height);
	mask_collected = true;
	//HANDLE_ERROR(cudaMemcpyAsync(pic_out_host,mask_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,dsf_stream));

	HANDLE_ERROR(cudaStreamSynchronize(dsf_stream));

	std::cout << "mask collected: " << std::endl;
	//std::cout << "#samples: " << averaged_samples << std::endl;
	//std::cout << "maskvalue is: " << pic_out_host[9300] << std::endl;


}
void dark_subtraction_filter::update(uint16_t * pic_in)
{
	if(mask_collected)
	{
		update_dark_subtraction(pic_in);
	}
	else
	{
		update_mask_collection(pic_in);
	}
}

void dark_subtraction_filter::load_mask(float * mask_arr)
{
	HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));
	memcpy(mask_in_host,mask_arr,width*height*sizeof(float));
	HANDLE_ERROR(cudaMemcpyAsync(mask_device,mask_in_host,width*height*sizeof(float),cudaMemcpyHostToDevice,dsf_stream));
	HANDLE_ERROR(cudaStreamSynchronize(dsf_stream));

	mask_collected = true;
	std::cout << "mask loaded" << std::endl;
}
float * dark_subtraction_filter::get_mask()
{
	float *  mask_out = new float[width*height];

	HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));
	HANDLE_ERROR(cudaMemcpy(mask_out,mask_device,width*height*sizeof(float),cudaMemcpyDeviceToHost));
	std::cout << "mask in CPU mem" << std::endl;
	return mask_out;


}
void dark_subtraction_filter::update_dark_subtraction(uint16_t * pic_in)
{
	HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));

	//Synchronous
	memcpy(pic_in_host,pic_in,width*height*sizeof(uint16_t));

	//Asynchronous
	HANDLE_ERROR(cudaMemcpyAsync(picture_device,pic_in_host,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,dsf_stream));

	apply_mask<<< gridDims, blockDims,0,dsf_stream>>>(picture_device, mask_device, result_device, width,height);
	HANDLE_ERROR(cudaMemcpyAsync(pic_out_host,result_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,dsf_stream));
}

uint32_t dark_subtraction_filter::update_mask_collection(uint16_t * pic_in)
{
	// Synchronous
	HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));

	memcpy(pic_in_host,pic_in,width*height*sizeof(uint16_t));
	averaged_samples++;
	//HANDLE_ERROR(cudaStreamSynchronize(dsf_stream));
	//std::cout << "#samples: " << averaged_samples << std::endl;
	//std::cout << "pic_in_host: " << pic_in_host[9300] << "maskval: " << pic_out_host[9300] <<std::endl;
	//Asynchronous
	HANDLE_ERROR(cudaMemcpyAsync(picture_device,pic_in_host,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,dsf_stream));

	sum_mask<<< gridDims, blockDims,0,dsf_stream>>>(picture_device, mask_device,width,height);

	return averaged_samples;
}
float * dark_subtraction_filter::wait_dark_subtraction()
{

	//boost::shared_array < float > zeros(new float[width*height]);
	if(mask_collected)
	{
		HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));

		// Synchronous
		HANDLE_ERROR(cudaStreamSynchronize(dsf_stream));
		HANDLE_ERROR( cudaPeekAtLastError() );

		//return picture_out;
	}
	//return zeros;
	return pic_out_host;
}

dark_subtraction_filter::dark_subtraction_filter(int nWidth, int nHeight)
{


	HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));
	mask_collected = false;
	width=nWidth;
	height=nHeight;
	blockDims = dim3(BLOCK_SIDE,BLOCK_SIDE,1);
	gridDims = dim3(width/BLOCK_SIDE, height/BLOCK_SIDE,1);
	//picture_out = boost::shared_array<float>(new float[width*height]);
	HANDLE_ERROR(cudaStreamCreate(&dsf_stream));
	HANDLE_ERROR(cudaMallocHost( (void **)&pic_in_host,width*height*sizeof(uint16_t))); //cudaHostAllocPortable??

	HANDLE_ERROR(cudaMallocHost( (void **)&pic_out_host,width*height*sizeof(float))); //cudaHostAllocPortable??

	HANDLE_ERROR(cudaMallocHost( (void **)&mask_in_host,width*height*sizeof(float))); //cudaHostAllocPortable??

	HANDLE_ERROR(cudaMalloc( (void **)&picture_device, width*height*sizeof(uint16_t)));
	HANDLE_ERROR(cudaMalloc( (void **)&mask_device, width*height*sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void **)&result_device, width*height*sizeof(float)));

	HANDLE_ERROR(cudaPeekAtLastError());
	std::cout << "done alloc" << std::endl;
}
dark_subtraction_filter::~dark_subtraction_filter()
{
	mask_collected = false; //Do this to prevent reading after object has been killed
	HANDLE_ERROR(cudaSetDevice(DSF_DEVICE_NUM));

	HANDLE_ERROR(cudaStreamDestroy(dsf_stream));
	HANDLE_ERROR(cudaFree(picture_device));
	HANDLE_ERROR(cudaFree(mask_device));
	HANDLE_ERROR(cudaFree(result_device));
	HANDLE_ERROR(cudaFreeHost(pic_in_host));
	HANDLE_ERROR(cudaFreeHost(pic_out_host));
}


