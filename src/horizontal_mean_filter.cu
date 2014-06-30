#include "horizontal_mean_filter.cuh"
#include "cuda_utils.cuh"
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))


/*
__global__ void horizontal_mean_kernel(uint16_t * pic_d, float * horiz_result, float * vert_result, int width, int height)
{
	//uint32_t col = blockIdx.x*blockDim.x + threadIdx.x;
	//uint32_t row = blockIdx.y*blockDim.y + threadIdx.y;
	//uint32_t offset = col + row*width;
	//uint32_t row = threadIdx.y;
	int large_side = width > height ? width : height;
	//ASSERT: Large side is always width
	float sum = 0;


	for(int i = 0; i < large_side; i++)
	{
		//Height interpretation


		//Width interpretation
	}
	//Eww, strided memory accesses
	if(blockIdx.x * grimDim.x + threadIdx.x > large_side)
	{

	}
	for(int w = 0; w < width; w++)
	{
		sum += pic_d[threadIdx.y*width + w];
	}
	result[threadIdx.y] = (sum/width);



}
horizontal_mean_filter::horizontal_mean_filter(int nWidth, int nHeight)
{
	HANDLE_ERROR(cudaSetDevice(MF_DEVICE_NUM));

	width = nWidth;
	height = nHeight;

	//blockDims = dim3(BLOCK_SIDE,BLOCK_SIDE,1);
	//gridDims = dim3(width/BLOCK_SIDE,height/BLOCK_SIDE,1);
	//blockDims = dim3(1,height/FRAME_DIVISION_FACTOR,1);
	//gridDims =  dim3(width,FRAME_DIVISION_FACTOR,1);
	int large_side = width > height ? width : height;
	gridDims = dim3(((large_side)/THREADS_PER_BLOCK)+1,1,1);
	blockDims = dim3(THREADS_PER_BLOCK,1,1);
	HANDLE_ERROR(cudaStreamCreate(&horizontal_stream));

	HANDLE_ERROR(cudaMalloc((void**)&picture_device,width*height*sizeof(uint16_t)));
	HANDLE_ERROR(cudaMalloc((void**)&result_device,height*sizeof(uint16_t)));


	HANDLE_ERROR(cudaMallocHost((void**)&pic_in_host,width*height*sizeof(uint16_t)));
	HANDLE_ERROR(cudaMallocHost((void**)&result_out_host,height*sizeof(uint16_t)));

	HANDLE_ERROR(cudaPeekAtLastError());

}



horizontal_mean_filter::~horizontal_mean_filter()
{
	HANDLE_ERROR(cudaSetDevice(MF_DEVICE_NUM));
	HANDLE_ERROR(cudaFree(picture_device));
	HANDLE_ERROR(cudaFree(result_device));


	HANDLE_ERROR(cudaFreeHost(pic_in_host));
	HANDLE_ERROR(cudaFreeHost(result_out_host));

	HANDLE_ERROR(cudaStreamDestroy(horizontal_stream));
}
void horizontal_mean_filter::start_horizontal_mean(uint16_t * pic_in)
{
	memcpy(pic_in_host,pic_in,width*height*sizeof(uint16_t));
	//Asynchronous
	HANDLE_ERROR(cudaMemcpyAsync(picture_device,pic_in_host,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,horizontal_stream));

	apply_mask<<< gridDims, blockDims,0,dsf_stream>>>(picture_device, mask_device, result_device, width,height);
	HANDLE_ERROR(cudaMemcpyAsync(pic_out_host,result_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,dsf_stream));
}
float * horizontal_mean_filter::wait_horizontal_mean()
{
	return NULL;
}
