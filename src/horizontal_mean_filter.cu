#include "horizontal_mean_filter.cuh"

horizontal_mean_filter::horizontal_mean_filter(int nWidth, int nHeight)
{
	width = nWidth;
	height = nHeight;
	cudaSetDevice(HMF_DEVICE_NUM);
	HANDLE_ERROR(cudaMalloc((void**)&picture_device,width*height*sizeof(uint16_t)));
	HANDLE_ERROR(cudaMalloc((void**)&result_device,width*sizeof(uint16_t)));


	HANDLE_ERROR(cudaMallocHost((void**)&pic_in_host,width*height*sizeof(uint16_t)));
	HANDLE_ERROR(cudaMallocHost((void**)&result_out_host,width*sizeof(uint16_t)));

	cudaStreamCreate(&horizontal_stream);
}


horizontal_mean_filter::~horizontal_mean_filter()
{
	cudaSetDevice(HMF_DEVICE_NUM);
		HANDLE_ERROR(cudaFree(picture_device));
		HANDLE_ERROR(cudaFree(result_device));


		HANDLE_ERROR(cudaFreeHost(pic_in_host));
		HANDLE_ERROR(cudaFreeHost(result_out_host));

		cudaStreamDestroy(horizontal_stream);
}
horizontal_mean_filter::start_horizontal_mean(uint16_t * pic_in)
{

}
boost::shared_array < float > horizontal_mean_filter::wait_horizontal_mean()
{

}
