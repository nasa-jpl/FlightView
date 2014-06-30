#include <stdint.h>

#ifndef HORIZONTALMEANFILTER_CUH_
#define HORIZONTALMEANFILTER_CUH_
const static int MF_DEVICE_NUM = 1;
const static int THREADS_PER_BLOCK = 512;
class horizontal_mean_filter {
public:
	horizontal_mean_filter() {};
	horizontal_mean_filter(int nWidth, int nHeight);

	virtual ~horizontal_mean_filter();

	void start_horizontal_mean(uint16_t * pic_in);
	float * wait_horizontal_mean();

private:
	int width;
	int height;

	uint16_t * pic_in_host;
	uint16_t * picture_device;

	float * result_device;
	float * result_out_host;

	dim3 blockDims;
	dim3 gridDims;

	cudaStream_t horizontal_stream;

	//boost::shared_array<float> result_out;

};

#endif /* HORIZONTALMEANFILTER_CUH_ */
