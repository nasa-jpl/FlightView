#include <stdint.h>
#include <ccomplex>
#ifndef HORIZONTALMEANFILTER_CUH_
#define HORIZONTALMEANFILTER_CUH_
#include <boost/thread.hpp>
#include "fft.hpp"
const static unsigned int MEAN_BUFFER_LENGTH = 256; //must be power of 2; will fail silently otherwise

class mean_filter {
public:
	mean_filter(int nWidth, int nHeight);

	virtual ~mean_filter();

	void start_mean(uint16_t * pic_in, float * vert_out, float * horiz_out, float * fft_out);

	void calculate_means();
	void wait_mean();
	fft myFFT;

private:
	boost::thread mean_thread;
	int width;
	int height;

	uint16_t * picture_in;
	float * vert;
	float * horiz;
	float  * fft_real_result;
	float frame_mean;
	float * mean_ring_buffer;
	unsigned int mean_ring_buffer_head;
	unsigned long frame_count;

};

#endif /* HORIZONTALMEANFILTER_CUH_ */
