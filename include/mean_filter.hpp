#include <cstdint>
#include <ccomplex>
#include <boost/thread.hpp>
#include <atomic>
#include "frame_c.hpp"
#include "fft.hpp"
#ifndef HORIZONTALMEANFILTER_CUH_
#define HORIZONTALMEANFILTER_CUH_

static float mean_ring_buffer[FFT_MEAN_BUFFER_LENGTH];
static std::atomic_uint_least16_t mean_ring_buffer_head;
class mean_filter {
public:
	mean_filter(frame_c * frame, unsigned long frame_count,int nWidth, int nHeight, bool useDSF);

	virtual ~mean_filter();

	//void start_mean(uint16_t * pic_in, float * vert_out, float * horiz_out, float * fft_out);
	void start_mean();

	void calculate_means();
	void wait_mean();
	fft myFFT;

private:
	boost::thread mean_thread;
	int width;
	int height;
	bool useDSF;
	float frame_mean;
	unsigned int mean_ring_buffer_fft_head;
	unsigned long frame_count;
	frame_c * frame;
};

#endif /* HORIZONTALMEANFILTER_CUH_ */
