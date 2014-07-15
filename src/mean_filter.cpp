#include "mean_filter.hpp"
#include "fft.hpp"
#include <atomic>

mean_filter::mean_filter(frame_c * frame, unsigned long frame_count, int nWidth, int nHeight)
{
	width = nWidth;
	height = nHeight;
	this->frame = frame;
	this->frame_count = frame_count;
}

mean_filter::~mean_filter()
{

}
//void mean_filter::start_mean(uint16_t * pic_in, float * vert_out, float * horiz_out, float * fft_out)
void mean_filter::start_mean(frame_c * frame)

{
	mean_thread = boost::thread(&mean_filter::calculate_means, this);

}
void mean_filter::calculate_means()
{
	for(int r = 0; r < height; r++)
	{
		frame->vertical_mean_profile[r]=0;
	}
	for(int c = 0; c < width; c++)
	{
		frame->horizontal_mean_profile[c]=0;
	}
	for(int r = 0; r < height; r++)
	{
		for(int c = 0; c < width; c++)
		{
			frame->vertical_mean_profile[r] += frame->image_data_ptr[r*width + c];
			frame->horizontal_mean_profile[c] += frame->image_data_ptr[r*width + c];
		}
	}

	for(int r = 0; r < height; r++)
	{
		frame->vertical_mean_profile[r]/=height;
	}
	frame_mean = 0;
	for(int c = 0; c < width; c++)
	{
		frame->horizontal_mean_profile[c]/=width;
		frame_mean += frame->horizontal_mean_profile[c];
	}
	frame_mean/=width;

	mean_ring_buffer_fft_head = mean_ring_buffer_head;
	mean_ring_buffer[mean_ring_buffer_head++] = frame_mean;
	if(mean_ring_buffer_head >= MEAN_BUFFER_LENGTH)
	{
		mean_ring_buffer_head = 0;
	}
	if(frame_count > MEAN_BUFFER_LENGTH)
	{
		myFFT.doRealFFT(mean_ring_buffer,MEAN_BUFFER_LENGTH, mean_ring_buffer_fft_head, frame->fftMagnitude);
	}

	frame->async_filtering_done = 1;

}

void mean_filter::wait_mean()
{
	mean_thread.join();
}
