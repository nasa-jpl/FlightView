#include "mean_filter.hpp"
#include "fft.hpp"

mean_filter::mean_filter(int nWidth, int nHeight)
{
	width = nWidth;
	height = nHeight;

	mean_ring_buffer = new float[MEAN_BUFFER_LENGTH];
	mean_ring_buffer_head = 0;
	frame_count = 0;
}

mean_filter::~mean_filter()
{

}
void mean_filter::start_mean(uint16_t * pic_in, float * vert_out, float * horiz_out, float * fft_out)
{
	vert = vert_out;
	horiz = horiz_out;
	fft_real_result = fft_out;
	picture_in = pic_in;
	mean_thread = boost::thread(&mean_filter::calculate_means, this);

}
void mean_filter::calculate_means()
{
	for(int r = 0; r < height; r++)
	{
		vert[r]=0;
	}
	for(int c = 0; c < width; c++)
	{
		horiz[c]=0;
	}
	for(int r = 0; r < height; r++)
	{
		for(int c = 0; c < width; c++)
		{
			vert[r] += picture_in[r*width + c];
			horiz[c] += picture_in[r*width + c];
		}
	}

	for(int r = 0; r < height; r++)
	{
		vert[r]/=height;
	}
	frame_mean = 0;
	for(int c = 0; c < width; c++)
	{
		horiz[c]/=width;
		frame_mean += horiz[c];
	}
	frame_mean/=width;

	mean_ring_buffer[mean_ring_buffer_head] = frame_mean;
	if(frame_count > MEAN_BUFFER_LENGTH)
	{
		fft_real_result = myFFT.doRealFFT(mean_ring_buffer,MEAN_BUFFER_LENGTH, mean_ring_buffer_head);
	}
	if(++mean_ring_buffer_head >= MEAN_BUFFER_LENGTH)
	{
		mean_ring_buffer_head = 0;
	}
	frame_count++;
}

void mean_filter::wait_mean()
{
	mean_thread.join();
}
