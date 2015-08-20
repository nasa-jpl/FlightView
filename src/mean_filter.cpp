#include "mean_filter.hpp"
#include "fft.hpp"
#include <atomic>

mean_filter::mean_filter(frame_c * frame,unsigned long frame_count,int startCol,int endCol,int startRow,int endRow,int actualWidth, \
                         bool useDSF,FFT_t FFTtype)
{
    beginCol = startCol;
    width = endCol;
    beginRow = startRow;
    height = endRow;
    frWidth = actualWidth;

	this->frame = frame;
	this->frame_count = frame_count;
	this->useDSF = useDSF;
    this->FFTtype = FFTtype;
}
void mean_filter::start_mean()
{
	mean_thread = boost::thread(&mean_filter::calculate_means, this);
}
void mean_filter::calculate_means()
{
    int horizDiff = width - beginCol;
    int vertDiff = height - beginRow;
    if( horizDiff == 0 )
    {
        horizDiff = 1;
        width++;
    }
    if( vertDiff == 0 )
    {
        vertDiff = 1;
        height++;
    }
    for(int r = beginRow; r < height; r++)
    {
        frame->vertical_mean_profile[r]=0;
    }
    for(int c = beginCol; c < width; c++)
    {
        frame->horizontal_mean_profile[c]=0;
    }
    for(int r = beginRow; r < height; r++)
    {
        for(int c = beginCol; c < width; c++)
        {
            if(!useDSF)
            {
                frame->vertical_mean_profile[r] += frame->image_data_ptr[r*frWidth + c];
                frame->horizontal_mean_profile[c] += frame->image_data_ptr[r*frWidth + c];
                if( FFTtype == TAP_PROFIL )
                    tap_profile[r*TAP_WIDTH + c%TAP_WIDTH] = frame->image_data_ptr[r*frWidth + c];
            }
            else if(useDSF)
            {
                frame->vertical_mean_profile[r] += frame->dark_subtracted_data[r*frWidth + c];
                frame->horizontal_mean_profile[c] += frame->dark_subtracted_data[r*frWidth + c];
                if( FFTtype == TAP_PROFIL )
                    tap_profile[r*TAP_WIDTH + c%TAP_WIDTH] = frame->dark_subtracted_data[r*frWidth + c];
            }
        }
    }
    for(int r = beginRow; r < height; r++)
    {
        frame->vertical_mean_profile[r] /= horizDiff;
    }

    // begin determining frame mean for FFT
    frame_mean = 0;
    for(int c = beginCol; c < width; c++)
    {
        frame->horizontal_mean_profile[c] /= (vertDiff);
        frame_mean += frame->horizontal_mean_profile[c];
	}
    frame_mean /= frWidth;

    mean_ring_buffer_fft_head = mean_ring_buffer_head;

    mean_ring_buffer[mean_ring_buffer_head++] = frame_mean;
    if(mean_ring_buffer_head >= FFT_MEAN_BUFFER_LENGTH)
    {
        mean_ring_buffer_head = 0;
    }
    if(frame_count > FFT_INPUT_LENGTH && FFTtype == PLANE_MEAN)
	{
		myFFT.doRealFFT(mean_ring_buffer, mean_ring_buffer_fft_head, frame->fftMagnitude);
    }
    else if( FFTtype == VERT_CROSS )
    {
        myFFT.doRealFFT(frame->vertical_mean_profile, 0, frame->fftMagnitude); // FOR THE VERTICAL CROSSHAIR FFT
    }
    else if( FFTtype == TAP_PROFIL )
    {
        myFFT.doRealFFT(tap_profile, 0, frame->fftMagnitude);
    }

    frame->async_filtering_done = 1;
    delete this; //I can honestly say this is the ugliest line of C++ I've ever written.
}
void mean_filter::wait_mean()
{
	mean_thread.join();
}
