#include "mean_filter.hpp"
#include "fft.hpp"
#include <atomic>
#include <stdio.h>

mean_filter::mean_filter(frame_c * frame,unsigned long frame_count,int startCol,\
                         int endCol,int startRow,int endRow,int actualWidth, \
                         bool useDSF,FFT_t FFTtype,\
                         int lh_start, int lh_end,\
                         int cent_start, int cent_end,\
                         int rh_start, int rh_end)
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

    // copy the latest overlay parameters from the frame:
    // new with every frame... bad idea
    this->lh_start = lh_start;
    this->lh_end = lh_end;
    this->cent_start = cent_start;
    this->cent_end = cent_end;
    this->rh_start = rh_start;
    this->rh_end = rh_end;

}
void mean_filter::start_mean()
{
    mean_thread = boost::thread(&mean_filter::calculate_means, this);
}
void mean_filter::calculate_means()
{
    // bool is_overlay_plot = false;
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
    memset(tap_profile, 0, MAX_HEIGHT*TAP_WIDTH*sizeof(*tap_profile));
    memset(frame->vertical_mean_profile, 0, MAX_HEIGHT*sizeof(*(frame->vertical_mean_profile)));
    memset(frame->horizontal_mean_profile, 0, MAX_WIDTH*sizeof(*(frame->horizontal_mean_profile)));
    memset(frame->vertical_mean_profile_lh, 0, MAX_HEIGHT*sizeof(*(frame->vertical_mean_profile_lh)));
    memset(frame->vertical_mean_profile_rh, 0, MAX_HEIGHT*sizeof(*(frame->vertical_mean_profile_rh)));

    // Remove this later, debug code:
    if(!(frame_count % 100))
    {
        usleep(10000);
        std::cout << "----- from CUDA take object: -----\n";
        std::cout << "frame_count:   " << frame_count << std::endl;
        if(!lh_start)
            std::cout << "-----========= WARNING, lh_start == 0 WARNING WARNING WARNING =========-----" << std::endl;
        // The above should only occur at the start of liveview, before any useful parameters are passed in.
        std::cout << "lh_start:   " << lh_start <<   ", lh_end:   " << lh_end << std::endl;
        std::cout << "rh_start:   " << rh_start <<   ", rh_end:   " << rh_end << std::endl;
        std::cout << "cent_start: " << cent_start << ", cent_end: " << cent_end << std::endl;
        std::cout << "----- end from CUDA take object --\n";
    }

    if( (cent_start != 0) && (cent_end !=0) )
    {
        // is_overlay_plot = true;
        beginCol = cent_start;
        width = cent_end;
        horizDiff = cent_end-cent_start + 1;
        if(!(frame_count % 100))
        {
            std::cout << "beginCol: " << beginCol << "width: " << width << std::endl;
            std::cout << "beginRow: " << beginRow << "height: " << height << std::endl;
        }
    }

    for(int r = beginRow; r < height; r++)
    {
        for(int c = beginCol; c < width; c++)
        {
            if(!useDSF)
            {
                frame->vertical_mean_profile[r] += frame->image_data_ptr[r*frWidth + c];
                frame->horizontal_mean_profile[c] += frame->image_data_ptr[r*frWidth + c];
                if(FFTtype == TAP_PROFIL)
                    tap_profile[r * TAP_WIDTH + c % TAP_WIDTH] = frame->image_data_ptr[r * frWidth + c];
            }
            else if(useDSF)
            {
                frame->vertical_mean_profile[r] += frame->dark_subtracted_data[r*frWidth + c];
                frame->horizontal_mean_profile[c] += frame->dark_subtracted_data[r*frWidth + c];
                if(FFTtype == TAP_PROFIL)
                    tap_profile[r * TAP_WIDTH + c % TAP_WIDTH] = frame->dark_subtracted_data[r*frWidth + c];
            }
        }
    }


    // LH and RH profiles:
    if(!useDSF)
    {
        for(int r = 0; r < height; r++)
        {
            // for each row, grab the data at UI-selected col=start and col=end
            for(int c = lh_start; c < lh_end; c++)
            {
                frame->vertical_mean_profile_lh[r] += frame->image_data_ptr[r*frWidth + c];
            }
            for(int c = rh_start; c < rh_end; c++)
            {
                frame->vertical_mean_profile_rh[r] += frame->image_data_ptr[r*frWidth + c];
            }
        }
    } else {
        for(int r = 0; r < height; r++)
        {
            for(int c = lh_start; c < lh_end; c++)
            {
                frame->vertical_mean_profile_lh[r] += frame->dark_subtracted_data[r*frWidth + c];
            }
            for(int c = rh_start; c < rh_end; c++)
            {
                frame->vertical_mean_profile_rh[r] += frame->dark_subtracted_data[r*frWidth + c];
            }
        }
    }

    for(int r = beginRow; r < height; r++)
    {
        frame->vertical_mean_profile[r] /= horizDiff;
        frame->vertical_mean_profile_lh[r] /= (lh_end-lh_start+1);
        frame->vertical_mean_profile_rh[r] /= (rh_end-rh_start+1);
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
        mean_ring_buffer_head = 0;
    if(frame_count > FFT_INPUT_LENGTH && FFTtype == PLANE_MEAN)
		myFFT.doRealFFT(mean_ring_buffer, mean_ring_buffer_fft_head, frame->fftMagnitude);
    else if( FFTtype == VERT_CROSS )
        myFFT.doRealFFT(frame->vertical_mean_profile, 0, frame->fftMagnitude); // FOR THE VERTICAL CROSSHAIR FFT
    else if( FFTtype == TAP_PROFIL )
        myFFT.doRealFFT(tap_profile, 0, frame->fftMagnitude);

    frame->async_filtering_done = 1;
    delete this; //I can honestly say this is the ugliest line of C++ I've ever written.
}
void mean_filter::wait_mean()
{
	mean_thread.join();
}
