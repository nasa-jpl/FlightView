#include "dark_subtraction_filter.hpp"

//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
//Kernel code, this runs on the GPU (device)

// define VERBOSE

void dark_subtraction_filter::start_mask_collection()
{
    /*! \brief Initializes the mask array to 0 and sends a signal to begin collecting image data */
    if(mean_inProgress) {
        return;
    }
    mask_collected = false;
    averaged_samples = 0;
	for(unsigned int i = 0; i < width*height; i++)
	{
        //mask[i]=0; // This gets initialized with the constructor,
                     // and from here on, will contain the "last" mask.
        mask_accum[i] = 0;
	}
}

void dark_subtraction_filter::finish_mask_collection()
{
    /*! \brief Averages each pixel value in the mask and sends a signal to begin dark subtracting images. */
    // Average each accumulated pixel value, and copy this to the mask.
#ifdef VERBOSE
    std::cout << "mask averaging starting, samples: " << averaged_samples << std::endl;
#endif
    if(mean_inProgress) {
        std::cerr << "ERROR, mask averaging is already in progress! Mutex fail!" << std::endl;
        return;
    }
    pthread_setname_np(pthread_self(), "MASKMEAN");

    mean_inProgress = true;
	for(unsigned int i = 0; i < width*height; i++)
	{
        // for debugging only, add delay here to simulate additional load
        // and make it possible to catch the thread:
        // usleep(10);
        mask[i] = mask_accum[i] / averaged_samples;
        
	}
    mean_inProgress = false;
	mask_collected = true;
#ifdef VERBOSE
    std::cout << "mask averaging completed, samples: " << averaged_samples << std::endl;
#endif
}
void dark_subtraction_filter::update(uint16_t * pic_in, float * pic_out)
{
    /*! \brief A loop which determines the behavior of this filter for incoming images.
     * \param pic_in The incoming frame from the device
     * \param pic_out The dark subtracted image
     * update_mask_collection(uint16_t* pic_in) must be serialized to avoid errors in the mask data.
     */
	if(mask_collected)
	{
        // A dark mask has already been collected
        // and we are not collecting one at the moment.
        // So, just use the mask.
        update_dark_subtraction(pic_in, pic_out); // use the mask. pic_out = pic_in - mask;
	}
	else
	{
		mask_mutex.lock();
        update_mask_collection(pic_in); // accumulate on the mask_accum (mask_accum += pic_in)
        update_dark_subtraction(pic_in, pic_out); // use the prior mask for now (mask)
		mask_mutex.unlock();
	}
}
void dark_subtraction_filter::load_mask(float* mask_arr)
{
    /*! \brief Copy the memory for a mask into the internal mask array.
     * \param mask_arr The mask to load into the filter as float array
     */
    mask_mutex.lock();
	memcpy(mask,mask_arr,width*height*sizeof(float));
	mask_collected = true;
    mask_mutex.unlock();
#ifdef VERBOSE
	std::cout << "mask loaded" << std::endl;
#endif
}
float* dark_subtraction_filter::get_mask()
{
    /*! \brief Returns the currently loaded mask in this instance of the filter. */
	return mask;
}
void dark_subtraction_filter::update_dark_subtraction(uint16_t* pic_in, float* pic_out)
{
    /*! \brief Subtracts the dark mask from the image data for each pixel.
     * \param pic_in Raw data that contains two bytes per pixel.
     */
	for(unsigned int i = 0; i < width*height; i++)
	{
		pic_out[i] = pic_in[i] - mask[i];
	}
}
void dark_subtraction_filter::static_dark_subtract(unsigned int* pic_in, float* pic_out)
{
    /*! \brief Subtracts the dark mask from the image data for each pixel in a discrete image.
     * \param pic_in An image in an unsigned int format, 4 bytes per pixel. */
    for(unsigned int i = 0; i < width*height; i++)
    {
        pic_out[i] = (float)pic_in[i] - mask[i];
    }
}
uint32_t dark_subtraction_filter::update_mask_collection(uint16_t* pic_in)
{
    /*! \brief Collect the current image.
     *
     * This section must be locked with the mask_collected variable to prevent serialization errors. */

    // Do not add more frames to the accum if
    // processing is happening at the moment.
    if(mean_inProgress)
        return averaged_samples;

    if(!mask_collected)
    {
        for(unsigned int i = 0; i<width*height; i++)
        {
            //mask[i] = pic_in[i] + mask[i];
            mask_accum[i] = pic_in[i] + mask_accum[i];
        }
        averaged_samples++;
    }
    return averaged_samples;
}

dark_subtraction_filter::dark_subtraction_filter(int nWidth, int nHeight)
{
    /*! \brief Initializes the filter for a specified frame geometry.
     * \param nWidth The new frame width
     * \param nHeight The new frame height
     */
    mask_collected = false;
    width = nWidth;
    height = nHeight;
    for(unsigned int i = 0; i < width*height; i++)
    {
        mask[i]=0;
    }
}
dark_subtraction_filter::~dark_subtraction_filter()
{
    /*! When deallocating the filter, dark subtraction must be turned off to avoid
     * bad memory access. */
	mask_collected = false; //Do this to prevent reading after object has been killed
}
