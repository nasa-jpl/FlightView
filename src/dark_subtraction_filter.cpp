#include "dark_subtraction_filter.hpp"

//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
//Kernel code, this runs on the GPU (device)

/* #define VERBOSE */

void dark_subtraction_filter::start_mask_collection()
{
    /*! \brief Initializes the mask array to 0 and sends a signal to begin collecting image data */
	mask_collected = false;
    averaged_samples = 0;
	for(unsigned int i = 0; i < width*height; i++)
	{
		mask[i]=0;
	}
}
void dark_subtraction_filter::finish_mask_collection()
{
    /*! \brief Averages each pixel value in the mask and sends a signal to begin dark subtracting images. */
	for(unsigned int i = 0; i < width*height; i++)
	{
        mask[i] /= averaged_samples;
	}
	mask_collected = true;
#ifdef VERBOSE
	std::cout << "mask collected: " << std::endl;
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
		update_dark_subtraction(pic_in, pic_out);
	}
	else
	{
		mask_mutex.lock();
		update_mask_collection(pic_in);
		mask_mutex.unlock();
	}
}
void dark_subtraction_filter::load_mask(float* mask_arr)
{
    /*! \brief Copy the memory for a mask into the internal mask array.
     * \param mask_arr The mask to load into the filter as float array
     */
	memcpy(mask,mask_arr,width*height*sizeof(float));
	mask_collected = true;
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
	if(!mask_collected)
	{
    	for(unsigned int i = 0; i<width*height; i++)
		{
        	mask[i] = pic_in[i] + mask[i];
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
