#include "white_ref_filter.hpp"

//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
//Kernel code, this runs on the GPU (device)

// define VERBOSE

void white_ref_filter::start_mask_collection()
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

void white_ref_filter::finish_mask_collection()
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
    if(extMaskReady)
        *extMaskReady = false;
    pthread_setname_np(pthread_self(), "WR_MEAN");

    if(!dsf->maskReady()) {
        std::cerr << "Warning, white reference average cannot be completed without a dark mask." << std::endl;
        return;
    }

    mean_inProgress = true;
    // Take the mean
    for(unsigned int i = 0; i < width*height; i++)
    {
        // for debugging only, add delay here to simulate additional load
        // and make it possible to catch the thread:
        // usleep(10);
        mask[i] = mask_accum[i] / averaged_samples;
        
    }
    // Do the DSF:
    // EHL: Change, the data coming in are already DSF'd
    //dsf->static_dark_subtract(mask, mask); // output and input are the same memory

    mean_inProgress = false;
    mask_collected = true;
    if(extMaskReady)
        *extMaskReady = true;
#ifdef VERBOSE
    std::cout << "mask averaging completed, samples: " << averaged_samples << std::endl;
#endif
}
//void white_ref_filter::update(uint16_t * pic_in, float * pic_out)
//{
//    /*! \brief A loop which determines the behavior of this filter for incoming images.
//     * \param pic_in The incoming frame from the device
//     * \param pic_out The dark subtracted image
//     * update_mask_collection(uint16_t* pic_in) must be serialized to avoid errors in the mask data.
//     */
//	if(mask_collected)
//	{
//        // A dark mask has already been collected
//        // and we are not collecting one at the moment.
//        // So, just use the mask.
//        update_wr(pic_in, pic_out); // use the mask. pic_out = pic_in - mask;
//	}
//	else
//	{
//		mask_mutex.lock();
//        update_mask_collection(pic_in); // accumulate on the mask_accum (mask_accum += pic_in)
//        update_wr(pic_in, pic_out); // use the prior mask for now (mask)
//		mask_mutex.unlock();
//	}
//}

//void white_ref_filter::update(float * pic_in, float * pic_out)
//{
//    /*! \brief A loop which determines the behavior of this filter for incoming images.
//     * \param pic_in The incoming frame from the device
//     * \param pic_out The dark subtracted image
//     * update_mask_collection(uint16_t* pic_in) must be serialized to avoid errors in the mask data.
//     */
//    if(mask_collected)
//    {
//        // A dark mask has already been collected
//        // and we are not collecting one at the moment.
//        // So, just use the mask.
//        update_wr(pic_in, pic_out); // use the mask. pic_out = pic_in - mask;
//    }
//    else
//    {
//        mask_mutex.lock();
//        update_mask_collection(pic_in); // accumulate on the mask_accum (mask_accum += pic_in)
//        update_wr(pic_in, pic_out); // use the prior mask for now (mask)
//        mask_mutex.unlock();
//    }
//}

void white_ref_filter::updateTaking(float * pic_in, float * pic_out)
{
    // We're collecting WR right now, so update the mask accumulator
    // and process the given pic_out while we're at it with whatever prior WR is available.

    auto timeout = std::chrono::milliseconds(250);


    if(!mask_collected) {
        std::unique_lock<std::timed_mutex> lock (tmutex, std::defer_lock);
        //mask_mutex.lock();
        if(lock.try_lock_for(timeout)) {
            update_mask_collection(pic_in); // accumulate on the mask_accum (mask_accum += pic_in)
            update_wr(pic_in, pic_out); // use the prior mask for now (mask)
        } else {
            std::cerr << "ERROR: Mutex lock condition reached unexpectedly." << std::endl << std::flush;
            // do something to reset the state of things... but what?
            mask_collected = true; // mark it as done or else we'll probably never get there.
        }
        //mask_mutex.unlock();
    }
}

void white_ref_filter::updateFrame(float * pic_in, float * pic_out)
{
    // We already have a mask.
    // Take the incoming new frame, and update the pic_out frame
    if(mask_collected)
    {
        // A dark mask has already been collected
        // and we are not collecting one at the moment.
        // So, just use the mask.
        update_wr(pic_in, pic_out); // use the mask. pic_out = pic_in - mask;
    }
}

void white_ref_filter::load_mask(float* mask_arr)
{
    /*! \brief Copy the memory for a mask into the internal mask array.
     * \param mask_arr The mask to load into the filter as float array
     */
    if( (!mask_arr) || (!mask) ){
        abort();
    }
    mask_mutex.lock();
//    if(dsf && dsf->maskReady()) {
//        dsf->static_dark_subtract(mask_arr, mask);
//    } else {
        memcpy(mask,mask_arr,width*height*sizeof(float));
    //}

    std::cerr << "Loaded white reference mask into memory." << std::endl << std::flush;
    //testPatternMe(mask);
    //std::cerr << "Changed loaded mask into test pattern." << std::endl << std::flush;

    mask_collected = true;
    mask_mutex.unlock();
#ifdef VERBOSE
	std::cout << "mask loaded" << std::endl;
#endif
}
float* white_ref_filter::get_mask()
{
    /*! \brief Returns the currently loaded mask in this instance of the filter. */
	return mask;
}
void white_ref_filter::update_wr(uint16_t* pic_in, float* pic_out)
{
    abort(); // don't call this function....
    /*! \brief Subtracts the dark mask from the image data for each pixel.
     * \param pic_in Raw data that contains two bytes per pixel.
     */
    // Let's be sensible here and skip that top metadata row.
    // Nothing good can come from dividing that row...
    if(mask)
        for(unsigned int i = width; i < (width*height); i++)
        //for(unsigned int i = 0; i < width*height; i++)
        {
            if (mask[i] == 0) {
                pic_out[i] = pic_in[i];
            } else {
                pic_out[i] = (float)pic_in[i] / mask[i];
            }
        }
}
void white_ref_filter::update_wr(float* pic_in, float* pic_out)
{
    /*! \brief Subtracts the dark mask from the image data for each pixel.
     * \param pic_in float data that contains dark subtracted data
     */
    // testPatternMe(pic_in);
    int zeroCounter = 0;
    float minValue = 2.0;
    int scalingValue = 1000;
    if(mask)
        for(unsigned int i = width; i < (width*height); i++)
        //for(unsigned int i = 0; i < width*height; i++)
        {
            if (mask[i] < minValue) {
                pic_out[i] = pic_in[i];
                zeroCounter++;
            } else {
                //pic_out[i] = pic_in[i]; // a good debug test
                pic_out[i] = scalingValue*pic_in[i] / mask[i];
            }
            //if(i%123==0)
            //    std::cerr << "mask[i] = " << mask[i] << ", pic_in[i] = " << pic_in[i] << ", pic_out[i] = " << pic_out[i] << std::endl;
        }
    // std::cerr << "Zero counter: " << zeroCounter << std::endl <<std::flush; // 41 dead pixels will show up here
    // testPatternMe(pic_out); // this works.
    // testForZeros(pic_out);
}

void white_ref_filter::update_wr(unsigned int* pic_in, float* pic_out)
{
    /*! \brief Subtracts the dark mask from the image data for each pixel in a discrete image.
     * \param pic_in An image in an unsigned int format, 4 bytes per pixel. */
    float minValue = 2.0;
    int scalingValue = 1000;

    if(mask)
        for(unsigned int i = width; i < (width*height); i++)
            //for(unsigned int i = 0; i < width*height; i++)
        {
            if (mask[i] < minValue) {
                pic_out[i] = pic_in[i];
            } else {
                pic_out[i] = scalingValue*(float)pic_in[i] / mask[i];
            }
        }

}
uint32_t white_ref_filter::update_mask_collection(uint16_t* pic_in)
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

uint32_t white_ref_filter::update_mask_collection(float* pic_in)
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

white_ref_filter::white_ref_filter(int nWidth, int nHeight,
                                   bool *extMaskReadyFlag, dark_subtraction_filter *dsf)
{
    /*! \brief Initializes the filter for a specified frame geometry.
     * \param nWidth The new frame width
     * \param nHeight The new frame height
     */
    mask_collected = false;
    width = nWidth;
    height = nHeight;
    if(dsf == nullptr)
        abort();

    this->dsf = dsf;

    this->mask = (float*)calloc(nWidth*nHeight, sizeof(float));

    if(mask==NULL)
        abort();

    // A test pattern:
    testPatternMe(mask);

    if(extMaskReadyFlag == NULL) {
        // Let's crash right here so we can catch this insanity if it occurs.
        abort();
    }
    this->extMaskReady = extMaskReadyFlag;
}

void white_ref_filter::testForZeros(float *array) {
    unsigned int zeroCounter = 0;
    unsigned int nearZeroCounter = 0;
    for(unsigned int i=0; i < height*width; i++) {
        if( array[i] == 0) {
            zeroCounter++;
        } else if (  abs(array[i]*100) < 1 ) {
            nearZeroCounter++;
        }
    }
    if(zeroCounter || nearZeroCounter) {
        std::cerr << "Zero count: " << zeroCounter << ", nearZero: " << nearZeroCounter << std::endl << std::flush;
    }
}

void white_ref_filter::testPatternMe(float *array) {
    for(unsigned int i=0; i < height*width; i++) {
        array[i] = i;
    }
}

void white_ref_filter::testPatternMe(uint16_t *array) {
    for(unsigned int i=0; i < height*width; i++) {
        array[i] = i;
    }
}

white_ref_filter::~white_ref_filter()
{
    /*! When deallocating the filter, dark subtraction must be turned off to avoid
     * bad memory access. */
	mask_collected = false; //Do this to prevent reading after object has been killed
}
