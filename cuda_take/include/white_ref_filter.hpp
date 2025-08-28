/*
 * dark_subtraction_filter.cuh
 *
 *  Created on: May 22, 2014
 *      Author: nlevy
 */

#ifndef WHITE_REF_FILTER_CUH_
#define WHITE_REF_FILTER_CUH_

#include <stdint.h>
#include <unistd.h>
#include <mutex>
#include <chrono>

#include "edtinc.h"
#include "constants.h"

#include "dark_subtraction_filter.hpp"

/*! \file
 * \brief Applies Dark Subtraction Masks to images
 * \paragraph
 *
 * The filter checks whether or not to use one of two behaviors each frame. It may select to record a frame by adding it to the mask,
 * or to apply the dark subtraction filter to the current frame. When requested to record dark frames, the mask will be initialized to
 * zero and the loop will collect frames. It is important to serialize access to the mask while recording because if collection is stopped
 * in the middle of a frame then there will be errors in the mask. Finally, when recording is stopped, each pixel value is averaged.
 * \paragraph
 *
 * The subtraction itself is very intuitive. As the mask array is exactly one frame large, each pixel value in the arriving image is subtacted
 * by the value in the mask to get the output dark subtracted image. For live images, this is acheived with
 * update_dark_subtraction(uint16_t* pic_in, float* pic_out) and with static_dark_subtract(unsigned int* pic_in, float* pic_out) for discrete
 * images. Although the distinction is arbitrary, the functions are split based on differences in the frontend.
 */

class white_ref_filter
{
public:
    white_ref_filter() {} // Useless default constructor
    white_ref_filter(int nWidth, int nHeight, bool *extMaskReadyFlag, dark_subtraction_filter *dsf);
    virtual ~white_ref_filter();
    void update_wr(uint16_t* pic_in, float* pic_out); // RAW data as input
    void update_wr(float* pic_in, float* pic_out); // (already) dark subtracted data as input
    void update_wr(unsigned int* pic_in, float* pic_out);
	float * wait_dark_subtraction();
	void start_mask_collection();
    uint32_t update_mask_collection(uint16_t *pic_in);
    uint32_t update_mask_collection(float *pic_in);
    //void update(uint16_t * pic_in, float * pic_out);
    //void update(float * pic_in, float * pic_out);
    void updateTaking(float * pic_in, float * pic_out);
    void updateFrame(float * pic_in, float * pic_out);
    void finish_mask_collection();
	void load_mask(float * mask_arr);
	float * get_mask();

    std::mutex mask_mutex;
    std::timed_mutex tmutex;

private:
    void testPatternMe(uint16_t* array);
    void testPatternMe(float* array);
    void testForZeros(float* array);
	bool mask_collected;
    bool mean_inProgress = false;
    bool *extMaskReady = NULL;
    dark_subtraction_filter *dsf = NULL;
	//boost::shared_array<float> picture_out;
	unsigned int width;
	unsigned int height;
	unsigned int averaged_samples;

    double mask_accum[MAX_SIZE];
    float *mask = NULL;
    float *darkMask = NULL;
    bool haveDarkMask = false;

};

#endif /* WHITE_REF_FILTER_CUH_ */
