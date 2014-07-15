/*
 * frame_c.hpp
 *
 *  Created on: Jul 15, 2014
 *      Author: nlevy
 */
#include <atomic>
#include "constants.h"
#ifndef FRAME_C_HPP_
#define FRAME_C_HPP_


struct frame_c{
	uint16_t raw_data_ptr[MAX_SIZE];
	uint16_t * image_data_ptr;
	float dark_subtracted_data[MAX_SIZE];
	float std_dev_data[MAX_SIZE];
	uint32_t std_dev_histogram[NUMBER_OF_BINS];
	float vertical_mean_profile[MAX_HEIGHT];
	float horizontal_mean_profile[MAX_WIDTH];
	float fftMagnitude[MEAN_BUFFER_LENGTH/2];
	std::atomic_int_least8_t delete_counter; //NOTE!! It is incredibly critical that this number is equal to the number of frameview_widgets that are drawing + the number of mean profiles, too many and memory will leak; too few and invalid data will be displayed.
	std::atomic_int_least8_t async_filtering_done;

	frame_c() {
		delete_counter = 5;
		async_filtering_done = 0;
	};
};


#endif /* FRAME_C_HPP_ */
