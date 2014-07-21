/*
 * frame_c.hpp
 *
 *  Created on: Jul 15, 2014
 *      Author: nlevy
 */
//#include <atomic>
#include "constants.h"
#include "boost/thread.hpp"
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
	float fftMagnitude[FFT_INPUT_LENGTH/2];

private:
	/* Cuda will not compile with the atomics */
	//std::atomic_int_least8_t delete_counter; //NOTE!! It is incredibly critical that this number is equal to the number of frameview_widgets that are drawing + the number of mean profiles, too many and memory will leak; too few and invalid data will be displayed.
	//std::atomic_int_least8_t async_filtering_done;
	//std::atomic_int_least8_t valid_std_dev;
	int8_t delete_counter;
	int8_t async_filtering_done;
	int8_t valid_std_dev;

	boost::shared_mutex dc_mux;
	boost::shared_mutex af_mux;
	boost::shared_mutex vsd_mux;
public:
	frame_c();
	int8_t get_delete_counter();
	void set_delete_counter(int8_t val);
	int8_t get_async_filtering_done();
	void set_async_filtering_done(int8_t val);
	int8_t get_valid_std_dev();
	void set_valid_std_dev(int8_t val);




};


#endif /* FRAME_C_HPP_ */
