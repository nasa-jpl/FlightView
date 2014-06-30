/*
 * takeobject.hpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#ifndef TAKEOBJECT_HPP_
#define TAKEOBJECT_HPP_

/* No longer using C++11 stuff, now using Boost, they are supposed to have the same syntax
#include <thread>
#include <atomic>
#include <mutex>
*/
//#include <boost/atomic.hpp> //Atomic isn't in the boost library till 1.5.4, debian wheezy has 1.4.9 :(
#include <stdint.h>
#include <stdio.h>
#include <ostream>
#include <string>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>


#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#include <boost/circular_buffer.hpp>
#include "std_dev_filter.cuh"
#include "chroma_translate_filter.cuh"
#include "dark_subtraction_filter.cuh"
#include "edtinc.h"

static const bool CHECK_FOR_MISSED_FRAMES_6604A = true;
class take_object {
	PdvDev * pdv_p;
	boost::thread pdv_thread;
	boost::thread save_thread;
	bool pdv_thread_run;

	bool newFrameAvailable;
	uint16_t * raw_data_ptr;
	uint16_t * image_data_ptr;
	boost::shared_array<float> std_dev_data;
	boost::shared_array<float> dark_subtraction_data;
	boost::shared_array <uint32_t> std_dev_histogram_data;


	unsigned int size;

	unsigned int channel;
	unsigned int numbufs;
	unsigned int frame_history_size;
	unsigned int filter_refresh_rate;

	int std_dev_filter_N;
	int lastfc;
	uint64_t count;
	chroma_translate_filter ctf;
	dark_subtraction_filter * dsf;
	std_dev_filter * sdvf;

	bool do_raw_save;
	bool dsf_save_available;
	bool std_dev_save_available;

	uint16_t * raw_save_ptr;
	boost::shared_array < float > dsf_save_ptr;
	boost::shared_array < float > std_dev_save_ptr;
	FILE * raw_save_file;
	FILE * dsf_save_file;
	FILE * std_dev_save_file;


public:
	take_object(int channel_num = 0, int number_of_buffers = 64, int fmsize = 1000, int filter_refresh_rate = 10);
	virtual ~take_object();
	void start();
	uint16_t * getImagePtr();
	uint16_t * getRawPtr();
	uint16_t * waitRawPtr();

	boost::shared_mutex data_mutex;

	unsigned int dataHeight;
	unsigned int frHeight;
	unsigned int frWidth;
	bool isChroma;
	boost::shared_array<float> getStdDevData();

	boost::shared_array<float> getDarkSubtractedData();
	boost::shared_array<uint32_t>getHistogramData();
	std::vector<float> * getHistogramBins();
	bool std_dev_ready();

	void startCapturingDSFMask();
	void finishCapturingDSFMask();
	void loadDSFMask(const char * file_name);


	void startSavingRaws(const char * );
	void stopSavingRaws();

	void startSavingDSFs(const char * );
	void stopSavingDSFs();

	void startSavingSTD_DEVs(const char * );
	void stopSavingSTD_DEVs();


	void setStdDev_N(int s);
private:
	void pdv_init();
	void savingLoop();


};

#endif /* TAKEOBJECT_HPP_ */
