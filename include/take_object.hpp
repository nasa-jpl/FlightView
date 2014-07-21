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
#include <memory>
#include <atomic>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "std_dev_filter.hpp"
#include "chroma_translate_filter.cuh"
#include "dark_subtraction_filter.cuh"
#include "mean_filter.hpp"
#include "edtinc.h"
#include "camera_types.h"
#include "frame_c.hpp"


static const bool CHECK_FOR_MISSED_FRAMES_6604A = true;
const static int NUMBER_OF_FRAMES_TO_BUFFER = 100;





class take_object {
	PdvDev * pdv_p;
	boost::thread pdv_thread;

	unsigned int size;

	unsigned int channel;
	unsigned int numbufs;
	unsigned int filter_refresh_rate;

	int std_dev_filter_N;
	int lastfc;
	uint64_t count;
	chroma_translate_filter ctf;
	dark_subtraction_filter * dsf;
	std_dev_filter * sdvf;
	//mean_filter * mf;

	unsigned int save_count;
	bool do_raw_save;

	bool saveFrameAvailable;

	uint16_t * raw_save_ptr;

	FILE * raw_save_file;

	u_char * dumb_ptr;
	unsigned int dataHeight;
	unsigned int frHeight;
	unsigned int frWidth;
	//frame_c * curFrame;
	//std::shared_ptr<frame_c> curFrame;
	frame_c* curFrame;
	int pdv_thread_run = 0;

public:
	take_object(int channel_num = 0, int number_of_buffers = 64, int fmsize = 1000, int filter_refresh_rate = 10);
	virtual ~take_object();
	void start();

	unsigned int getDataHeight();
	unsigned int getFrameHeight();
	unsigned int getFrameWidth();

	bool dsfMaskCollected;

	std::vector<float> * getHistogramBins();


	bool std_dev_ready();

	void startCapturingDSFMask();
	void finishCapturingDSFMask();
	void loadDSFMask(std::string file_name);


	void startSavingRaws(std::string, unsigned int );
	void stopSavingRaws();

	void setStdDev_N(int s);

	void doSave();
	camera_t cam_type;
	unsigned int save_framenum;
	//std::list<std::shared_ptr<frame_c> > frame_list;
	std::list<frame_c * > frame_list;

private:
	void pdv_loop();

};

#endif /* TAKEOBJECT_HPP_ */
