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
#include <cstdint>
#include <cstdio>
#include <ostream>
#include <string>
#include <atomic>
#include <boost/shared_array.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "std_dev_filter.hpp"
#include "chroma_translate_filter.hpp"
#include "dark_subtraction_filter.hpp"
#include "mean_filter.hpp"
#include "edtinc.h"
#include "camera_types.h"
#include "frame_c.hpp"
#include "constants.h"

static const bool CHECK_FOR_MISSED_FRAMES_6604A = true;
const static int NUMBER_OF_FRAMES_TO_BUFFER = 1500;





class take_object {
	PdvDev * pdv_p;
	boost::thread pdv_thread;
	boost::thread saving_thread;
	unsigned int size;

	unsigned int channel;
	unsigned int numbufs;
	unsigned int filter_refresh_rate;

	int std_dev_filter_N;
	int lastfc;
	dark_subtraction_filter * dsf;
	std_dev_filter * sdvf;
	//mean_filter * mf;

	unsigned int save_count;
	bool do_raw_save;

	bool saveFrameAvailable;

	uint16_t * raw_save_ptr;


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
    bool useDSF = false;


	bool std_dev_ready();

    void setInversion( bool, unsigned int );
    void chromaPixRemap( bool );
    void update_start_row( int );
    void update_end_row( int );

	void startCapturingDSFMask();
	void finishCapturingDSFMask();
	void loadDSFMask(std::string file_name);

    void updateHorizRange( int, int );
    void updateVertRange( int, int );

	void startSavingRaws(std::string, unsigned int );
	void stopSavingRaws();

	void setStdDev_N(int s);

	void doSave(frame_c * frame);
	camera_t cam_type;
	//std::atomic_uint_fast32_t save_framenum;
	std::atomic <uint_fast32_t> save_framenum;
	//std::list<std::shared_ptr<frame_c> > frame_list;
	//std::list<frame_c * > frame_list;
	frame_c * frame_ring_buffer;
	unsigned long count = 0;
	std::list<uint16_t *> saving_list;

private:
	void pdv_loop();
	void savingLoop(std::string);

    unsigned int invFactor; // inversion factor as determined by the maximum possible pixel magnitude
    int meanStartRow;
    int meanHeight;
    int meanStartCol;
    int meanWidth;
    bool inverted = false;
    bool chromaPix = false; // Enable Chroma Pixel Mapping? (Chroma Translate filter?)

};

#endif /* TAKEOBJECT_HPP_ */
