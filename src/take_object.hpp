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
#include <boost/thread.hpp>
//#include <boost/atomic.hpp> //Atomic isn't in the boost library till 1.5.4, debian wheezy has 1.4.9 :(
#include <boost/thread/mutex.hpp>
#include <boost/circular_buffer.hpp>
#include "edtinc.h"
#include "frame.hpp"
class take_object {
	PdvDev * pdv_p;
	boost::circular_buffer<frame> frame_buffer;
	boost::thread pdv_thread;

	unsigned int size;
	unsigned int height;
	unsigned int width;
	unsigned int channel;
	unsigned int numbufs;
	unsigned int frame_history_size;
public:
	take_object(int channel_num = 0, int number_of_buffers = 4, int fmsize = 1000);
	virtual ~take_object();
	void start();
private:
	void pdv_init();
	void append_to_frame_buffer(u_char *, int framesize);

};

#endif /* TAKEOBJECT_HPP_ */
