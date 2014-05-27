/*
 * takeobject.hpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#ifndef TAKEOBJECT_HPP_
#define TAKEOBJECT_HPP_

#include <thread>
#include <atomic>
#include <mutex>
#include "edtinc.h"
class take_object {
	PdvDev * pdv_p;
	std::thread pdv_thread;
	unsigned int channel;
	unsigned int numbufs;
	unsigned int frame_memory_size;
public:
	take_object(int channel_num = 2, int number_of_buffers = 4, int fmsize = 1000);
	virtual ~take_object();
	void start();
private:
	void pdv_init();

};

#endif /* TAKEOBJECT_HPP_ */
