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
	std::thread t;
	unsigned int channel;
public:
	take_object(int channel_num = 2);
	virtual ~take_object();
	void start();
};

#endif /* TAKEOBJECT_HPP_ */
