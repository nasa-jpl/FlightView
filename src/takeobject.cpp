/*
 * takeobject.cpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#include "takeobject.hpp"

take_object::take_object(int channel_num)
{
	this->channel = channel_num;
}
take_object::~take_object()
{
	// TODO Auto-generated destructor stub
}


void take_object::start()
{
	PdvDev * pdv_p = NULL;
	pdv_p = pdv_open_channel(EDT_INTERFACE,0,this->channel);
	if(pdv_p == NULL)
	{
		//throw exception
	}
}
