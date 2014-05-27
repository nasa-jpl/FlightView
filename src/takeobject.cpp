/*
 * takeobject.cpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#include "takeobject.hpp"
#include <iostream>
take_object::take_object(int channel_num, int number_of_buffers)
{
	this->channel = channel_num;
	this->numbufs = number_of_buffers;
}
take_object::~take_object()
{
	pdv_close(pdv_p);
	delete this->pdv_p;
}


void take_object::start()
{
	//The internet says that I need to pass a pointer to the threadable function, and a reference to the calling instance of take_object (this)
	pdv_thread = std::thread(&take_object::pdv_init, this);
}
void take_object::pdv_init()
{
	this->pdv_p = NULL;
	this->pdv_p = pdv_open_channel(EDT_INTERFACE,0,this->channel);
	if(pdv_p == NULL)
	{
		std::cout << "Oh gawd, couldn't open channel" << std::endl;
		//TODO: throw exception
		return;
	}
	//TODO: Sizing stuff
	int size = pdv_get_dmasize(pdv_p); //Not using this at the moment
	pdv_multibuf(pdv_p,this->numbufs);

}
