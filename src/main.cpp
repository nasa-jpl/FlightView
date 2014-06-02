
#include "take_object.hpp"
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
using namespace std;
#define INFINITE
int main()
{
	take_object to;
	boost::unique_lock<boost::mutex> lock(to.framebuffer_mutex);
	to.start();
	to.newFrameAvailable.wait(lock);
	//std::cout << "notified once" << std::endl;
	//while(1)
	int size = 640*481*2;
	std::string fname;

#ifndef INFINITE
	for(int i = 0; i < 50; i++)
	{
		to.newFrameAvailable.wait(lock);
		std::cout << "new frame availblable fc: " << (* to.getFrontFrame()).framecount << " timestamp: " << to.getFrontFrame()->cmTime << std::endl;
		fname = "raws/raw_f" + boost::lexical_cast<std::string>(i) + ".raw";
		char *cstr = new char[fname.length() + 1];
		strcpy(cstr, fname.c_str());
		dvu_write_raw(size, to.getFrontFrame()->raw_data, cstr);

	}
#else
	int i = 0;
	while(1)
	{
		to.newFrameAvailable.wait(lock);
		std::cout << "new frame availblable fc: " << (* to.getFrontFrame()).framecount << " timestamp: " << to.getFrontFrame()->cmTime << std::endl;
		std::cout<<to.height;

		if(i > 30)
		{
		u_char * result = to.getStdDevFrame(30).get();

		std::cout << (result[1001] << 8 || result[1000]) << std::endl;
		return 3;
		}
		i++;
	}
#endif
	return 0;
}
