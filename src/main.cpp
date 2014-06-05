
#include "take_object.hpp"
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
#include <queue>
using namespace std;
#define INFINITE
int main()
{
	take_object to;
	to.start();
	//std::cout << "notified once" << std::endl;
	//while(1)
	int size = 640*481*2;
	std::string fname;


	int i = 0;
	int read_me = 250;
	int samples = 500;
	std::queue<uint16_t> pixel_hist;
	uint16_t lastfc = 0;
	to.initFilters(samples);
	while(1)
	{
		std::cout << i << std::endl;
		boost::shared_ptr<frame> frame =to.getFrontFrame();
		//lock.unlock(); //Once we've got a pointer to the frame, let lock go!
		//std::cout << "new frame availblable fc: " << frame->framecount << " timestamp: " << frame->cmTime << std::endl;
		//std::cout<<to.height;
		int value_targ = (frame->image_data_ptr[read_me*BYTES_PER_PIXEL+1] << 8 | frame->image_data_ptr[read_me]);

		if(!to.isChroma && frame->framecount -1 != lastfc)
		{
			std::cerr << "WARN MISSED FRAME" << frame->framecount << " " << lastfc << std::endl;
		}
		lastfc = frame->framecount;

		std::cout << frame->image_data_ptr[20] << std::endl;
		if(i > samples && i%10 == 0)
		{
			/*
			while(! pixel_hist.empty())
			{
				std::cout << ' ' << pixel_hist.front() << ',';
				pixel_hist.pop(); //Unlike every other language ever, this does not return a value.
			}
			 */
			//boost::shared_array <float> bpt = to.getStdDevData();
			//std::cout << bpt[read_me] << std::endl;
			//std::cout  << "\nstd_dev: " << bpt[read_me] <<   std::endl;
			return 0;
		}
		//pixel_hist.push(value_targ);
		//std::cout << value_targ << std::endl;

		i++;

	}
	return 0;
}
