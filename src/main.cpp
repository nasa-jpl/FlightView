
#include "take_object.hpp"
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
#include <queue>
#include "dark_subtraction_filter.cuh"
using namespace std;
#define INFINITE

//This method is used for testing bits and pieces of things w/o a sensor hooked up to the computer
int maino()
{
	int i = 0;
	int read_me = 250;
	int samples = 300;
	int read_addr = 9300;
	int width = 1280;
	int height = 720;
	uint16_t fake[1280*720];
	for(int y = 0; y<height;y++)
	{
		for(int x = 0; x<width;x++)
		{
			fake[y*width+x] = 5761;
		}
	}
	dark_subtraction_filter dsf(width,height);
	dsf.start_mask_collection();
	//for(int i = 0; i < 300; i++)
	{
		dsf.update_mask_collection(fake);
	}
	dsf.finish_mask_collection();

	dsf.update_dark_subtraction(fake);


	std::cout << fake[read_addr] << ", masked is: " << fixed << setprecision(3) << dsf.wait_dark_subtraction()[read_addr] << std::endl;
	std::cout << " no segfault" << std::endl;
}
int main()
{
	take_object to;
	to.start();
	//std::cout << "notified once" << std::endl;
	//while(1)
	std::string fname;


	int i = 0;
	int read_me = 250;
	int samples = 300;
	int read_addr = 9300;
	std::queue<uint16_t> pixel_hist;
	uint16_t lastfc = 0;
	//to.initFilters(samples);
	std::cout << "isChroma? " << to.isChroma << std::endl;
	to.startCapturingDSFMask();
	while(1)
	{

		boost::shared_ptr<frame> frame =to.getFrontFrame(); //This blocks until a frame is available
		if(!to.isChroma && frame->framecount -1 != lastfc)
		{
			std::cerr << "WARN MISSED FRAME" << frame->framecount << " " << lastfc << std::endl;
		}
		lastfc = frame->framecount;
		/*
		if(i%(samples/10) == 0 && i <= samples)
		{
			std::cout << "i is:" << i << " value is:" <<frame->image_data_ptr[read_addr] << std::endl;
		}
*/

		if(i%(samples/10) == 0 && i != samples)
		{
			//std::cout << frame->image_data_ptr[read_addr] << std::endl;

			std::cout << frame->image_data_ptr[read_addr] << ", masked is: " << fixed << setprecision(3) << frame->dsf_data[read_addr] << std::endl;
		}

		if(i == samples)
		{
			to.finishCapturingDSFMask();
			std::cout << "mask_value" << to.getDarkSubtractedData()[read_addr] << std::endl;
		}

		i++;

	}
	return 0;
}
