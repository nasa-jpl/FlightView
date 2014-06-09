
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

int mainno()
{
	int width = 1280;
	int height = 720;
	uint16_t fake[1280*720];
	dark_subtraction_filter dsf(width,height);
	dsf.update_mask_collection(fake);
	fake[100000] = 2;
	dsf.update_mask_collection(fake);
	dsf.update_mask_collection(fake);
	dsf.update_mask_collection(fake);


	std::cout << " no segfault" << std::endl;
}
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
	int samples = 30;
	int read_addr = 9300;
	std::queue<uint16_t> pixel_hist;
	uint16_t lastfc = 0;
	//to.initFilters(samples);
	to.startCapturingDSFMask();
			while(1)
			{

				//std::cout << "i is:"<<i << std::endl;
				boost::shared_ptr<frame> frame =to.getFrontFrame(); //This blocks until a frame is available
				if(!to.isChroma && frame->framecount -1 != lastfc)
				{
					std::cerr << "WARN MISSED FRAME" << frame->framecount << " " << lastfc << std::endl;
				}
				lastfc = frame->framecount;
				if(i%(samples/10) == 0 && i <= samples)
				{
					std::cout << "i is:" << i << " value is:" <<frame->image_data_ptr[read_addr] << std::endl;
				}


				if(i%(samples/10) == 0 && i > samples)
				{
					//std::cout << frame->image_data_ptr[read_addr] << std::endl;

					std::cout << frame->image_data_ptr[read_addr] << ", masked is: " << to.getDarkSubtractedData()[read_addr] << std::endl;
				}

				if(i == samples)
				{
					to.finishCapturingDSFMask();
					std::cout << "mask collected" << std::endl;
				}

				i++;

			}
	return 0;
}
