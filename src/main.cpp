
#include "take_object.hpp"
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
#include <queue>
#include "dark_subtraction_filter.cuh"
#include "std_dev_filter.cuh"
#include <fstream>
using namespace std;


//This method is used for testing the std. deviation filter

void std_dev_test()
{
	int width = 640;
	int height = 480;
	int area = width*height;
	const char * fname = "float_data2.bin";
	uint16_t * a = new uint16_t[area];
	uint16_t * b = new uint16_t[area];

	for(int i = 0; i < area; i++)
	{
		a[i] = 8000;
		if(i < area/2)
		{
			b[i] = i%16000; //for first half of array std. dev is ~ 10
		}
		else
		{
			b[i] = 8000; //For second half of array std. dev = 0
		}
	}
	std_dev_filter * sdvf = new std_dev_filter(width,height);
	cout << "created sdvf obj" << endl;
	for(int i = 0; i < 1010; i++)
	{
		sdvf->update_GPU_buffer(a); //add 100 each of a and b array to std_dev buffer.
		sdvf->update_GPU_buffer(b);
	}
	cout << "update buffer called 200x" << endl;
	sdvf->start_std_dev_filter(150);
	cout << "waiting on std dev filter" << endl;
	boost::shared_array < float > result = sdvf->wait_std_dev_filter();

/*
	//For figuring out whether things are getting put on the GPU correctly.
	uint16_t * pictures_on_device = sdvf->getEntireRingBuffer();
	int frame_read_num = 3;
	for(int offset = 0; offset < width*height; offset++)
	{
		result[offset] = (float) *(pictures_on_device + offset+(width*height*frame_read_num));
	}
	*/
	cout << "result @ 100: " << result[1000] << " result @ 2area/3: " << result[2*area/3] << endl;
	cout << "result @ 100: " << result[10020] << " result @ 2area/3: " << result[2*area/3] << endl;

	ofstream myFile (fname, ios::out | ios::binary);
	myFile.write ((char*)result.get(), height*width*sizeof(float));
	myFile.close();

	char cmd[100];
	sprintf(cmd, "show_raw_float %s %i %i", fname, width, height);
	system(cmd);
}
//This method is used for testing bits and pieces of things w/o a sensor hooked up to the computer
void dsf_test()
{
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
void sensor_grab_test()
{
	take_object to;
	to.start();
	//std::cout << "notified once" << std::endl;
	//while(1)
	std::string fname;


	int i = 0;
	//int read_me = 250;
	int samples = 300;

	int start_saves = 1100;

	int end_saves = 3100;

	int std_dev_mem = 999;
	int read_addr = 9300;
	std::queue<uint16_t> pixel_hist;
	//to.initFilters(samples);
	std::cout << "isChroma? " << to.isChroma << std::endl;
	to.setStdDev_N(std_dev_mem);
	to.startCapturingDSFMask();
	while(1)
	{
		boost::shared_ptr<frame> frame =to.getFrontFrame(); //This blocks until a frame is available

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
		if(i == start_saves)
		{
			to.startSavingRaws("cuda_take_out_1000sd.raw");
			to.startSavingSTD_DEVs("cuda_devs_f32_1000sd.raw");
		}
		if(i == end_saves)
		{
			to.stopSavingRaws();
			to.stopSavingSTD_DEVs();
		}
		i++;
	}
}

int main()
{
	sensor_grab_test();
	//std_dev_test();
	return 0;
}

