
#include "take_object.hpp"
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
#include <queue>
#include "dark_subtraction_filter.hpp"
#include "std_dev_filter.hpp"
//#include "fft.hpp"
#include <fstream>
using namespace std;



void simple_sensor_grab()
{
	take_object to;
	to.start();
	cout << "ruunning to" << std::endl;
	unsigned long c = 0;

	unsigned oldcount = 0;
	while(c <= 8)
	{
		unsigned long delta = to.count - oldcount;
		oldcount = to.count;
		printf("how to delta: %lu\n",delta);
		usleep(1000000);
		//if(c==3)
			//to.startSavingRaws("ruhroh.raw",1000);
		//printf("save framenum %u, listsize %u", to.save_framenum, to.saving_list.size());
		c++;
	}
	//delete to;
	printf("delete to!\n");
}
void fft_test()
{
	fft myFFT;
	float * data_in = new float[1024];
	float * data_out = new float[1024];

	FILE * f = fopen("fake_fourier_in.bin","rb");
	fread(data_in,sizeof(float),FFT_INPUT_LENGTH,f);
	fclose(f);
	//data_in = myFFT.doRealFFT(data_in,1024,0);
	myFFT.doRealFFT(data_in,0,data_out);
	FILE * fw = fopen("rfft_out.bin","wb");
	fwrite(data_out,sizeof(float),FFT_INPUT_LENGTH/2,fw);
	fclose(fw);
	printf("fft calculated\n");

}
void simple_pdv_test()
{
	PdvDev * pdv_p = pdv_open_channel(EDT_INTERFACE,0,0);
	pdv_start_images(pdv_p,64);
	//unsigned int height = pdv_get_height(pdv_p);
	unsigned int width = pdv_get_width(pdv_p);
	uint16_t * ptr;
	while(1)
	{
		ptr =reinterpret_cast<uint16_t *>(pdv_wait_image(pdv_p));
		printf("@ 100 100 %u", ptr[width*100 + 100]);
	}


}
int main()
{
	//simple_pdv_test();
	//fft_test();
	//sensor_grab_test();
	simple_sensor_grab();
	//std_dev_test();
	return 0;
}

