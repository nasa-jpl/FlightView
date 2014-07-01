#include "mean_filter.hpp"


mean_filter::mean_filter(int nWidth, int nHeight)
{
	width = nWidth;
	height = nHeight;
	vert = new float[height];
	horiz = new float[width];
	picture_in = new uint16_t[width*height];
}

mean_filter::~mean_filter()
{

}
void mean_filter::start_mean(uint16_t * pic_in)
{
	memcpy(picture_in,pic_in,width*height*sizeof(uint16_t));
	mean_thread = boost::thread(&mean_filter::calculate_means, this);

}
void mean_filter::calculate_means()
{
	for(int r = 0; r < height; r++)
	{
		vert[r]=0;
	}
	for(int c = 0; c < width; c++)
	{
		horiz[c]=0;
	}
	for(int r = 0; r < height; r++)
	{
		for(int c = 0; c < width; c++)
		{
			vert[r] += picture_in[r*width + c];
			horiz[c] += picture_in[r*width + c];
		}
	}

	for(int r = 0; r < height; r++)
	{
		vert[r]/=height;
	}
	for(int c = 0; c < width; c++)
	{
		horiz[c]/=width;
	}
}
float * mean_filter::wait_horizontal_mean()
{
	mean_thread.join();
	return horiz;
}
float * mean_filter::wait_vertical_mean()
{
	mean_thread.join();
	return vert;
}
