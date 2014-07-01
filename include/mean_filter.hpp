#include <stdint.h>

#ifndef HORIZONTALMEANFILTER_CUH_
#define HORIZONTALMEANFILTER_CUH_
#include <boost/thread.hpp>
class mean_filter {
public:
	mean_filter(int nWidth, int nHeight);

	virtual ~mean_filter();

	void start_mean(uint16_t * pic_in);

	void calculate_means();
	float * wait_horizontal_mean();
	float * wait_vertical_mean();
private:
	boost::thread mean_thread;
	int width;
	int height;

	uint16_t * picture_in;
	float * vert;
	float * horiz;
};

#endif /* HORIZONTALMEANFILTER_CUH_ */
