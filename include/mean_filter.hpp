#ifndef MEAN_FILTER_HPP
#define MEAN_FILTER_HPP
#include <cstdint>
#include <ccomplex>
#include <boost/thread.hpp>
#include <atomic>
#include "frame_c.hpp"
#include "fft.hpp"
#include "constants.h"

/*! \brief Calculates the mean of image data within an x and y range and performs the Fast Fourier Transform.
 * \paragraph
 *
 * The Mean Filter calculates the vertical and horizontal mean values of the data and the FFT in a separate thread from the
 * main producer loop. Each time an instance of the mean filter is called, it is opened in a new boost thread, which is deallocated
 * at the end of the calculation. The type of FFT and whether or not to use dark subtracted data is determined outside of the scope
 * of this filter, so we must pass in this information, along with the coorinates from which to perform the mean, as parameters.
 * By default, a frame mean will simply be a mean using the frame's geometry as input parameters.
 *
 * FFT types are also defined in this header.
 * \author JP Ryan
 * \author Noah Levy
 */

static float mean_ring_buffer[FFT_MEAN_BUFFER_LENGTH];
static std::atomic_uint_least16_t mean_ring_buffer_head;
enum FFT_t {PLANE_MEAN, VERT_CROSS, TAP_PROFIL};

class mean_filter {
public:
    mean_filter(frame_c * frame,
                unsigned long frame_count,
                int startCol,
                int endCol,
                int startRow,
                int endRow,
                int actualWidth,
                bool useDSF,
                FFT_t FFTtype);
    // Ridiculous parameter list lol :P

	void start_mean();
	void calculate_means();
	void wait_mean();

	fft myFFT;

private:
	boost::thread mean_thread;
    int beginCol;
	int width;
    int beginRow;
    int height;
    int frWidth;
	bool useDSF;
    int FFTtype;
    float tap_profile[TAP_WIDTH*MAX_HEIGHT];
	float frame_mean;
	unsigned int mean_ring_buffer_fft_head;
	unsigned long frame_count;
	frame_c * frame;
};

#endif /* MEAN_FILTER_HPP */
