#include <cstdio>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include "constants.h"
#ifndef FFT_H_
#define FFT_H_

/*! \file
 * \brief Calulates the fast fourier transform of a time series.
 *
 * The FFT is calculated using the complex value of the time series, using the bit twiddling method. It is then converted back to its real
 * magnitude only.
 */

class fft {
	std::complex<float> * CFFT;

public:
	fft();
	virtual ~fft();
void doRealFFT(float * arr, unsigned int ring_head, float *fft_real_result);

std::complex<float> * doFFT(float * arr, unsigned int ring_head);
std::complex<float> * doFFT(std::complex<float> * arr, unsigned int len);

private:
void bitReverseOrder(std::complex<float> * arr, unsigned int len);
};

#endif /* FFT_H_ */
