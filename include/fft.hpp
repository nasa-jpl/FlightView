/*
 * fft.h
 *
 *  Created on: Jul 7, 2014
 *      Author: nlevy
 */
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include "constants.h"
#ifndef FFT_H_
#define FFT_H_

class fft {
	std::complex<float> * CFFT;

public:
	fft() {
		CFFT = new std::complex<float>[MAX_FFT_SIZE];
	};
void doRealFFT(float * arr, unsigned int len, unsigned int ring_head, float *fft_real_result);

std::complex<float> * doFFT(float * arr, unsigned int len, unsigned int ring_head);
std::complex<float> * doFFT(std::complex<float> * arr, unsigned int len);

private:
void bitReverseOrder(std::complex<float> * arr, unsigned int len);
};

#endif /* FFT_H_ */
