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
#ifndef FFT_H_
#define FFT_H_
static const unsigned int MAX_FFT_SIZE = 4096;

class fft {
	float * realFFT;
	std::complex<float> * CFFT;

public:
	fft() {
		realFFT=new float[MAX_FFT_SIZE/2];
		CFFT = new std::complex<float>[MAX_FFT_SIZE];
	};
float * doRealFFT(float * arr, unsigned int len, unsigned int ring_head);

std::complex<float> * doFFT(float * arr, unsigned int len, unsigned int ring_head);
std::complex<float> * doFFT(std::complex<float> * arr, unsigned int len);

private:
void bitReverseOrder(std::complex<float> * arr, unsigned int len);
};

#endif /* FFT_H_ */
