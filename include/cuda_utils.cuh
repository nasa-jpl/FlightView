/*
 * cuda_utils.cuh
 *
 *  Created on: Jun 2, 2014
 *      Author: nlevy
 */
#include <cstdio>
#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_
//From CUDA by Example
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file,int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
};

static int getDeviceCount()
{
	int nDevices;
	HANDLE_ERROR(cudaGetDeviceCount(&nDevices));
	return nDevices;
}


#endif /* CUDA_UTILS_CUH_ */
