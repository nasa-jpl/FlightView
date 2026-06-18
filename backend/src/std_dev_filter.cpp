#include "std_dev_filter.hpp"
#include "constants.h"
#include <math.h>
#include <iostream>
#include <cstring>

#ifdef USE_CUDA
#include "cuda_utils.cuh"
#include <cuda_profiler_api.h>
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
#else
#define HANDLE_ERROR(err) (err)
// CPU implementation uses OpenMP for parallelization
#include <omp.h>
#endif

std_dev_filter::std_dev_filter(int nWidth, int nHeight, int cudaDeviceNumber)
{
    /*! \brief Allocate memory and specify device.
     * \param nWidth The frame width. This is specified initially and cannot be changed during operation.
     * \param nHeight The frame height. This is speccified initially and cannot be changed during operation.
     *
     * To set up the kernel, we have to first allocate memory on the device to which we can copy incoming frames. As the standard
     * deviation calculation is split into two components (The std. dev. image itself and the histogram), there are two separate
     * allocation steps. Additionally, within these steps, any memory specified for input must also have an associated array for output.
     *
     * Finally, the memory for the histogram is specified to copied from the device to the host asynchronously as frames come in.
     */
//	int STD_DEV_DEVICE_NUM = cudaDeviceNumberStatic; 
  //  HANDLE_ERROR(cudaSetDevice(cudaDeviceNumberStatic));

    //printf("[std_dev_filter]: desired CUDA device number: %d, total device count: %d\n",
//		    STD_DEV_DEVICE_NUM, getDeviceCount());
	this->cudaDeviceNumber = cudaDeviceNumber;
	this->STD_DEV_DEVICE_NUM = cudaDeviceNumber;

	width = nWidth; // Making the assumption that all frames in a frame buffer are the same size
	height = nHeight;
	gpu_buffer_head = 0; // read point for the ring buffer data structure
	currentN = 0; // number of complete frames in the ring buffer

#if defined(__APPLE__) && defined(USE_METAL)
	// Initialize pointers to nullptr
	metal_context = nullptr;
	metal_output_ptr = nullptr;
	metal_histogram_ptr = nullptr;
	pictures_cpu = nullptr;
	picture_out_cpu = nullptr;
	histogram_out_cpu = nullptr;
#endif

#ifdef USE_CUDA
	int cudaDevNumberChecker = -1;
	cudaGetDevice(&cudaDevNumberChecker);
	printf("STD_DEV_FILTER: CUDA device actually being used: %d\n", cudaDevNumberChecker);

	// It seems like values above 21 simply do not work.
	for(int d=21; d > 1; d--)
	{
		if((nHeight%d)==0)
		{
			optimalBlockSizeY = d;
			break;
		}
	}
	for(int d=21; d > 1; d--)
	{
		if((nWidth%d)==0)
		{
			optimalBlockSizeX = d;
			break;
		}
	}
	printf("[std_dev_filter]: Optimal GPU Block Size X: %d, Y: %d\n", optimalBlockSizeX, optimalBlockSizeY);

	HANDLE_ERROR(cudaStreamCreate(&std_dev_stream));
	HANDLE_ERROR(cudaMalloc( (void **)&pictures_device, width*height*sizeof(uint16_t)*GPU_FRAME_BUFFER_SIZE)); // Allocate a huge amount of memory on the GPU (N times the size of each frame stored as a u_char)
	HANDLE_ERROR(cudaMalloc( (void **)&picture_out_device, width*height*sizeof(float))); // Allocate memory on GPU for reduce target

	HANDLE_ERROR(cudaMalloc( (void **)&histogram_bins_device, NUMBER_OF_BINS*sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void **)&histogram_out_device, NUMBER_OF_BINS*sizeof(uint32_t)));
	memcpy(histogram_bins,getHistogramBinValues().data(),NUMBER_OF_BINS*sizeof(float));

	HANDLE_ERROR(cudaMemcpyAsync(histogram_bins_device,histogram_bins,NUMBER_OF_BINS*sizeof(float),cudaMemcpyHostToDevice,std_dev_stream)); // Incrementally copies data to device (as each frame comes in it gets copied)
#elif defined(__APPLE__) && defined(USE_METAL)
	// Metal GPU implementation (macOS only)
	printf("[std_dev_filter]: Using Metal GPU acceleration\n");
	
	metal_context = metal_stddev_init(width, height);
	
	if(!metal_context) {
		std::cerr << "[std_dev_filter]: Metal initialization failed, falling back to CPU" << std::endl;
		// Fall back to CPU implementation - allocate CPU buffers
		printf("[std_dev_filter]: Using CPU implementation with OpenMP (threads: %d)\n", omp_get_max_threads());
		
		pictures_cpu = (uint16_t*)aligned_alloc(64, width*height*sizeof(uint16_t)*GPU_FRAME_BUFFER_SIZE);
		picture_out_cpu = (float*)aligned_alloc(64, width*height*sizeof(float));
		histogram_out_cpu = (uint32_t*)aligned_alloc(64, NUMBER_OF_BINS*sizeof(uint32_t));
		
		if(!pictures_cpu || !picture_out_cpu || !histogram_out_cpu) {
			std::cerr << "ERROR: Failed to allocate memory for std_dev_filter CPU buffers" << std::endl;
			abort();
		}
		
		memcpy(histogram_bins, getHistogramBinValues().data(), NUMBER_OF_BINS*sizeof(float));
	} else {
		// Metal initialized successfully
		metal_output_ptr = metal_stddev_get_output(metal_context);
		metal_histogram_ptr = metal_stddev_get_histogram(metal_context);
		
		memcpy(histogram_bins, getHistogramBinValues().data(), NUMBER_OF_BINS*sizeof(float));
	}
	
#else
	// CPU-only implementation: allocate ring buffer in system memory
	printf("[std_dev_filter]: Using CPU implementation with OpenMP (threads: %d)\n", omp_get_max_threads());

	pictures_cpu = (uint16_t*)aligned_alloc(64, width*height*sizeof(uint16_t)*GPU_FRAME_BUFFER_SIZE);
	picture_out_cpu = (float*)aligned_alloc(64, width*height*sizeof(float));
	histogram_out_cpu = (uint32_t*)aligned_alloc(64, NUMBER_OF_BINS*sizeof(uint32_t));

	if(!pictures_cpu || !picture_out_cpu || !histogram_out_cpu) {
		std::cerr << "ERROR: Failed to allocate memory for std_dev_filter CPU buffers" << std::endl;
		abort();
	}

	memcpy(histogram_bins, getHistogramBinValues().data(), NUMBER_OF_BINS*sizeof(float));
#endif
}
std_dev_filter::~std_dev_filter()
{
    /*! Free all devices and allocated memory (except the current picture), and set the device stream to be destroyed. */
#ifdef USE_CUDA
    HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
    HANDLE_ERROR(cudaFree(pictures_device)); // Do not free current picture because it points to a location inside pictures_device
    HANDLE_ERROR(cudaFree(picture_out_device));
    HANDLE_ERROR(cudaFree(histogram_out_device));
    HANDLE_ERROR(cudaFree(histogram_bins_device));
    HANDLE_ERROR(cudaStreamDestroy(std_dev_stream));
#elif defined(__APPLE__) && defined(USE_METAL)
    // Metal GPU cleanup (macOS only)
    if(metal_context) {
        metal_stddev_cleanup(metal_context);
        metal_context = nullptr;
    }
    // Also clean up CPU fallback if it was allocated
    if(pictures_cpu) {
        free(pictures_cpu);
        free(picture_out_cpu);
        free(histogram_out_cpu);
    }
#else
    // CPU-only: free regular memory
    free(pictures_cpu);
    free(picture_out_cpu);
    free(histogram_out_cpu);
#endif
}

void std_dev_filter::update_GPU_buffer(frame_c * frame, unsigned int N, bool skip_compute)
{
    /*! \brief CPU/GPU code for the standard deviation calculation.
     * \param frame The current frame to be worked on.
     * \param N The number of frames to use in the buffer, or the integration length of the calculation.
     * \param skip_compute If true, uploads frame to GPU but skips computation (for frameskip mode).
     */
    static int count __attribute__((unused)) = 0;

#ifdef USE_CUDA
    // GPU implementation
    // Synchronous
    /* Step 1: Set the device, get the status, and create a pointer to the current position on the device ring buffer. */
    HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
    cudaError std_dev_stream_status = cudaStreamQuery(std_dev_stream);
    char *device_ptr = ((char *)(pictures_device)) + (gpu_buffer_head*width*height*sizeof(uint16_t));

    // Asynchronous
    /* Step 2: Copy the current image on the host to the device ring buffer. */
    HANDLE_ERROR(cudaMemcpyAsync(device_ptr,frame->image_data_ptr,width*height*sizeof(uint16_t),cudaMemcpyHostToDevice,std_dev_stream)); // Incrementally copies data to device (as each frame comes in it gets copied)

    if(cudaSuccess == cudaStreamQuery(std_dev_stream) && DEBUG)
    {
        printf("really weird\n"); // Noah wrote this debug line. I'm not sure when or why it triggers...
    }

    // Only perform computation if not skipping (keeps GPU pipeline active but reduces workload)
    if(!skip_compute && cudaSuccess == std_dev_stream_status)
    {
        /* Step 3: If there are no errors, check that there are std. dev. frames ready to be displayed */
        if(prevFrame != NULL)
        {
            prevFrame->has_valid_std_dev = 2; // Ready to display
        }

        frame->has_valid_std_dev = 1; // is processing
        prevFrame = frame;

        /* Step 4: Set the number of blocks and the number of threads per block */
        //dim3 blockDims(BLOCK_SIZE,BLOCK_SIZE,1); // We have 2-dimensional blocks of 20x20 threads... These threads will share their "block_histogram" array on the device
        dim3 blockDims(optimalBlockSizeX,optimalBlockSizeY,1); // We have 2-dimensional blocks of 20x20 threads... These threads will share their "block_histogram" array on the device

        dim3 gridDims(width/blockDims.x, height/blockDims.y,1); // Determine the number of blocks needed for the image

        /* Step 5: Initialize the histogram output array. */
        HANDLE_ERROR(cudaMemsetAsync(histogram_out_device,0,NUMBER_OF_BINS*sizeof(uint32_t),std_dev_stream));

        /* Step 6: Launch the kernel using the wrapper function defined in the device code. */
        std_dev_filter_kernel_wrapper(gridDims,blockDims,0,std_dev_stream,pictures_device, picture_out_device, histogram_bins_device, histogram_out_device, width, height, gpu_buffer_head, N);

        /* Step 7: Check for errors and copy the output arrays off the device. */
        HANDLE_ERROR(cudaPeekAtLastError());
        HANDLE_ERROR(cudaMemcpyAsync(frame->std_dev_data,picture_out_device,width*height*sizeof(float),cudaMemcpyDeviceToHost,std_dev_stream)); //Despite the name, these calls are synchronous w/ respect to the CPU
        HANDLE_ERROR(cudaMemcpyAsync(frame->std_dev_histogram,histogram_out_device,NUMBER_OF_BINS*sizeof(uint32_t),cudaMemcpyDeviceToHost,std_dev_stream));
    }
#elif defined(__APPLE__) && defined(USE_METAL)
    // Metal GPU implementation (macOS only) with CPU fallback
    
    if(metal_context) {
        // Use Metal GPU
        // Step 1: Upload current frame to GPU ring buffer
        metal_stddev_update_frame(metal_context, frame->image_data_ptr);
        
        // Step 2: Increment currentN
        if(currentN < MAX_N) {
            currentN++;
        }
        
        // Check if we have enough frames for computation
        unsigned int usableN = (currentN < N) ? currentN : N;
        
        if(!skip_compute && usableN >= 2) {  // Need at least 2 frames for std dev
            // Mark previous frame as ready
            if(prevFrame != NULL) {
                prevFrame->has_valid_std_dev = 2; // Ready to display
            }
            
            frame->has_valid_std_dev = 1; // is processing
            prevFrame = frame;
            
            // Step 3: Compute stddev on GPU
            metal_stddev_compute(metal_context, usableN, true, histogram_bins);
            
            // Step 4: Copy results from shared memory (already mapped)
            memcpy(frame->std_dev_data, metal_output_ptr, width * height * sizeof(float));
            memcpy(frame->std_dev_histogram, metal_histogram_ptr, NUMBER_OF_BINS * sizeof(uint32_t));
        }
    } else {
        // Fall back to CPU implementation
        // Step 1: Copy current frame into ring buffer
        uint16_t *buffer_ptr = pictures_cpu + (gpu_buffer_head * width * height);
        memcpy(buffer_ptr, frame->image_data_ptr, width * height * sizeof(uint16_t));
        
        // Step 2: Increment buffer position and frame count
        gpu_buffer_head = (gpu_buffer_head + 1) % GPU_FRAME_BUFFER_SIZE;
        if(currentN < MAX_N) {
            currentN++;
        }
        
        // Check if we have enough frames for computation
        unsigned int usableN = (currentN < N) ? currentN : N;
        
        if(!skip_compute && usableN >= 2) {  // Need at least 2 frames for std dev
            // Mark previous frame as ready
            if(prevFrame != NULL) {
                prevFrame->has_valid_std_dev = 2; // Ready to display
            }
            
            frame->has_valid_std_dev = 1; // is processing
            prevFrame = frame;
            
            // Step 3: Compute stddev on CPU using OpenMP
            unsigned int start_idx = (gpu_buffer_head >= usableN) ? (gpu_buffer_head - usableN) : (GPU_FRAME_BUFFER_SIZE + gpu_buffer_head - usableN);
            
            // Clear histogram
            memset(histogram_out_cpu, 0, NUMBER_OF_BINS * sizeof(uint32_t));
            
            // Compute for each pixel
            #pragma omp parallel for collapse(2)
            for(unsigned int y = 0; y < height; y++) {
                for(unsigned int x = 0; x < width; x++) {
                    unsigned int pixel_idx = y * width + x;
                    
                    // Calculate mean
                    float sum = 0.0f;
                    for(unsigned int i = 0; i < usableN; i++) {
                        unsigned int frame_idx = (start_idx + i) % GPU_FRAME_BUFFER_SIZE;
                        uint16_t value = pictures_cpu[frame_idx * width * height + pixel_idx];
                        sum += value;
                    }
                    float mean = sum / usableN;
                    
                    // Calculate variance
                    float variance_sum = 0.0f;
                    for(unsigned int i = 0; i < usableN; i++) {
                        unsigned int frame_idx = (start_idx + i) % GPU_FRAME_BUFFER_SIZE;
                        uint16_t value = pictures_cpu[frame_idx * width * height + pixel_idx];
                        float diff = value - mean;
                        variance_sum += diff * diff;
                    }
                    
                    float stddev = sqrtf(variance_sum / (usableN - 1));
                    picture_out_cpu[pixel_idx] = stddev;
                    
                    // Update histogram (thread-safe atomic increment)
                    for(int bin = 0; bin < NUMBER_OF_BINS; bin++) {
                        if(stddev >= histogram_bins[bin] && (bin == NUMBER_OF_BINS-1 || stddev < histogram_bins[bin+1])) {
                            #pragma omp atomic
                            histogram_out_cpu[bin]++;
                            break;
                        }
                    }
                }
            }
            
            // Copy results to frame
            memcpy(frame->std_dev_data, picture_out_cpu, width * height * sizeof(float));
            memcpy(frame->std_dev_histogram, histogram_out_cpu, NUMBER_OF_BINS * sizeof(uint32_t));
        }
    }
#else
    // CPU implementation using OpenMP for parallelization
    
    // Step 1: Copy current frame into ring buffer
    uint16_t *buffer_ptr = pictures_cpu + (gpu_buffer_head * width * height);
    memcpy(buffer_ptr, frame->image_data_ptr, width * height * sizeof(uint16_t));
    
    // Step 2: Increment currentN before computing (we now have one more frame)
    if(currentN < MAX_N) {
        currentN++;
    }
    
    // Check if we have enough frames for computation
    unsigned int usableN = (currentN < N) ? currentN : N;
    
    if(!skip_compute && usableN >= 2) {  // Need at least 2 frames for std dev
        // Mark previous frame as ready
        if(prevFrame != NULL) {
            prevFrame->has_valid_std_dev = 2; // Ready to display
        }
        
        frame->has_valid_std_dev = 1; // is processing
        prevFrame = frame;
        
        // Step 3: Compute mean and std dev for each pixel using OpenMP
        memset(histogram_out_cpu, 0, NUMBER_OF_BINS * sizeof(uint32_t));
        
        const unsigned int frame_size = width * height;
        
        #pragma omp parallel
        {
            // Thread-local histogram
            uint32_t local_histogram[NUMBER_OF_BINS] = {0};
            
            #pragma omp for schedule(static)
            for(unsigned int pixel = 0; pixel < frame_size; pixel++) {
                // Compute mean and variance for this pixel across N frames
                double sum = 0.0;
                double sum_sq = 0.0;
                
                // Access frames in ring buffer
                for(unsigned int f = 0; f < usableN; f++) {
                    int frame_idx = (gpu_buffer_head + GPU_FRAME_BUFFER_SIZE - f) % GPU_FRAME_BUFFER_SIZE;
                    uint16_t val = pictures_cpu[frame_idx * frame_size + pixel];
                    sum += val;
                    sum_sq += (double)val * val;
                }
                
                double mean = sum / usableN;
                double variance = (sum_sq / usableN) - (mean * mean);
                double std_dev = sqrt(variance > 0.0 ? variance : 0.0);
                
                picture_out_cpu[pixel] = (float)std_dev;
                
                // Update histogram (find appropriate bin)
                for(unsigned int bin = 0; bin < NUMBER_OF_BINS; bin++) {
                    if(std_dev <= histogram_bins[bin]) {
                        local_histogram[bin]++;
                        break;
                    }
                }
            }
            
            // Combine thread-local histograms
            #pragma omp critical
            {
                for(unsigned int bin = 0; bin < NUMBER_OF_BINS; bin++) {
                    histogram_out_cpu[bin] += local_histogram[bin];
                }
            }
        }
        
        // Step 4: Copy results to frame
        memcpy(frame->std_dev_data, picture_out_cpu, frame_size * sizeof(float));
        memcpy(frame->std_dev_histogram, histogram_out_cpu, NUMBER_OF_BINS * sizeof(uint32_t));
        
        // Note: Do NOT set frame->has_valid_std_dev = 2 here!
        // The frame stays at status 1 (processing) and will be marked as 2 (ready)
        // on the NEXT frame (see line 180 above where prevFrame is marked ready).
        // This matches the async behavior expected by frame_worker.cpp
    }
#endif

    // Synchronous
    /*! Step 8: Increment the ring buffer. As this is a ring buffer, the
     * gpu_buffer_head will return to the beginning of the array when it reaches the end of the allocated space.
     */
    if(++gpu_buffer_head == GPU_FRAME_BUFFER_SIZE) //Increment and test for ring buffer overflow
        gpu_buffer_head = 0; // If overflow, than start overwriting the front
#if !defined(USE_CUDA) && !(defined(__APPLE__) && defined(USE_METAL))
    // For CPU: currentN already incremented before calculation
#else
    // For CUDA/Metal: increment currentN here (after async processing starts)
    if(currentN < MAX_N) // If the frame buffer has not been fully populated
    {
        currentN++; //Increment how much history is available
    }
#endif
    count++;
}
uint16_t * std_dev_filter::getEntireRingBuffer() //For testing only
{
    /*! Captures the ring buffer of standard deviation frames. */
#ifdef USE_CUDA
    HANDLE_ERROR(cudaSetDevice(STD_DEV_DEVICE_NUM));
    uint16_t * out = new uint16_t[width*height*MAX_N];
    HANDLE_ERROR(cudaMemcpy(out,pictures_device,width*height*sizeof(uint16_t)*MAX_N,cudaMemcpyDeviceToHost));
    return out;
#elif defined(__APPLE__) && defined(USE_METAL)
    // Metal or CPU fallback
    if(metal_context) {
        // Metal doesn't expose the ring buffer directly
        std::cerr << "[std_dev_filter]: getEntireRingBuffer not supported with Metal" << std::endl;
        return nullptr;
    } else {
        // CPU fallback
        uint16_t * out = new uint16_t[width*height*MAX_N];
        memcpy(out, pictures_cpu, width*height*sizeof(uint16_t)*MAX_N);
        return out;
    }
#else
    uint16_t * out = new uint16_t[width*height*MAX_N];
    memcpy(out, pictures_cpu, width*height*sizeof(uint16_t)*MAX_N);
    return out;
#endif
}
std::vector <float> * std_dev_filter::getHistogramBins()
{
    /*! Captures all current histogram bins. */
    shb.assign(histogram_bins,histogram_bins+NUMBER_OF_BINS);
    return &shb;
}
bool std_dev_filter::outputReady()
{
    /*! Returns true if std. dev. frames are ready to be plotted. */
    return currentN >= 2; // Need at least 2 frames for std dev calculation
}
