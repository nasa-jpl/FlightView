// Metal GPU acceleration for std_dev_filter
// macOS-only header

#ifndef STD_DEV_FILTER_METAL_H
#define STD_DEV_FILTER_METAL_H

#ifdef __APPLE__
#ifdef USE_METAL

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal context
// Returns opaque context pointer, or nullptr on failure
void* metal_stddev_init(int width, int height);

// Cleanup Metal context
void metal_stddev_cleanup(void* context);

// Upload new frame to GPU ring buffer
void metal_stddev_update_frame(void* context, const uint16_t* frame_data);

// Compute standard deviation across N frames
// If compute_histogram is true and histogram_bins is provided, also computes histogram
void metal_stddev_compute(void* context, uint32_t N, bool compute_histogram,
                         const float* histogram_bins);

// Get pointer to stddev output (shared memory, directly accessible from CPU)
float* metal_stddev_get_output(void* context);

// Get pointer to histogram output (shared memory, directly accessible from CPU)
uint32_t* metal_stddev_get_histogram(void* context);

#ifdef __cplusplus
}
#endif

#endif // USE_METAL
#endif // __APPLE__

#endif // STD_DEV_FILTER_METAL_H
