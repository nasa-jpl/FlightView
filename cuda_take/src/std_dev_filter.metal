#include <metal_stdlib>
using namespace metal;

// Standard deviation computation kernel
// Processes entire image in parallel, computing stddev across N frames for each pixel
kernel void compute_stddev(
    constant uint16_t* frames [[buffer(0)]],      // Ring buffer of N frames
    device float* output [[buffer(1)]],           // Output stddev image
    constant uint32_t& width [[buffer(2)]],
    constant uint32_t& height [[buffer(3)]],
    constant uint32_t& N [[buffer(4)]],           // Number of frames to use
    constant uint32_t& buffer_size [[buffer(5)]], // Total ring buffer size (GPU_FRAME_BUFFER_SIZE)
    constant uint32_t& buffer_head [[buffer(6)]], // Current write position
    uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * width + gid.x;
    
    // Use Welford's online algorithm for numerical stability
    // This avoids catastrophic cancellation in variance calculation
    float mean = 0.0f;
    float M2 = 0.0f;
    
    // Calculate which frame to start from (going backwards from head)
    uint32_t start_frame = (buffer_head >= N) ? (buffer_head - N) : (buffer_size + buffer_head - N);
    
    // Iterate through N frames in ring buffer
    for (uint32_t i = 0; i < N; i++) {
        // Calculate frame index in ring buffer
        uint32_t frame_idx = (start_frame + i) % buffer_size;
        uint32_t frame_offset = frame_idx * width * height;
        
        // Get pixel value from this frame
        float value = float(frames[frame_offset + pixel_idx]);
        
        // Welford's algorithm for running mean and variance
        float delta = value - mean;
        mean += delta / float(i + 1);
        float delta2 = value - mean;
        M2 += delta * delta2;
    }
    
    // Compute standard deviation
    float variance = (N > 1) ? (M2 / float(N - 1)) : 0.0f;
    float stddev = sqrt(variance);
    
    // Write result
    output[pixel_idx] = stddev;
}

// Histogram computation kernel (optional, for display)
kernel void compute_histogram(
    device float* stddev_image [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant float* bins [[buffer(2)]],
    constant uint32_t& num_bins [[buffer(3)]],
    constant uint32_t& width [[buffer(4)]],
    constant uint32_t& height [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    uint32_t pixel_idx = gid.y * width + gid.x;
    float value = stddev_image[pixel_idx];
    
    // Find appropriate bin using log scale
    float log_val = log(value + 1.0f);
    
    // Binary search would be better for many bins, but linear is fine for small counts
    for (uint32_t b = 0; b < num_bins - 1; b++) {
        if (log_val >= bins[b] && log_val < bins[b + 1]) {
            atomic_fetch_add_explicit(&histogram[b], 1, memory_order_relaxed);
            break;
        }
    }
    
    // Handle edge case for maximum value
    if (gid.x == 0 && gid.y == 0) {
        float max_bin = bins[num_bins - 1];
        if (log_val >= max_bin) {
            atomic_fetch_add_explicit(&histogram[num_bins - 1], 1, memory_order_relaxed);
        }
    }
}
