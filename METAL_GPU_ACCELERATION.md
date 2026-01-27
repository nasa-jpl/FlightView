# Metal GPU Acceleration for macOS

## Overview

**Metal** is Apple's GPU compute framework - the direct equivalent to CUDA on NVIDIA GPUs. It can provide 10-20x speedup over CPU OpenMP implementation for the standard deviation filter.

**Current Performance:**
- CUDA (Linux): ~1-2 ms
- CPU OpenMP (macOS): 33 ms
- **Metal (macOS, estimated)**: 2-5 ms

## Architecture

The code already has a clean separation:
```cpp
#ifdef USE_CUDA
    // GPU CUDA code
#else
    // CPU OpenMP code
#endif
```

We'll add:
```cpp
#ifdef USE_CUDA
    // GPU CUDA code
#elif defined(__APPLE__) && defined(USE_METAL)
    // GPU Metal code
#else
    // CPU OpenMP code
#endif
```

---

## Implementation Plan

### Phase 1: Metal Compute Shader (Core GPU Work)

Create a Metal compute shader that does the stddev calculation on GPU.

**File: `cuda_take/src/std_dev_filter.metal`**

```metal
#include <metal_stdlib>
using namespace metal;

// Compute mean and variance for standard deviation
kernel void compute_stddev(
    constant uint16_t* frames [[buffer(0)]],      // Ring buffer of N frames
    device float* output [[buffer(1)]],           // Output stddev image
    constant uint32_t& width [[buffer(2)]],
    constant uint32_t& height [[buffer(3)]],
    constant uint32_t& N [[buffer(4)]],           // Number of frames
    constant uint32_t& buffer_head [[buffer(5)]], // Ring buffer position
    uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= width || gid.y >= height) return;
    
    uint32_t pixel_idx = gid.y * width + gid.x;
    
    // Welford's online algorithm for numerical stability
    float mean = 0.0f;
    float M2 = 0.0f;
    
    // Iterate through N frames in ring buffer
    for (uint32_t i = 0; i < N; i++) {
        uint32_t frame_idx = (buffer_head + i) % GPU_FRAME_BUFFER_SIZE;
        uint32_t frame_offset = frame_idx * width * height;
        float value = float(frames[frame_offset + pixel_idx]);
        
        // Welford's algorithm
        float delta = value - mean;
        mean += delta / float(i + 1);
        float delta2 = value - mean;
        M2 += delta * delta2;
    }
    
    // Compute standard deviation
    float variance = M2 / float(N);
    float stddev = sqrt(variance);
    
    output[pixel_idx] = stddev;
}

// Compute histogram (optional, can be separate kernel)
kernel void compute_histogram(
    device float* stddev_image [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant float* bins [[buffer(2)]],
    constant uint32_t& num_bins [[buffer(3)]],
    constant uint32_t& num_pixels [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_pixels) return;
    
    float value = stddev_image[tid];
    
    // Find appropriate bin using log scale
    float log_val = log(value + 1.0f);
    
    for (uint32_t b = 0; b < num_bins - 1; b++) {
        if (log_val >= bins[b] && log_val < bins[b + 1]) {
            atomic_fetch_add_explicit(&histogram[b], 1, memory_order_relaxed);
            break;
        }
    }
}
```

### Phase 2: Metal C++ Host Code

**File: `cuda_take/src/std_dev_filter_metal.mm`** (Objective-C++)

```cpp
#ifdef __APPLE__
#ifdef USE_METAL

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include "std_dev_filter.hpp"
#include <iostream>

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> stddevPipeline;
    id<MTLComputePipelineState> histogramPipeline;
    
    id<MTLBuffer> frameBuffer;        // Ring buffer (GPU)
    id<MTLBuffer> outputBuffer;       // Stddev output (GPU)
    id<MTLBuffer> histogramBuffer;    // Histogram output (GPU)
    id<MTLBuffer> histogramBinsBuffer; // Histogram bins (GPU)
    id<MTLBuffer> paramsBuffer;       // Parameters (width, height, N, etc.)
    
    MTLSize threadsPerGrid;
    MTLSize threadsPerThreadgroup;
};

// Initialize Metal
MetalContext* init_metal(int width, int height) {
    MetalContext* ctx = new MetalContext();
    
    // Get default GPU device
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        std::cerr << "Metal: No GPU device found" << std::endl;
        delete ctx;
        return nullptr;
    }
    
    std::cout << "Metal: Using GPU: " << [ctx->device.name UTF8String] << std::endl;
    
    // Create command queue
    ctx->commandQueue = [ctx->device newCommandQueue];
    
    // Load Metal library
    NSError* error = nil;
    NSString* libraryPath = @"./std_dev_filter.metallib"; // Pre-compiled
    id<MTLLibrary> library = [ctx->device newLibraryWithFile:libraryPath error:&error];
    
    if (!library) {
        // Fall back to default library (if compiled into app)
        library = [ctx->device newDefaultLibrary];
    }
    
    if (!library) {
        std::cerr << "Metal: Failed to load shader library" << std::endl;
        if (error) {
            std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
        }
        delete ctx;
        return nullptr;
    }
    
    // Get kernel functions
    id<MTLFunction> stddevFunc = [library newFunctionWithName:@"compute_stddev"];
    id<MTLFunction> histogramFunc = [library newFunctionWithName:@"compute_histogram"];
    
    // Create pipeline states
    ctx->stddevPipeline = [ctx->device newComputePipelineStateWithFunction:stddevFunc error:&error];
    ctx->histogramPipeline = [ctx->device newComputePipelineStateWithFunction:histogramFunc error:&error];
    
    if (!ctx->stddevPipeline || !ctx->histogramPipeline) {
        std::cerr << "Metal: Failed to create pipeline state" << std::endl;
        delete ctx;
        return nullptr;
    }
    
    // Allocate GPU buffers
    size_t frameBufferSize = width * height * sizeof(uint16_t) * GPU_FRAME_BUFFER_SIZE;
    ctx->frameBuffer = [ctx->device newBufferWithLength:frameBufferSize 
                                                options:MTLResourceStorageModePrivate];
    
    size_t outputSize = width * height * sizeof(float);
    ctx->outputBuffer = [ctx->device newBufferWithLength:outputSize
                                                 options:MTLResourceStorageModeShared];
    
    ctx->histogramBuffer = [ctx->device newBufferWithLength:NUMBER_OF_BINS * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
    
    ctx->histogramBinsBuffer = [ctx->device newBufferWithLength:NUMBER_OF_BINS * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    
    // Set up thread configuration
    ctx->threadsPerGrid = MTLSizeMake(width, height, 1);
    
    // Use optimal threadgroup size
    NSUInteger maxThreadsPerGroup = ctx->stddevPipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threadgroupWidth = 16;  // Good default for most GPUs
    NSUInteger threadgroupHeight = 16;
    
    if (threadgroupWidth * threadgroupHeight > maxThreadsPerGroup) {
        threadgroupWidth = 8;
        threadgroupHeight = 8;
    }
    
    ctx->threadsPerThreadgroup = MTLSizeMake(threadgroupWidth, threadgroupHeight, 1);
    
    std::cout << "Metal: Initialized with " << width << "x" << height 
              << ", threadgroup: " << threadgroupWidth << "x" << threadgroupHeight << std::endl;
    
    return ctx;
}

// Update buffer with new frame
void metal_update_frame(MetalContext* ctx, uint16_t* frame_data, 
                       int width, int height, int buffer_head) {
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
    
    // Create blit encoder to copy frame to GPU
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    
    // Create staging buffer for this frame
    size_t frameSize = width * height * sizeof(uint16_t);
    id<MTLBuffer> stagingBuffer = [ctx->device newBufferWithBytes:frame_data
                                                           length:frameSize
                                                          options:MTLResourceStorageModeShared];
    
    // Copy to ring buffer position
    size_t bufferOffset = buffer_head * frameSize;
    [blitEncoder copyFromBuffer:stagingBuffer
                   sourceOffset:0
                       toBuffer:ctx->frameBuffer
              destinationOffset:bufferOffset
                           size:frameSize];
    
    [blitEncoder endEncoding];
    [commandBuffer commit];
}

// Compute stddev on GPU
void metal_compute_stddev(MetalContext* ctx, int width, int height, 
                         int N, int buffer_head) {
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
    
    // Create compute encoder
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline
    [encoder setComputePipelineState:ctx->stddevPipeline];
    
    // Set buffers
    [encoder setBuffer:ctx->frameBuffer offset:0 atIndex:0];
    [encoder setBuffer:ctx->outputBuffer offset:0 atIndex:1];
    [encoder setBytes:&width length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&height length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&buffer_head length:sizeof(uint32_t) atIndex:5];
    
    // Dispatch threads
    [encoder dispatchThreads:ctx->threadsPerGrid
       threadsPerThreadgroup:ctx->threadsPerThreadgroup];
    
    [encoder endEncoding];
    
    // Commit and wait (or use completion handler for async)
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// Get results
float* metal_get_stddev_result(MetalContext* ctx, int width, int height) {
    // Output buffer is MTLResourceStorageModeShared, so we can access directly
    return (float*)[ctx->outputBuffer contents];
}

#endif // USE_METAL
#endif // __APPLE__
```

### Phase 3: Integrate into std_dev_filter.hpp

**Modify `std_dev_filter.hpp`:**

```cpp
#ifdef __APPLE__
#ifdef USE_METAL
// Forward declare Metal context
struct MetalContext;
#endif
#endif

class std_dev_filter
{
private:
#ifdef USE_CUDA
    // Existing CUDA members
    uint16_t * pictures_device;
    float * picture_out_device;
    cudaStream_t std_dev_stream;
#elif defined(__APPLE__) && defined(USE_METAL)
    // Metal members
    MetalContext* metal_ctx;
    float* picture_out_metal;  // Mapped from GPU
#else
    // CPU members
    uint16_t * pictures_cpu;
    float * picture_out_cpu;
#endif
    // ... rest of class
};
```

**Modify `std_dev_filter.cpp` constructor:**

```cpp
std_dev_filter::std_dev_filter(int nWidth, int nHeight, int cudaDeviceNumber)
{
    width = nWidth;
    height = nHeight;
    gpu_buffer_head = 0;
    currentN = 0;

#ifdef USE_CUDA
    // Existing CUDA initialization
    printf("[std_dev_filter]: Using CUDA GPU\n");
    // ... CUDA code ...
    
#elif defined(__APPLE__) && defined(USE_METAL)
    // Metal initialization
    printf("[std_dev_filter]: Using Metal GPU\n");
    metal_ctx = init_metal(width, height);
    if (!metal_ctx) {
        std::cerr << "Failed to initialize Metal, falling back to CPU" << std::endl;
        // Fall through to CPU implementation
        goto cpu_impl;
    }
    
    // Get mapped output buffer
    picture_out_metal = metal_get_stddev_result(metal_ctx, width, height);
    
#else
cpu_impl:
    // Existing CPU implementation
    printf("[std_dev_filter]: Using CPU implementation with OpenMP (threads: %d)\n", 
           omp_get_max_threads());
    // ... CPU code ...
#endif
}
```

**Modify `update_GPU_buffer()`:**

```cpp
void std_dev_filter::update_GPU_buffer(frame_c * frame, unsigned int N)
{
#ifdef USE_CUDA
    // Existing CUDA code
    
#elif defined(__APPLE__) && defined(USE_METAL)
    // Metal code
    if (metal_ctx) {
        // Upload frame to GPU
        metal_update_frame(metal_ctx, frame->image_data_ptr, 
                          width, height, gpu_buffer_head);
        
        // Update buffer head
        gpu_buffer_head = (gpu_buffer_head + 1) % GPU_FRAME_BUFFER_SIZE;
        currentN = (currentN < N) ? (currentN + 1) : N;
        
        if (currentN == N) {
            // Compute stddev
            metal_compute_stddev(metal_ctx, width, height, N, gpu_buffer_head);
            
            // Copy result to frame
            memcpy(frame->std_dev_data, picture_out_metal, 
                   width * height * sizeof(float));
            frame->has_valid_std_dev = 2;
        }
        return;
    }
    // Fall through to CPU if Metal failed
    
#else
    // Existing CPU OpenMP code
    // ... CPU implementation ...
#endif
}
```

---

## Building with Metal

### Update Build System

**For qmake (.pro file):**

```qmake
# In liveview.pro or cuda_take.pro

macx {
    # Enable Metal on macOS
    DEFINES += USE_METAL
    
    # Add Metal framework
    LIBS += -framework Metal -framework MetalKit -framework Foundation
    
    # Add Metal source files (Objective-C++)
    OBJECTIVE_SOURCES += \
        cuda_take/src/std_dev_filter_metal.mm
    
    # Compile Metal shaders
    metal_compiler.commands = xcrun -sdk macosx metal \
        -c $$PWD/cuda_take/src/std_dev_filter.metal \
        -o $$PWD/cuda_take/src/std_dev_filter.air && \
        xcrun -sdk macosx metallib \
        $$PWD/cuda_take/src/std_dev_filter.air \
        -o $$PWD/std_dev_filter.metallib
    
    metal_compiler.depends = cuda_take/src/std_dev_filter.metal
    metal_compiler.target = std_dev_filter.metallib
    
    QMAKE_EXTRA_TARGETS += metal_compiler
    PRE_TARGETDEPS += std_dev_filter.metallib
}
```

### Build Commands

```bash
cd /Users/eliggett/Documents/liveview/20260126/FlightView

# Compile Metal shader first
xcrun -sdk macosx metal -c cuda_take/src/std_dev_filter.metal -o std_dev_filter.air
xcrun -sdk macosx metallib std_dev_filter.air -o std_dev_filter.metallib

# Build FlightView
qmake liveview.pro
make -j8
```

---

## Performance Expectations

### M1/M2 Mac (Unified Memory Architecture)

| Implementation | Time per Frame | Speedup vs CPU |
|----------------|----------------|----------------|
| CPU OpenMP (current) | 33,000 µs | 1x |
| Metal GPU (estimated) | 2,000-5,000 µs | **6-16x** |
| CUDA (Linux comparison) | 1,500-2,500 µs | 13-22x |

**Why Metal is fast on Apple Silicon:**
- Unified memory (no PCIe bottleneck)
- Many GPU cores (7-32 cores depending on chip)
- Optimized for Apple hardware
- Hardware-accelerated memory operations

### Intel Mac

Metal still works but performance gain is less dramatic:
- Discrete GPU: 3-8x faster than CPU
- Integrated GPU: 2-4x faster than CPU

---

## Alternative: Accelerate Framework (CPU SIMD)

If Metal implementation is too complex initially, Apple's **Accelerate framework** provides optimized CPU vector operations:

```cpp
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

void compute_stddev_accelerate(uint16_t** frames, float* output, 
                               int width, int height, int N) {
    int num_pixels = width * height;
    
    // Convert uint16 to float for all N frames
    float* frames_float = (float*)malloc(num_pixels * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        vDSP_vfltu16(frames[i], 1, &frames_float[i * num_pixels], 1, num_pixels);
    }
    
    // Compute mean
    float* mean = (float*)malloc(num_pixels * sizeof(float));
    vDSP_meanv(frames_float, N, mean, num_pixels);
    
    // Compute variance
    float* variance = (float*)malloc(num_pixels * sizeof(float));
    for (int p = 0; p < num_pixels; p++) {
        float sum_sq = 0;
        for (int i = 0; i < N; i++) {
            float diff = frames_float[i * num_pixels + p] - mean[p];
            sum_sq += diff * diff;
        }
        variance[p] = sum_sq / N;
    }
    
    // Compute sqrt for stddev
    vvsqrtf(output, variance, &num_pixels);
    
    free(frames_float);
    free(mean);
    free(variance);
}
#endif
```

**Performance**: 3-5x faster than OpenMP (SIMD + optimized math)  
**Complexity**: Low - just C function calls  
**Benefit**: Works on all Macs, easier to implement than Metal

---

## Recommended Implementation Order

### Phase 1 (Quick Win - 1-2 hours)
**Use Accelerate framework for CPU SIMD**
- Moderate speedup (3-5x)
- Easy to integrate
- No GPU required

### Phase 2 (Best Performance - 1-2 days)
**Implement Metal compute shaders**
- Best speedup (6-16x on Apple Silicon)
- Matches CUDA performance
- Uses GPU properly

### Phase 3 (Optional Optimization)
**Optimize Metal implementation**
- Async command buffers
- Metal Performance Shaders (MPS)
- Shared memory optimizations

---

## Testing

```bash
# Build with Metal
cd /Users/eliggett/Documents/liveview/20260126/FlightView
qmake liveview.pro CONFIG+=metal
make -j8

# Test performance
./liveview --headless --rtpnextgen --rtpwidth 1280 --rtpheight 328 \
  --no-gps --skipframes 3 --rtpinterface lo0 2>&1 | tee profile_metal.log

# Check stddev filter time in output
grep "stddev filter:" profile_metal.log
```

**Expected output with Metal:**
```
stddev filter: 3000 µs   (vs 33000 µs CPU)
TOTAL:         3500 µs   (vs 34000 µs CPU)
```

---

## Summary

**Best option for macOS**: **Metal GPU acceleration**
- Direct equivalent to CUDA
- 6-16x speedup on M1/M2 Macs
- Native Apple GPU support

**Easier alternative**: **Accelerate framework**
- 3-5x speedup
- CPU SIMD vectorization
- Much simpler to implement

**Both are macOS-specific** and would use:
```cpp
#ifdef __APPLE__
#ifdef USE_METAL
    // Metal GPU code
#else
    // Accelerate SIMD code
#endif
#endif
```

I can help implement either approach. Metal gives best performance but requires more work. Accelerate is a quick win with moderate improvement.

Which would you like to pursue?
