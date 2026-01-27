// Metal GPU acceleration for std_dev_filter - macOS ONLY
// This file provides GPU compute functionality equivalent to CUDA on Linux

#ifdef __APPLE__
#ifdef USE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "std_dev_filter_metal.h"
#include "constants.h"
#include <iostream>
#include <cstring>

struct MetalStdDevContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> stddevPipeline;
    id<MTLComputePipelineState> histogramPipeline;
    
    // GPU buffers
    id<MTLBuffer> frameRingBuffer;     // Ring buffer of frames (GPU memory)
    id<MTLBuffer> outputBuffer;        // Stddev output (shared with CPU)
    id<MTLBuffer> histogramBuffer;     // Histogram output (shared with CPU)
    id<MTLBuffer> histogramBinsBuffer; // Histogram bins (GPU)
    
    // Thread configuration
    MTLSize threadsPerGrid;
    MTLSize threadsPerThreadgroup;
    
    // Parameters
    uint32_t width;
    uint32_t height;
    uint32_t buffer_head;
    
    // Mapped pointers for CPU access
    float* output_ptr;
    uint32_t* histogram_ptr;
};

void* metal_stddev_init(int width, int height) {
    @autoreleasepool {
        MetalStdDevContext* ctx = new MetalStdDevContext();
        
        ctx->width = width;
        ctx->height = height;
        ctx->buffer_head = 0;
        
        // Get default GPU device
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            std::cerr << "[std_dev_filter]: Metal - No GPU device found" << std::endl;
            delete ctx;
            return nullptr;
        }
        
        std::cout << "[std_dev_filter]: Metal GPU initialized: " 
                  << [ctx->device.name UTF8String] << std::endl;
        
        // Create command queue
        ctx->commandQueue = [ctx->device newCommandQueue];
        if (!ctx->commandQueue) {
            std::cerr << "[std_dev_filter]: Metal - Failed to create command queue" << std::endl;
            delete ctx;
            return nullptr;
        }
        
        // Load Metal shader library
        NSError* error = nil;
        
        // Try to load pre-compiled metallib
        NSString* exePath = [[NSBundle mainBundle] executablePath];
        NSString* exeDir = [exePath stringByDeletingLastPathComponent];
        NSString* metallibPath = [exeDir stringByAppendingPathComponent:@"std_dev_filter.metallib"];
        
        // Use newLibraryWithURL (macOS 10.13+) instead of deprecated newLibraryWithFile
        NSURL* metallibURL = [NSURL fileURLWithPath:metallibPath];
        id<MTLLibrary> library = [ctx->device newLibraryWithURL:metallibURL error:&error];
        
        if (!library) {
            // Try default library (if compiled into executable)
            library = [ctx->device newDefaultLibrary];
        }
        
        if (!library) {
            std::cerr << "[std_dev_filter]: Metal - Failed to load shader library" << std::endl;
            if (error) {
                std::cerr << "  Error: " << [[error localizedDescription] UTF8String] << std::endl;
            }
            std::cerr << "  Searched at: " << [metallibPath UTF8String] << std::endl;
            delete ctx;
            return nullptr;
        }
        
        // Get kernel functions
        id<MTLFunction> stddevFunc = [library newFunctionWithName:@"compute_stddev"];
        id<MTLFunction> histogramFunc = [library newFunctionWithName:@"compute_histogram"];
        
        if (!stddevFunc) {
            std::cerr << "[std_dev_filter]: Metal - Failed to find compute_stddev kernel" << std::endl;
            delete ctx;
            return nullptr;
        }
        
        // Create pipeline states
        ctx->stddevPipeline = [ctx->device newComputePipelineStateWithFunction:stddevFunc error:&error];
        if (!ctx->stddevPipeline) {
            std::cerr << "[std_dev_filter]: Metal - Failed to create stddev pipeline" << std::endl;
            if (error) {
                std::cerr << "  Error: " << [[error localizedDescription] UTF8String] << std::endl;
            }
            delete ctx;
            return nullptr;
        }
        
        if (histogramFunc) {
            ctx->histogramPipeline = [ctx->device newComputePipelineStateWithFunction:histogramFunc 
                                                                                 error:&error];
        }
        
        // Allocate GPU buffers
        size_t frameBufferSize = width * height * sizeof(uint16_t) * GPU_FRAME_BUFFER_SIZE;
        ctx->frameRingBuffer = [ctx->device newBufferWithLength:frameBufferSize 
                                                        options:MTLResourceStorageModePrivate];
        
        if (!ctx->frameRingBuffer) {
            std::cerr << "[std_dev_filter]: Metal - Failed to allocate frame ring buffer ("
                      << (frameBufferSize / (1024*1024)) << " MB)" << std::endl;
            delete ctx;
            return nullptr;
        }
        
        size_t outputSize = width * height * sizeof(float);
        ctx->outputBuffer = [ctx->device newBufferWithLength:outputSize
                                                     options:MTLResourceStorageModeShared];
        
        ctx->histogramBuffer = [ctx->device newBufferWithLength:NUMBER_OF_BINS * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
        
        ctx->histogramBinsBuffer = [ctx->device newBufferWithLength:NUMBER_OF_BINS * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        if (!ctx->outputBuffer || !ctx->histogramBuffer || !ctx->histogramBinsBuffer) {
            std::cerr << "[std_dev_filter]: Metal - Failed to allocate output buffers" << std::endl;
            delete ctx;
            return nullptr;
        }
        
        // Get mapped pointers for CPU access (shared memory)
        ctx->output_ptr = (float*)[ctx->outputBuffer contents];
        ctx->histogram_ptr = (uint32_t*)[ctx->histogramBuffer contents];
        
        // Set up thread configuration
        ctx->threadsPerGrid = MTLSizeMake(width, height, 1);
        
        // Determine optimal threadgroup size
        NSUInteger maxThreadsPerGroup = ctx->stddevPipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger threadgroupWidth = 16;
        NSUInteger threadgroupHeight = 16;
        
        // Adjust if too large
        while (threadgroupWidth * threadgroupHeight > maxThreadsPerGroup) {
            threadgroupWidth /= 2;
            threadgroupHeight /= 2;
        }
        
        // Make sure threadgroup size divides image dimensions well
        while (width % threadgroupWidth != 0 && threadgroupWidth > 1) {
            threadgroupWidth--;
        }
        while (height % threadgroupHeight != 0 && threadgroupHeight > 1) {
            threadgroupHeight--;
        }
        
        ctx->threadsPerThreadgroup = MTLSizeMake(threadgroupWidth, threadgroupHeight, 1);
        
        std::cout << "[std_dev_filter]: Metal - Image: " << width << "x" << height 
                  << ", Threadgroup: " << threadgroupWidth << "x" << threadgroupHeight
                  << ", Max threads/group: " << maxThreadsPerGroup << std::endl;
        
        return (void*)ctx;
    }
}

void metal_stddev_cleanup(void* context) {
    if (!context) return;
    
    @autoreleasepool {
        MetalStdDevContext* ctx = (MetalStdDevContext*)context;
        
        // Metal objects are reference counted (ARC), so just delete the struct
        // The Objective-C objects will be released automatically
        delete ctx;
    }
}

void metal_stddev_update_frame(void* context, const uint16_t* frame_data) {
    if (!context || !frame_data) return;
    
    @autoreleasepool {
        MetalStdDevContext* ctx = (MetalStdDevContext*)context;
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        commandBuffer.label = @"Frame Upload";
        
        // Create blit encoder for efficient memory copy
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        
        // Create staging buffer with frame data (shared memory for CPU write)
        size_t frameSize = ctx->width * ctx->height * sizeof(uint16_t);
        id<MTLBuffer> stagingBuffer = [ctx->device newBufferWithBytes:frame_data
                                                               length:frameSize
                                                              options:MTLResourceStorageModeShared];
        
        // Copy to appropriate position in ring buffer
        size_t ringBufferOffset = ctx->buffer_head * frameSize;
        [blitEncoder copyFromBuffer:stagingBuffer
                       sourceOffset:0
                           toBuffer:ctx->frameRingBuffer
                  destinationOffset:ringBufferOffset
                               size:frameSize];
        
        [blitEncoder endEncoding];
        [commandBuffer commit];
        
        // Update ring buffer position
        ctx->buffer_head = (ctx->buffer_head + 1) % GPU_FRAME_BUFFER_SIZE;
    }
}

void metal_stddev_compute(void* context, uint32_t N, bool compute_histogram, 
                         const float* histogram_bins) {
    if (!context) return;
    
    @autoreleasepool {
        MetalStdDevContext* ctx = (MetalStdDevContext*)context;
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        commandBuffer.label = @"Compute StdDev";
        
        // Create compute encoder
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        encoder.label = @"StdDev Kernel";
        
        // Set pipeline
        [encoder setComputePipelineState:ctx->stddevPipeline];
        
        // Set buffers and parameters
        uint32_t buffer_size = GPU_FRAME_BUFFER_SIZE;
        [encoder setBuffer:ctx->frameRingBuffer offset:0 atIndex:0];
        [encoder setBuffer:ctx->outputBuffer offset:0 atIndex:1];
        [encoder setBytes:&ctx->width length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&ctx->height length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&buffer_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&ctx->buffer_head length:sizeof(uint32_t) atIndex:6];
        
        // Dispatch threads
        [encoder dispatchThreads:ctx->threadsPerGrid
           threadsPerThreadgroup:ctx->threadsPerThreadgroup];
        
        [encoder endEncoding];
        
        // Optionally compute histogram
        if (compute_histogram && ctx->histogramPipeline && histogram_bins) {
            // Clear histogram
            memset(ctx->histogram_ptr, 0, NUMBER_OF_BINS * sizeof(uint32_t));
            
            // Upload histogram bins
            memcpy([ctx->histogramBinsBuffer contents], histogram_bins, 
                   NUMBER_OF_BINS * sizeof(float));
            
            id<MTLComputeCommandEncoder> histEncoder = [commandBuffer computeCommandEncoder];
            histEncoder.label = @"Histogram Kernel";
            
            [histEncoder setComputePipelineState:ctx->histogramPipeline];
            [histEncoder setBuffer:ctx->outputBuffer offset:0 atIndex:0];
            [histEncoder setBuffer:ctx->histogramBuffer offset:0 atIndex:1];
            [histEncoder setBuffer:ctx->histogramBinsBuffer offset:0 atIndex:2];
            uint32_t num_bins = NUMBER_OF_BINS;
            [histEncoder setBytes:&num_bins length:sizeof(uint32_t) atIndex:3];
            [histEncoder setBytes:&ctx->width length:sizeof(uint32_t) atIndex:4];
            [histEncoder setBytes:&ctx->height length:sizeof(uint32_t) atIndex:5];
            
            [histEncoder dispatchThreads:ctx->threadsPerGrid
                   threadsPerThreadgroup:ctx->threadsPerThreadgroup];
            
            [histEncoder endEncoding];
        }
        
        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

float* metal_stddev_get_output(void* context) {
    if (!context) return nullptr;
    
    MetalStdDevContext* ctx = (MetalStdDevContext*)context;
    return ctx->output_ptr;
}

uint32_t* metal_stddev_get_histogram(void* context) {
    if (!context) return nullptr;
    
    MetalStdDevContext* ctx = (MetalStdDevContext*)context;
    return ctx->histogram_ptr;
}

#endif // USE_METAL
#endif // __APPLE__
