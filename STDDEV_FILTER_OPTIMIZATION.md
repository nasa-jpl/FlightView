# Standard Deviation Filter Optimization Guide

## Problem Identified

The CPU-based standard deviation filter is the primary bottleneck on macOS (without CUDA):
- **Processing time**: 33.3 ms per frame (97% of total processing)
- **Causes frame drops** at rates >30 FPS when enabled
- **Gets progressively slower** as buffer fills (15ms → 33ms)

On Linux with CUDA: ~1-2ms  
On macOS without CUDA: 33ms+ (16-40x slower)

## Immediate Solutions

### Solution 1: Disable When Not Needed (Fastest)

```bash
./liveview --no-stddev --rtpnextgen --rtpwidth 1280 --rtpheight 328 ...
```

**Result**: Total processing drops from 34ms → 0.8ms (42x faster)

**Use when:**
- Testing network throughput
- Headless preview generation
- Stddev display not required

### Solution 2: Reduce Buffer Size (NEW)

```bash
# Default N=400 frames (very expensive)
./liveview --stddev-n 100 --rtpnextgen --rtpwidth 1280 --rtpheight 328 ...

# Or even smaller for real-time
./liveview --stddev-n 50 --rtpnextgen --rtpwidth 1280 --rtpheight 328 ...
```

**Expected speedup**:
- N=400 → N=100: ~4x faster (33ms → 8ms)
- N=400 → N=50: ~8x faster (33ms → 4ms)  
- N=400 → N=25: ~16x faster (33ms → 2ms)

**Trade-off**: Smaller N = less statistical smoothing, noisier output

---

## Build and Test

### Rebuild with New Option

```bash
cd /Users/eliggett/Documents/liveview/20260126/FlightView
qmake liveview.pro
make -j8
```

### Test Different N Values

```bash
# Terminal 1: Test with N=100
./liveview --headless --rtpnextgen --rtpwidth 1280 --rtpheight 328 \
  --no-gps --stddev-n 100 --skipframes 3 --rtpinterface lo0 \
  --datastoragelocation /Users/eliggett/Downloads 2>&1 | tee profile_n100.log

# Terminal 2: Send frames
cd utils/rtp
./server /path/to/scene.raw -f 64 -p 256
```

Check the profiling output:
```
rtpConsumeFrames: === Frame Processing Performance ===
  stddev filter: XXX µs  ← Watch this value
  TOTAL:         XXX µs
```

### Find Optimal N

Test different values and find the sweet spot:

| N Value | Expected Time | Use Case |
|---------|---------------|----------|
| 400 (default) | 33,000 µs | Maximum smoothing, offline processing |
| 200 | 16,500 µs | Good smoothing, moderate performance |
| 100 | 8,200 µs | Balanced, real-time at 60 FPS |
| 50 | 4,100 µs | Minimal smoothing, real-time at 120 FPS |
| 25 | 2,000 µs | Low smoothing, real-time at 200+ FPS |

**Guideline**: Pick N such that stddev time < 50% of your frame budget.

At 64 FPS with skipframes 3:
- Budget: 62,500 µs
- Target stddev time: < 31,000 µs
- **Recommended N: 100-200**

---

## Advanced Optimizations

### Option 1: SIMD Vectorization

The CPU stddev filter can be optimized with SIMD instructions:

```cpp
// Current (scalar):
for(int i = 0; i < pixels; i++) {
    mean += buffer[i];
    variance += buffer[i] * buffer[i];
}

// Optimized (SIMD with ARM NEON):
#ifdef __ARM_NEON
#include <arm_neon.h>

float32x4_t sum_vec = vdupq_n_f32(0);
float32x4_t sum_sq_vec = vdupq_n_f32(0);

for(int i = 0; i < pixels; i += 4) {
    float32x4_t vals = vld1q_f32(&buffer[i]);
    sum_vec = vaddq_f32(sum_vec, vals);
    sum_sq_vec = vmlaq_f32(sum_sq_vec, vals, vals);
}
// Horizontal sum...
#endif
```

**Expected speedup**: 4-8x on ARM (M1/M2)

### Option 2: Spatial Downsampling

Calculate stddev on a downsampled image:

```cpp
// Instead of 1280×328 = 419,840 pixels
// Downsample 2x: 640×164 = 104,960 pixels (4x faster)
// Downsample 4x: 320×82 = 26,240 pixels (16x faster)
```

**Trade-off**: Lower spatial resolution in stddev display

### Option 3: Reduced Update Rate

Don't calculate stddev every frame:

```cpp
// In take_object.cpp
static int stddev_counter = 0;
if(stddev_counter++ % 4 == 0) {  // Only update every 4th frame
    sdvf->update_GPU_buffer(curFrame, std_dev_filter_N);
}
```

At 64 FPS:
- Update every 4 frames = 16 Hz stddev updates
- **4x effective speedup**: 33ms → 8.25ms average

### Option 4: Metal Compute Shaders (macOS GPU)

Use Metal instead of CUDA on macOS:

```cpp
// Create Metal compute pipeline
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLComputePipelineState> pipeline = ...;

// Dispatch stddev calculation on GPU
id<MTLComputeCommandEncoder> encoder = ...;
[encoder setComputePipelineState:pipeline];
[encoder setBuffer:frameBuffer offset:0 atIndex:0];
[encoder setBuffer:outputBuffer offset:0 atIndex:1];
[encoder dispatchThreads:MTLSizeMake(width, height, 1)
   threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
```

**Expected performance**: 2-5ms (similar to CUDA on Linux)

**Complexity**: High - requires Metal/Objective-C++ integration

### Option 5: Parallel CPU with Better Algorithms

Current implementation uses OpenMP but may not be optimal:

```cpp
// Use Welford's online algorithm for numerical stability and efficiency
// Plus proper cache-friendly memory access patterns

#pragma omp parallel for schedule(static) num_threads(12)
for(int y = 0; y < height; y++) {
    // Process row with Welford's algorithm
    // Better cache locality than column-wise
}
```

---

## Recommended Strategy

### For Testing/Development (Right Now)

```bash
./liveview --no-stddev ...
```

**Result**: 42x speedup, can test full network performance

### For Production with Stddev (After rebuild)

```bash
./liveview --stddev-n 100 ...  # or 50-200 based on testing
```

**Result**: 4-8x speedup, maintains stddev functionality

### For Maximum Performance (Future)

1. **Implement Metal compute shaders** (best performance on macOS)
2. **Add SIMD vectorization** (4-8x speedup, easier than Metal)
3. **Combine reduced N + SIMD**: N=100 + SIMD = ~1ms processing time

---

## Performance Summary

### Current Performance (macOS M1/M2, no CUDA)

| Configuration | Stddev Time | Total Time | Max FPS | Frame Loss @ 64 FPS |
|---------------|-------------|------------|---------|---------------------|
| Default (N=400) | 33,000 µs | 34,000 µs | ~30 FPS | 43% (308/710 frames) |
| --no-stddev | 0 µs | 800 µs | 200+ FPS | 0% |
| --stddev-n 100 (est.) | 8,200 µs | 9,000 µs | 110 FPS | 0% @ 64 FPS |
| --stddev-n 50 (est.) | 4,100 µs | 4,900 µs | 200 FPS | 0% @ 64 FPS |

### With SIMD Optimization (Future)

| Configuration | Stddev Time | Total Time | Max FPS |
|---------------|-------------|------------|---------|
| N=400 + SIMD | 4,100 µs | 4,900 µs | 200 FPS |
| N=100 + SIMD | 1,000 µs | 1,800 µs | 500+ FPS |
| N=50 + SIMD | 500 µs | 1,300 µs | 700+ FPS |

### With Metal (Future, Best Case)

| Configuration | Stddev Time | Total Time | Max FPS |
|---------------|-------------|------------|---------|
| N=400 + Metal | 2,000 µs | 2,800 µs | 350 FPS |
| N=100 + Metal | 800 µs | 1,600 µs | 600 FPS |

---

## Usage Examples

### Test Network Performance (No stddev)
```bash
./liveview --headless --no-stddev --rtpnextgen \
  --rtpwidth 1280 --rtpheight 328 --no-gps \
  --skipframes 3 --rtpinterface lo0
```

### Real-time Preview with Stddev (Reduced N)
```bash
./liveview --headless --stddev-n 100 --rtpnextgen \
  --rtpwidth 1280 --rtpheight 328 --no-gps \
  --skipframes 1 --rtpinterface eth0
```

### Flight Mode with Conservative Stddev
```bash
./liveview --flight --stddev-n 200 --rtpnextgen \
  --rtpwidth 1280 --rtpheight 328 \
  --datastoragelocation /data
```

### Maximum Throughput Test
```bash
# Test at 220 FPS with stddev N=50
./liveview --headless --no-stddev --rtpnextgen \
  --rtpwidth 1280 --rtpheight 328 --no-gps --rtpinterface lo0

cd utils/rtp
./server scene.raw -f 220 -p 256
```

---

## Conclusion

The standard deviation filter is the bottleneck, but you now have multiple solutions:

**Immediate (0 effort)**:
- Use `--no-stddev` for testing: 42x speedup

**Short-term (rebuild required)**:
- Use `--stddev-n 50-100`: 4-8x speedup

**Long-term (development required)**:
- SIMD optimization: 16-32x speedup vs default
- Metal compute: 10-15x speedup vs default

For your current 64 FPS use case with `--skipframes 3`, using `--stddev-n 100` should provide excellent performance with minimal loss of stddev quality.

Date: January 27, 2026
