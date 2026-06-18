# Frame Skip GPU Usage Fix

## Problem
When using `--frameskip n` command-line argument, GPU usage was **increasing** instead of decreasing, particularly noticeable with NVIDIA driver 580.95.05 on Ubuntu 22.04.

## Root Cause
The original implementation skipped both frame uploads AND kernel computation when frameskipping:
- GPU processing happened only every nth frame (e.g., every 10th frame with `--frameskip 10`)
- This created **bursty GPU activity** - alternating between idle and active states
- Modern NVIDIA drivers (580.x) handle bursty workloads inefficiently:
  - GPU clock boost behavior increases clocks for short bursts
  - GPU cannot enter low-power states effectively between sporadic launches
  - Command submission overhead increases with gaps in work
  - Power consumption paradoxically increases despite less computation

## Solution
Modified the standard deviation filter to **decouple frame uploads from computation**:

1. **Always upload frames to GPU** - keeps memory transfer pipeline active
2. **Conditionally skip expensive kernel computation** - reduces actual workload
3. **Maintain smooth GPU activity** - avoids bursty power management issues

## Changes Made

### 1. `backend/include/std_dev_filter.hpp`
Added optional `skip_compute` parameter to `update_GPU_buffer()`:
```cpp
void update_GPU_buffer(frame_c *, unsigned int, bool skip_compute = false);
```

### 2. `backend/src/std_dev_filter.cpp`
- Modified `update_GPU_buffer()` to accept `skip_compute` parameter
- Frame uploads to GPU ring buffer happen every frame (CUDA, Metal, and CPU paths)
- Kernel computation (expensive part) is conditionally skipped based on parameter
- Applies to all implementation paths: CUDA, Metal (macOS), and CPU fallback

### 3. `backend/src/acquire.cpp`
Modified RTP consumer loop (`rtp_consumer_loop()`):
- Moved `update_GPU_buffer()` call **outside** the frameskip conditional
- Always calls `update_GPU_buffer()` to maintain GPU pipeline
- Passes `skip_stddev_compute = (count % frameSkip != 0)` when frameskip is enabled
- Other filter operations (dark subtraction, white reference, mean) remain inside conditional

## Behavior

### Without `--frameskip` (unchanged):
- Every frame uploaded to GPU ✓
- Every frame computed ✓
- Smooth continuous GPU activity

### With `--frameskip 10` (improved):
- Every frame uploaded to GPU ✓ (new)
- Only every 10th frame computed ✓
- Maintains smooth GPU pipeline activity
- Reduces actual computation workload
- Avoids bursty power management issues

## Expected Results
- **Reduced GPU utilization** when using frameskip (as originally intended)
- **Lower power consumption** due to avoiding burst/idle cycles
- **Maintained functionality** - stddev still computed with temporal sampling
- **Compatible** with CUDA, Metal (macOS), and CPU implementations

## Testing Recommendations
1. Monitor GPU utilization with `nvidia-smi dmon` or `nvtop`
2. Compare power draw with and without `--frameskip`
3. Verify stddev results are computed correctly (every nth frame)
4. Test with various frameskip values (2, 5, 10, 20)

## Notes
- Frame uploads (memcpy) are relatively cheap (~microseconds)
- Kernel computation is expensive (~milliseconds)
- Keeping GPU "warm" with continuous uploads reduces driver overhead
- This approach is compatible with modern GPU power management strategies
