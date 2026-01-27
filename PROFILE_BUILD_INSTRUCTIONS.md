# Build and Profile Instructions

## Build with Profiling Instrumentation

The code has been instrumented with detailed timing measurements in `take_object.cpp`.

### Build Steps

```bash
cd /Users/eliggett/Documents/liveview/20260126/FlightView

# Generate Makefile from Qt project
qmake liveview.pro

# Build with optimizations
make -j8

# Or rebuild from scratch
make clean
qmake liveview.pro
make -j8
```

### Run Profiling Test

```bash
# Terminal 1: Start FlightView with profiling
./liveview --headless --rtpnextgen --rtpwidth 1280 --rtpheight 328 --no-gps --skipframes 3 \
  --rtpinterface lo0 --datastoragelocation /Users/eliggett/Downloads 2>&1 | tee profile_output.log

# Terminal 2: Send frames at 64 FPS
cd utils/rtp
./server /Users/eliggett/Documents/AVIRIS-III/Data/20230627/chlus/scene.raw -f 64 -p 256
```

### What to Look For

The instrumented code will print performance reports every 100 processed frames:

```
=== Frame Processing Performance (avg over 100 frames) ===
  getFrameWait:  XXX µs
  memcpy (RTP):  XXX µs
  2s complement: XXX µs (if enabled)
  inversion:     XXX µs (if enabled)
  shm copy:      XXX µs (if enabled)
  stddev filter: XXX µs (if enabled)
  dark subtract: XXX µs
  white ref:     XXX µs
  mean filter:   XXX µs
  TOTAL:         XXX µs (X.XX ms)
```

### Expected Budget

With `--skipframes 3` at 64 FPS:
- **Processing rate**: 16 frames/sec
- **Available time**: 62.5 ms per processed frame (62,500 µs)
- **If TOTAL > 62,500 µs**: That operation is the bottleneck

### Interpreting Results

**Good performance:**
- getFrameWait: < 1,000 µs
- memcpy (RTP): < 1,500 µs (839 KB)
- shm copy: < 1,500 µs (if enabled)
- dark subtract: < 3,000 µs
- white ref: < 3,000 µs
- mean filter: < 5,000 µs
- **TOTAL: < 15,000 µs (15 ms)**

**If any operation is much higher:**
- That's your bottleneck
- Check if memory bandwidth is saturated
- Check if CPU is thermal throttling
- Check if other processes are competing

## Alternative: Xcode Instruments Profiling

For detailed CPU profiling:

```bash
# Build in Release mode
qmake liveview.pro CONFIG+=release
make -j8

# Profile with Instruments
instruments -t "Time Profiler" -D profile_64fps.trace \
  ./liveview --headless --rtpnextgen --rtpwidth 1280 --rtpheight 328 \
  --no-gps --skipframes 3 --rtpinterface lo0 &

# In another terminal, run server
cd utils/rtp
./server /path/to/scene.raw -f 64 -p 256

# Let run for 30-60 seconds
sleep 60

# Stop profiling
killall liveview

# Open trace in Instruments
open profile_64fps.trace
```

Look for:
- Hottest functions in `take_object::rtpConsumeFrames`
- CPU usage per thread
- Memory allocations
- System calls

## Quick Checks

### 1. Check CPU Usage
```bash
# While running test
top -pid $(pgrep liveview) -stats pid,cpu,mem,threads,time
```

**Expected:**
- CPU < 100% for single thread (should have plenty of headroom)
- If near 100%: processing is CPU-bound

### 2. Check Memory Bandwidth
```bash
# Monitor memory pressure
memory_pressure
```

**If "Critical":** Memory bandwidth is saturated

### 3. Check Thermal Throttling
```bash
# M1/M2 Macs
sudo powermetrics --samplers smc -n 1 | grep -i temp

# Intel Macs
sudo powermetrics --samplers cpu_power -n 1 | grep -i temp
```

**If > 90°C:** CPU may be throttling

## Troubleshooting

### Issue: Build fails

```bash
# Check Qt installation
which qmake
qmake --version

# If not found, install Qt
brew install qt
```

### Issue: Cannot find symbols

```bash
# Clean and rebuild
make clean
rm -f Makefile*
qmake liveview.pro
make -j8
```

### Issue: Profiling output not appearing

Check that `LOG` macro is working:
```bash
./liveview --version
# Should show compile info
```

If no output, logging may be disabled. Try:
```bash
export QT_LOGGING_RULES="*=true"
./liveview ...
```

## What We're Testing

The instrumentation will reveal which operation(s) are taking longer than expected:

1. **getFrameWait** - RTP buffer access (should be fast)
2. **memcpy** - Memory bandwidth (should be ~1ms for 839KB)
3. **Dark/White filters** - CPU computation (expect 2-4ms each)
4. **Mean filter** - CPU + possibly FFT (expect 3-8ms)

**Total should be < 15ms** to handle 16 FPS processing rate comfortably.

If total is 30-60ms, that explains why buffer fills up - you're only processing ~16-33 frames/sec instead of the needed 16 fps.

Date: January 27, 2026
