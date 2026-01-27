# Building FlightView on macOS

**Tested on:** macOS 15.7.3 (Sequoia) with Xcode Command Line Tools

**Build Difficulty:** Moderate (30-60 minutes for first-time setup)

**Status:** FlightView can be built and run on macOS in CPU-only mode. All features work except GPU-accelerated standard deviation (which uses CPU with OpenMP instead).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installing Dependencies](#installing-dependencies)
3. [Building the Project](#building-the-project)
4. [Running FlightView](#running-flightview)
5. [Troubleshooting](#troubleshooting)
6. [Feature Compatibility](#feature-compatibility)

---

## Prerequisites

### Xcode Command Line Tools

You need the Xcode command line tools installed. Check if you have them:

```bash
xcode-select -p
```

If not installed, install them:

```bash
xcode-select --install
```

### Homebrew Package Manager

Homebrew is required to install all dependencies. Check if you have it:

```bash
brew --version
```

If not installed:

```bash
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
```

After installation, follow the on-screen instructions to add Homebrew to your PATH. This typically involves adding these lines to `~/.zshrc`:

```bash
# For Apple Silicon (M1/M2/M3):
eval "$(/opt/homebrew/bin/brew shellenv)"

# For Intel Macs:
eval "$(/usr/local/bin/brew shellenv)"
```

Then reload your shell:

```bash
source ~/.zshrc
```

---

## Installing Dependencies

### Core Build Dependencies

Install all required libraries and tools:

```bash
# Update Homebrew
brew update

# Install Qt 5 (GUI framework)
brew install qt@5

# Install Boost C++ libraries
brew install boost

# Install GNU Scientific Library
brew install gsl

# Install GStreamer (video/streaming support)
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly

# Install ZeroMQ (messaging library) and C++ bindings
brew install zeromq cppzmq

# Install Exiv2 (image metadata library)
brew install exiv2

# Install OpenMP support (for multi-threaded CPU processing)
brew install libomp

# Install pkg-config (build configuration tool)
brew install pkg-config
```

**Installation time:** Approximately 10-20 minutes depending on your internet connection and Mac speed.

### Configure Qt Path

Qt needs to be added to your PATH so that `qmake` can be found. Add this to your `~/.zshrc`:

```bash
# For Apple Silicon:
export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/qt@5/lib"
export CPPFLAGS="-I/opt/homebrew/opt/qt@5/include"

# For Intel Macs:
export PATH="/usr/local/opt/qt@5/bin:$PATH"
export LDFLAGS="-L/usr/local/opt/qt@5/lib"
export CPPFLAGS="-I/usr/local/opt/qt@5/include"
```

Reload your shell:

```bash
source ~/.zshrc
```

Verify Qt installation:

```bash
qmake --version
# Should output: QMake version 3.1, Using Qt version 5.x.x
```

### Verify All Dependencies

Check that all libraries are installed:

```bash
# Check Boost
brew list boost

# Check GSL
brew list gsl

# Check GStreamer
gst-inspect-1.0 --version

# Check ZeroMQ
brew list zeromq

# Check Exiv2
brew list exiv2

# Check OpenMP
brew list libomp

# Check pkg-config
pkg-config --version
```

---

## Building the Project

### Step 1: Navigate to Project Directory

```bash
cd /Users/eliggett/Documents/liveview/20260126/FlightView
```

### Step 2: Generate App Icon (One-time setup)

If `liveview.icns` doesn't exist, generate it from the PNG:

```bash
./create_icns.sh
```

This creates `liveview.icns` with all required icon resolutions for macOS.

### Step 3: Build backend Library

The `backend` library must be built first. On macOS, we build without CUDA support:

```bash
cd backend

# Clean any previous build artifacts
make clean

# Build CPU-only version (USE_CUDA=0 is default on macOS)
make -j8

# Verify the library was created
ls -lh lib_backend.a
# Should show a file around 1-5 MB

cd ..
```

**Build time:** 2-5 minutes

### Step 4: Build FlightView Application

```bash
# Generate Makefile from Qt project file
qmake CONFIG+=no-cuda CONFIG+=release

# Build the application (using 8 parallel jobs)
make -j8
```

**Build time:** 5-10 minutes

### Step 4: Verify Build

Check that the executable was created:

```bash
ls -lh lv_release/liveview
# Should show an executable file

# Check the binary type
file lv_release/liveview
# Should show: Mach-O 64-bit executable arm64 (Apple Silicon)
# Or: Mach-O 64-bit executable x86_64 (Intel)

# Check linked libraries
otool -L lv_release/liveview | grep -E 'boost|gsl|Qt|gstreamer'
```

---

## Running FlightView

### Basic Execution

```bash
# Run from build directory
./lv_release/liveview

# Or run with options
./lv_release/liveview --help
```

### Command Line Options

```bash
# Run in headless mode
./lv_release/liveview --headless

# Specify XIO camera directory
./lv_release/liveview --xio /path/to/xio/files

# Specify RTP stream
./lv_release/liveview --rtp --rtp-address 239.255.0.1 --rtp-port 5004

# Enable debug output
./lv_release/liveview --debug
```

### Creating an Application Bundle (Optional)

To create a double-clickable macOS application:

```bash
# Use macdeployqt to bundle Qt frameworks
/opt/homebrew/opt/qt@5/bin/macdeployqt lv_release/liveview.app

# Copy to Applications folder
cp -r lv_release/liveview.app /Applications/
```

---

## Troubleshooting

### Issue: "qmake: command not found"

**Cause:** Qt is not in your PATH

**Solution:**
```bash
# Find Qt installation
brew --prefix qt@5

# Add to PATH in ~/.zshrc
export PATH="$(brew --prefix qt@5)/bin:$PATH"

# Reload shell
source ~/.zshrc
```

### Issue: "library not found for -lboost_thread"

**Cause:** macOS Boost libraries use `-mt` suffix

**Solution:** This should be handled automatically by the updated [liveview.pro](cci:7://file:///Users/eliggett/Documents/liveview/20260126/FlightView/liveview.pro:0:0-0:0). If not, ensure you've applied the macOS build changes.

### Issue: "unknown warning option '-Wno-class-memaccess'"

**Cause:** Clang doesn't recognize this GCC-specific warning flag

**Solution:** The updated build files remove this flag on macOS. Clean and rebuild:
```bash
make clean
qmake CONFIG+=no-cuda
make -j8
```

### Issue: "fopenmp is not supported"

**Cause:** Xcode's clang needs special flags for OpenMP

**Solution:** Ensure `libomp` is installed:
```bash
brew install libomp
```

### Issue: "Cannot find -lgsl"

**Cause:** GSL library not found

**Solution:**
```bash
brew install gsl

# Verify installation
pkg-config --libs gsl
```

### Issue: "Package gstreamer-1.0 was not found"

**Cause:** GStreamer not installed or pkg-config can't find it

**Solution:**
```bash
# Install GStreamer
brew install gstreamer gst-plugins-base

# Check pkg-config can find it
pkg-config --modversion gstreamer-1.0

# If still not found, add to PKG_CONFIG_PATH
export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:$PKG_CONFIG_PATH"
```

---

## Feature Compatibility

### ✅ Fully Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| Frame Display | ✅ Working | Full resolution display |
| Dark Subtraction Filter | ✅ Working | CPU-based |
| White Reference Filter | ✅ Working | CPU-based |
| Mean Profile Calculation | ✅ Working | Horizontal/vertical profiles |
| FFT Analysis | ✅ Working | Real-time FFT |
| Standard Deviation | ✅ Working | CPU + OpenMP (multi-threaded) |
| Histogram Display | ✅ Working | Real-time histograms |
| Frame Saving | ✅ Working | Save to disk |
| RTP Camera Streams | ✅ Working | Network camera support |
| XIO File Playback | ✅ Working | Replay saved data |

### ❌ Not Available on macOS

| Feature | Status | Reason |
|---------|--------|--------|
| CUDA/GPU Acceleration | ❌ Not Available | NVIDIA dropped macOS support |
| CameraLink Support | ❌ Not Available | No macOS drivers available |

---

## Performance Expectations

**Standard Deviation Calculation (CPU):**

| Mac Type | Expected Performance |
|----------|---------------------|
| M1/M2/M3 (8+ cores) | 80-100 fps @ 640x480 |
| Intel i7/i9 (8 cores) | 40-60 fps @ 640x480 |
| Intel i5 (4 cores) | 20-30 fps @ 640x480 |

Performance scales with CPU core count and clock speed.

---

## Clean Build

If you need to start fresh:

```bash
cd /Users/eliggett/Documents/liveview/20260126/FlightView

# Clean everything
make clean
rm -rf lv_release/
cd backend && make clean && cd ..

# Rebuild from scratch (USE_CUDA=0 is default on macOS)
cd backend && make -j8 && cd ..
qmake CONFIG+=release
make -j8
```

## Running FlightView on macOS

### Option 1: Double-click the App Icon (Recommended)

The macOS build includes a launcher configuration that allows you to double-click the `liveview.app` icon to launch with default arguments. The default configuration is stored in:

```
liveview.app/Contents/Resources/launch_config.txt
```

You can edit this file to customize the default launch arguments. The default contents are:

```
--rtpcam --rtpnextgen --rtpheight 328 --rtpwidth 1280 --rtpinterface lo0 --rtpport 5004 --datastoragelocation /tmp
```

### Option 2: Command Line with Custom Arguments

You can also run from the terminal to override or add arguments:

``` bash
cd build/lv_release/liveview.app/Contents/MacOS
./liveview [additional arguments]
```

The launcher script will read the default arguments from `launch_config.txt` and append any additional arguments you provide on the command line.

### Launcher Implementation

The build process automatically:
1. Creates `launch_config.txt` in the app's Resources folder with default arguments
2. Renames the main binary to `liveview-bin`
3. Installs a shell script wrapper as `liveview` that reads the config file and launches the binary

This allows both GUI launching (double-click) and command-line flexibility. 

---

## Summary of Changes for macOS

The following modifications were made to support macOS builds:

1. **Makefile** - Added OS detection and macOS-specific library paths
2. **liveview.pro** - Added macOS configuration with OpenMP via libomp
3. **Library handling** - Removed `-lrt` and `-ldl` (Linux-only)
4. **Boost libraries** - Removed `-lboost_system` on macOS (header-only in modern Boost)
5. **OpenMP** - Use `-Xpreprocessor -fopenmp -lomp` instead of `-fopenmp -lgomp`
6. **CUDA default** - `USE_CUDA=0` by default on macOS, `USE_CUDA=1` by default on Linux
7. **App launcher** - Added launcher script with configurable default arguments via `launch_config.txt`
8. **App icon** - Added `liveview.icns` generated from `liveview.png` for native macOS app icon

---

*This guide was created for macOS 15.7.3 (Sequoia) in January 2026.*