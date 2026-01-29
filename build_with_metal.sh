#!/bin/bash
# Build FlightView with Metal GPU acceleration (macOS only)

set -e  # Exit on error

# Source user's shell configuration to get PATH and environment
if [ -f ~/.zshrc ]; then
    source ~/.zshrc
fi

echo "========================================"
echo "Building FlightView with Metal GPU Support"
echo "========================================"

# Check if running on macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: Metal is only supported on macOS"
    exit 1
fi

# Get the directory where this script is located (the source directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Source directory: $SCRIPT_DIR"

# Remember where we started (build directory)
BUILD_DIR="$(pwd)"
echo "Build directory: $BUILD_DIR"

# Check if we're running from within the source directory
if [[ "$BUILD_DIR" == "$SCRIPT_DIR"* ]]; then
    echo ""
    echo "========================================"
    echo "ERROR: Wrong directory!"
    echo "========================================"
    echo ""
    echo "This script should NOT be run from within the source directory."
    echo "Please run it from a separate build directory."
    echo ""
    echo "Example:"
    echo "  mkdir -p ../build"
    echo "  cd ../build"
    echo "  ../FlightView/build_with_metal.sh"
    echo ""
    echo "Current directory: $BUILD_DIR"
    echo "Source directory:  $SCRIPT_DIR"
    echo ""
    exit 1
fi

# Step 1: Compile Metal shader
echo ""
echo "Step 1: Compiling Metal shader..."
cd "$SCRIPT_DIR/backend/src"

if [ ! -f std_dev_filter.metal ]; then
    echo "ERROR: std_dev_filter.metal not found"
    exit 1
fi

# Compile Metal source to AIR (Apple Intermediate Representation)
xcrun -sdk macosx metal -c std_dev_filter.metal -o std_dev_filter.air

if [ $? -ne 0 ]; then
    echo "ERROR: Metal compilation failed"
    exit 1
fi

# Link AIR to Metal library (place in build directory)
xcrun -sdk macosx metallib std_dev_filter.air -o "$BUILD_DIR/std_dev_filter.metallib"

if [ $? -ne 0 ]; then
    echo "ERROR: Metal library creation failed"
    exit 1
fi

echo "Metal shader compiled successfully: std_dev_filter.metallib"

# Clean up intermediate files
rm -f std_dev_filter.air

# Step 2: Build backend library with Metal support
echo ""
echo "Step 2: Building backend library with Metal..."
cd "$SCRIPT_DIR/backend"
make clean
make USE_METAL=1 -j8

if [ $? -ne 0 ]; then
    echo "ERROR: backend build failed"
    exit 1
fi

echo "backend library built successfully"

# Step 3: Build FlightView application
echo ""
echo "Step 3: Building FlightView application..."
cd "$BUILD_DIR"

# Run qmake from source directory
qmake "$SCRIPT_DIR/liveview.pro"

# Build
make -j8

if [ $? -ne 0 ]; then
    echo "ERROR: FlightView build failed"
    exit 1
fi

# Step 4: Copy Metal library to output directories
echo ""
echo "Step 4: Installing Metal shader library..."

# Copy to build directory output
if [ -d "$BUILD_DIR/lv_release" ]; then
    cp "$BUILD_DIR/std_dev_filter.metallib" "$BUILD_DIR/lv_release/"
    echo "Copied Metal library to $BUILD_DIR/lv_release/"
    
    # Also copy to app bundle if it exists
    if [ -d "$BUILD_DIR/lv_release/liveview.app" ]; then
        cp "$BUILD_DIR/std_dev_filter.metallib" "$BUILD_DIR/lv_release/liveview.app/Contents/MacOS/"
        echo "Copied Metal library to app bundle"
    fi
elif [ -f "$BUILD_DIR/liveview" ]; then
    # Built directly in build directory
    echo "Metal library is at: $BUILD_DIR/std_dev_filter.metallib"
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Build directory: $BUILD_DIR"
echo "Source directory: $SCRIPT_DIR"
echo ""
if [ -d "$BUILD_DIR/lv_release" ]; then
    echo "The compiled application is in: $BUILD_DIR/lv_release/"
    echo ""
    echo "To run:"
    echo "  cd $BUILD_DIR/lv_release"
    echo "  ./liveview --rtpnextgen --rtpwidth 1280 --rtpheight 328 --no-gps ..."
else
    echo "The compiled application is in: $BUILD_DIR/"
    echo ""
    echo "To run:"
    echo "  cd $BUILD_DIR"
    echo "  ./liveview --rtpnextgen --rtpwidth 1280 --rtpheight 328 --no-gps ..."
fi
echo ""
echo "Metal shader library: std_dev_filter.metallib"
echo "Metal GPU acceleration will be used for standard deviation calculations."
echo ""

# Step 5: Copy packaging script to build directory for convenience
echo "Step 5: Copying packaging script to build directory..."
if [ -f "$SCRIPT_DIR/package_macos_app.sh" ]; then
    cp "$SCRIPT_DIR/package_macos_app.sh" "$BUILD_DIR/"
    chmod +x "$BUILD_DIR/package_macos_app.sh"
    echo "Packaging script copied to: $BUILD_DIR/package_macos_app.sh"
    echo ""
    echo "To create a distributable DMG, run:"
    echo "  cd $BUILD_DIR"
    echo "  ./package_macos_app.sh"
else
    echo "Note: package_macos_app.sh not found in source directory"
fi
echo ""
