# macOS Application Packaging Guide

**Goal:** Create a self-contained FlightView.app bundle that colleagues can run without installing Homebrew or any dependencies.

**Status:** Fully achievable using macOS app bundling and `macdeployqt`.

---

## Overview

macOS applications are distributed as **App Bundles** (`.app` directories) that contain:
- The executable binary
- All required dynamic libraries (frameworks)
- Resources (images, config files, Metal shaders)
- Metadata (Info.plist)

This guide will help you create a fully self-contained FlightView distribution.

---

## Strategy: Self-Contained App Bundle

### What Goes Inside the Bundle

```
FlightView.app/
├── Contents/
│   ├── MacOS/
│   │   ├── liveview              # Launcher script
│   │   ├── liveview-bin          # Main executable
│   │   └── std_dev_filter.metallib  # Metal GPU shader
│   ├── Frameworks/               # All dependencies bundled here
│   │   ├── QtCore.framework
│   │   ├── QtGui.framework
│   │   ├── QtWidgets.framework
│   │   ├── QtNetwork.framework
│   │   ├── libboost_thread.dylib
│   │   ├── libboost_filesystem.dylib
│   │   ├── libgsl.dylib
│   │   ├── libgslcblas.dylib
│   │   ├── libzmq.dylib
│   │   ├── libexiv2.dylib
│   │   ├── libomp.dylib
│   │   ├── GStreamer.framework (or dylibs)
│   │   └── ... (all transitive dependencies)
│   ├── Resources/
│   │   ├── liveview.icns
│   │   └── launch_config.txt
│   └── Info.plist
```

### Key Concepts

1. **Dynamic Libraries (.dylib)** - Copied into `Contents/Frameworks/`
2. **Library Paths** - Modified to point inside the bundle using `@rpath`
3. **Qt Deployment** - `macdeployqt` automates Qt framework bundling
4. **Manual Dependencies** - Non-Qt libraries must be copied manually
5. **Code Signing** - Optional but recommended for distribution

---

## Step-by-Step Packaging Process

### Phase 1: Build with Release Configuration

```bash
# Clean build to ensure everything is fresh
cd /path/to/FlightView
cd backend && make clean && cd ..
make clean
rm -rf lv_release/

# Build with Metal support
cd /path/to/build_directory
../FlightView/build_with_metal.sh

# Or build without Metal (CPU only)
cd backend && make -j8 && cd ..
qmake ../FlightView/liveview.pro CONFIG+=release
make -j8
```

This produces: `lv_release/liveview.app`

---

### Phase 2: Bundle Qt Frameworks (Automated)

`macdeployqt` is a Qt tool that automatically:
- Copies Qt frameworks into the bundle
- Updates library paths to use `@rpath`
- Fixes Qt dependencies

```bash
cd lv_release

# Determine your Homebrew prefix
if [ -d /opt/homebrew ]; then
    HOMEBREW_PREFIX=/opt/homebrew
else
    HOMEBREW_PREFIX=/usr/local
fi

# Run macdeployqt
$HOMEBREW_PREFIX/opt/qt@5/bin/macdeployqt liveview.app

# Verify Qt frameworks were bundled
ls -la liveview.app/Contents/Frameworks/
```

**What macdeployqt does:**
- Copies QtCore, QtGui, QtWidgets, QtNetwork, QtSvg, QtPrintSupport
- Copies Qt plugins (imageformats, platforms, styles)
- Updates all Qt library paths

---

### Phase 3: Bundle Non-Qt Libraries (Manual)

Create a script to bundle all remaining dependencies:

```bash
#!/bin/bash
# bundle_dependencies.sh - Bundle all non-Qt libraries

set -e

APP_BUNDLE="lv_release/liveview.app"
EXECUTABLE="$APP_BUNDLE/Contents/MacOS/liveview-bin"
FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"

# Detect Homebrew prefix
if [ -d /opt/homebrew ]; then
    BREW_PREFIX=/opt/homebrew
else
    BREW_PREFIX=/usr/local
fi

echo "Bundling non-Qt dependencies..."

# Create Frameworks directory if it doesn't exist
mkdir -p "$FRAMEWORKS_DIR"

# List of libraries to bundle (adjust as needed)
LIBS=(
    "$BREW_PREFIX/lib/libboost_thread-mt.dylib"
    "$BREW_PREFIX/lib/libboost_filesystem-mt.dylib"
    "$BREW_PREFIX/lib/libgsl.27.dylib"
    "$BREW_PREFIX/lib/libgslcblas.0.dylib"
    "$BREW_PREFIX/lib/libzmq.5.dylib"
    "$BREW_PREFIX/lib/libexiv2.28.dylib"
    "$BREW_PREFIX/opt/libomp/lib/libomp.dylib"
)

# Copy each library
for lib in "${LIBS[@]}"; do
    if [ -f "$lib" ]; then
        cp -v "$lib" "$FRAMEWORKS_DIR/"
    else
        echo "WARNING: Library not found: $lib"
    fi
done

echo "Bundling GStreamer libraries..."
# GStreamer has many dependencies, copy main ones
cp -v "$BREW_PREFIX"/lib/libgstreamer-1.0.*.dylib "$FRAMEWORKS_DIR/" || true
cp -v "$BREW_PREFIX"/lib/libgstapp-1.0.*.dylib "$FRAMEWORKS_DIR/" || true
cp -v "$BREW_PREFIX"/lib/libgstbase-1.0.*.dylib "$FRAMEWORKS_DIR/" || true
cp -v "$BREW_PREFIX"/lib/libglib-2.0.*.dylib "$FRAMEWORKS_DIR/" || true
cp -v "$BREW_PREFIX"/lib/libgobject-2.0.*.dylib "$FRAMEWORKS_DIR/" || true

# Copy GStreamer plugins (optional, needed for video codecs)
GSTREAMER_PLUGINS="$APP_BUNDLE/Contents/PlugIns/gstreamer"
mkdir -p "$GSTREAMER_PLUGINS"
cp -R "$BREW_PREFIX"/lib/gstreamer-1.0/*.so "$GSTREAMER_PLUGINS/" || true

echo "Libraries bundled successfully"
```

Save this as `bundle_dependencies.sh` and run:

```bash
chmod +x bundle_dependencies.sh
./bundle_dependencies.sh
```

---

### Phase 4: Fix Library Paths (Critical!)

After copying libraries, you must update the executable and all libraries to look for dependencies inside the bundle using `@rpath` instead of absolute Homebrew paths.

```bash
#!/bin/bash
# fix_library_paths.sh - Update all library references to use @rpath

set -e

APP_BUNDLE="lv_release/liveview.app"
EXECUTABLE="$APP_BUNDLE/Contents/MacOS/liveview-bin"
FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"

# Detect Homebrew prefix
if [ -d /opt/homebrew ]; then
    BREW_PREFIX=/opt/homebrew
else
    BREW_PREFIX=/usr/local
fi

echo "Fixing library paths in executable..."

# Add rpath to executable to find Frameworks directory
install_name_tool -add_rpath "@executable_path/../Frameworks" "$EXECUTABLE" 2>/dev/null || true

# Fix paths in main executable
for lib in "$FRAMEWORKS_DIR"/*.dylib; do
    libname=$(basename "$lib")
    echo "Fixing $libname in executable"
    install_name_tool -change "$BREW_PREFIX/lib/$libname" "@rpath/$libname" "$EXECUTABLE" 2>/dev/null || true
    install_name_tool -change "$BREW_PREFIX/opt/libomp/lib/$libname" "@rpath/$libname" "$EXECUTABLE" 2>/dev/null || true
done

echo "Fixing library paths between libraries..."

# Fix inter-library dependencies
for lib in "$FRAMEWORKS_DIR"/*.dylib; do
    libname=$(basename "$lib")
    echo "Processing $libname"
    
    # Update the library's own install name to use @rpath
    install_name_tool -id "@rpath/$libname" "$lib" 2>/dev/null || true
    
    # Fix dependencies of this library
    for dep in "$FRAMEWORKS_DIR"/*.dylib; do
        depname=$(basename "$dep")
        install_name_tool -change "$BREW_PREFIX/lib/$depname" "@rpath/$depname" "$lib" 2>/dev/null || true
        install_name_tool -change "$BREW_PREFIX/opt/libomp/lib/$depname" "@rpath/$depname" "$lib" 2>/dev/null || true
    done
done

echo "Library paths fixed successfully"
```

Save as `fix_library_paths.sh` and run:

```bash
chmod +x fix_library_paths.sh
./fix_library_paths.sh
```

---

### Phase 5: Handle Transitive Dependencies

Some libraries depend on other libraries not directly used by your app. Use `otool` to find them:

```bash
#!/bin/bash
# find_missing_deps.sh - Find all transitive dependencies

APP_BUNDLE="lv_release/liveview.app"
EXECUTABLE="$APP_BUNDLE/Contents/MacOS/liveview-bin"
FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"

echo "Checking executable dependencies:"
otool -L "$EXECUTABLE" | grep -E '(/usr/local|/opt/homebrew)'

echo ""
echo "Checking library dependencies:"
for lib in "$FRAMEWORKS_DIR"/*.dylib; do
    echo "--- $(basename $lib) ---"
    otool -L "$lib" | grep -E '(/usr/local|/opt/homebrew)'
done
```

For each missing dependency found:
1. Copy it to `Contents/Frameworks/`
2. Run `fix_library_paths.sh` again

Repeat until no external paths remain (except system libraries like `/usr/lib/`).

---

### Phase 6: Code Signing (Recommended)

For distribution, sign the app to avoid macOS Gatekeeper warnings:

```bash
# Self-signed certificate (for local distribution)
codesign --force --deep --sign - liveview.app

# Or use a Developer ID certificate (for public distribution)
# codesign --force --deep --sign "Developer ID Application: Your Name" liveview.app
```

---

### Phase 7: Create Distribution Package

#### Option A: DMG File (Recommended)

Create a drag-and-drop disk image:

```bash
# Create a temporary directory
mkdir FlightView-dist
cp -R lv_release/liveview.app FlightView-dist/

# Optional: Add README
cat > FlightView-dist/README.txt << 'EOF'
FlightView for macOS

Installation:
1. Drag FlightView.app to your Applications folder
2. Double-click to launch
3. Configure launch arguments by editing:
   FlightView.app/Contents/Resources/launch_config.txt

For more information, see documentation at:
https://github.com/nasa-jpl/LiveView
EOF

# Create DMG
hdiutil create -volname "FlightView" \
    -srcfolder FlightView-dist \
    -ov -format UDZO \
    FlightView-macOS.dmg

# Clean up
rm -rf FlightView-dist

echo "Created: FlightView-macOS.dmg"
```

#### Option B: ZIP Archive

```bash
# Create a zip file
cd lv_release
zip -r ../FlightView-macOS.zip liveview.app
cd ..
```

---

## Automated All-in-One Script

Here's a complete script that does everything:

```bash
#!/bin/bash
# package_macos_app.sh - Complete packaging automation

set -e

echo "======================================"
echo "FlightView macOS Packaging Script"
echo "======================================"

# Detect Homebrew prefix
if [ -d /opt/homebrew ]; then
    BREW_PREFIX=/opt/homebrew
else
    BREW_PREFIX=/usr/local
fi

APP_BUNDLE="lv_release/liveview.app"
EXECUTABLE="$APP_BUNDLE/Contents/MacOS/liveview-bin"
FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"

# Check that app exists
if [ ! -d "$APP_BUNDLE" ]; then
    echo "ERROR: $APP_BUNDLE not found. Build the app first."
    exit 1
fi

# Step 1: Run macdeployqt
echo ""
echo "Step 1: Bundling Qt frameworks..."
$BREW_PREFIX/opt/qt@5/bin/macdeployqt "$APP_BUNDLE"

# Step 2: Bundle non-Qt libraries
echo ""
echo "Step 2: Copying non-Qt libraries..."
mkdir -p "$FRAMEWORKS_DIR"

# Function to copy library and its symlinks
copy_lib() {
    local lib_pattern="$1"
    local target_dir="$2"
    
    # Find the actual library file (not symlink)
    local lib_file=$(find $(dirname "$lib_pattern") -name "$(basename $lib_pattern)" -type f 2>/dev/null | head -1)
    
    if [ -n "$lib_file" ]; then
        cp -v "$lib_file" "$target_dir/"
        # Create symlinks if needed
        local lib_name=$(basename "$lib_file")
        for link in $(find $(dirname "$lib_pattern") -name "$(basename $lib_pattern .dylib)*.dylib" -type l 2>/dev/null); do
            local link_name=$(basename "$link")
            ln -sf "$lib_name" "$target_dir/$link_name"
        done
    fi
}

# Copy Boost
copy_lib "$BREW_PREFIX/lib/libboost_thread-mt.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libboost_filesystem-mt.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libboost_system-mt.dylib" "$FRAMEWORKS_DIR"

# Copy GSL
copy_lib "$BREW_PREFIX/lib/libgsl.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libgslcblas.dylib" "$FRAMEWORKS_DIR"

# Copy ZeroMQ
copy_lib "$BREW_PREFIX/lib/libzmq.dylib" "$FRAMEWORKS_DIR"

# Copy Exiv2
copy_lib "$BREW_PREFIX/lib/libexiv2.dylib" "$FRAMEWORKS_DIR"

# Copy OpenMP
copy_lib "$BREW_PREFIX/opt/libomp/lib/libomp.dylib" "$FRAMEWORKS_DIR"

# Copy GStreamer (main libraries)
copy_lib "$BREW_PREFIX/lib/libgstreamer-1.0.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libgstapp-1.0.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libgstbase-1.0.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libglib-2.0.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libgobject-2.0.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libgio-2.0.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libgmodule-2.0.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libintl.dylib" "$FRAMEWORKS_DIR"
copy_lib "$BREW_PREFIX/lib/libpcre2-8.dylib" "$FRAMEWORKS_DIR"

# Step 3: Fix library paths
echo ""
echo "Step 3: Fixing library paths..."

# Add rpath to executable
install_name_tool -add_rpath "@executable_path/../Frameworks" "$EXECUTABLE" 2>/dev/null || true

# Function to fix paths in a binary
fix_paths() {
    local binary="$1"
    
    # Get all dependencies
    local deps=$(otool -L "$binary" | grep -E "(/usr/local|/opt/homebrew)" | awk '{print $1}')
    
    for dep in $deps; do
        local libname=$(basename "$dep")
        # Try to change the path
        install_name_tool -change "$dep" "@rpath/$libname" "$binary" 2>/dev/null || true
    done
}

# Fix executable
echo "Fixing executable..."
fix_paths "$EXECUTABLE"

# Fix all libraries
echo "Fixing libraries..."
for lib in "$FRAMEWORKS_DIR"/*.dylib; do
    if [ -f "$lib" ]; then
        echo "Processing $(basename $lib)..."
        # Update library's own install name
        install_name_tool -id "@rpath/$(basename $lib)" "$lib" 2>/dev/null || true
        # Fix dependencies
        fix_paths "$lib"
    fi
done

# Step 4: Check for missing dependencies
echo ""
echo "Step 4: Checking for missing dependencies..."

check_deps() {
    local binary="$1"
    local missing=$(otool -L "$binary" | grep -E "(/usr/local|/opt/homebrew)" || true)
    if [ -n "$missing" ]; then
        echo "WARNING: External dependencies found in $(basename $binary):"
        echo "$missing"
        return 1
    fi
    return 0
}

all_good=true
check_deps "$EXECUTABLE" || all_good=false

for lib in "$FRAMEWORKS_DIR"/*.dylib; do
    if [ -f "$lib" ]; then
        check_deps "$lib" || all_good=false
    fi
done

if [ "$all_good" = false ]; then
    echo ""
    echo "WARNING: Some external dependencies remain."
    echo "You may need to copy additional libraries and re-run this script."
else
    echo ""
    echo "✓ All dependencies are bundled!"
fi

# Step 5: Code signing
echo ""
echo "Step 5: Code signing..."
codesign --force --deep --sign - "$APP_BUNDLE"

# Step 6: Create DMG
echo ""
echo "Step 6: Creating DMG..."

DMG_DIR="FlightView-dist"
rm -rf "$DMG_DIR"
mkdir "$DMG_DIR"
cp -R "$APP_BUNDLE" "$DMG_DIR/"

cat > "$DMG_DIR/README.txt" << 'EOF'
FlightView for macOS

Installation:
1. Drag FlightView.app to your Applications folder
2. Double-click to launch with default settings
3. To customize launch arguments, edit:
   FlightView.app/Contents/Resources/launch_config.txt

For Metal GPU acceleration, the app requires:
- macOS 10.13 (High Sierra) or later
- A Mac with Metal-capable GPU

Documentation: https://github.com/nasa-jpl/LiveView
EOF

hdiutil create -volname "FlightView" \
    -srcfolder "$DMG_DIR" \
    -ov -format UDZO \
    FlightView-macOS.dmg

rm -rf "$DMG_DIR"

echo ""
echo "======================================"
echo "Packaging Complete!"
echo "======================================"
echo ""
echo "Created: FlightView-macOS.dmg"
echo ""
echo "To verify the bundle is self-contained:"
echo "  otool -L $EXECUTABLE"
echo ""
echo "To test on a clean system:"
echo "  1. Copy FlightView-macOS.dmg to a Mac without Homebrew"
echo "  2. Mount the DMG and drag the app to Applications"
echo "  3. Launch and verify all features work"
echo ""
