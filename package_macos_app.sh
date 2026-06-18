#!/bin/bash
# package_macos_app.sh - Complete FlightView macOS packaging automation
#
# This script creates a self-contained .app bundle and DMG for distribution.
# No Homebrew or external dependencies required on target machines.

set -e

echo "======================================"
echo "FlightView macOS Packaging Script"
echo "======================================"

# Detect Homebrew prefix
if [ -d /opt/homebrew ]; then
    BREW_PREFIX=/opt/homebrew
    echo "Detected Apple Silicon Mac (Homebrew at /opt/homebrew)"
else
    BREW_PREFIX=/usr/local
    echo "Detected Intel Mac (Homebrew at /usr/local)"
fi

APP_BUNDLE="lv_release/liveview.app"
EXECUTABLE="$APP_BUNDLE/Contents/MacOS/liveview-bin"
FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"

# Check if the binary exists (should be liveview-bin after build)
if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: Binary not found at $EXECUTABLE"
    echo "Looking for the actual binary..."
    
    # Check if it's just 'liveview' (pre-launcher rename)
    if [ -f "$APP_BUNDLE/Contents/MacOS/liveview" ]; then
        # Check if it's the actual binary or the shell script
        if file "$APP_BUNDLE/Contents/MacOS/liveview" | grep -q "Mach-O"; then
            echo "Found Mach-O binary at liveview (not yet renamed to liveview-bin)"
            EXECUTABLE="$APP_BUNDLE/Contents/MacOS/liveview"
        else
            echo "ERROR: liveview exists but is a script, not the binary."
            echo "The build process should have created liveview-bin."
            exit 1
        fi
    else
        echo "ERROR: No binary found in $APP_BUNDLE/Contents/MacOS/"
        ls -la "$APP_BUNDLE/Contents/MacOS/" || true
        exit 1
    fi
fi

echo "Using executable: $EXECUTABLE"

# Check that app exists
if [ ! -d "$APP_BUNDLE" ]; then
    echo ""
    echo "ERROR: $APP_BUNDLE not found."
    echo "Please build the app first using build_with_metal.sh or regular build process."
    echo ""
    exit 1
fi

# Check that Metal shader exists if built with Metal
METAL_LIB="$APP_BUNDLE/Contents/MacOS/std_dev_filter.metallib"
if [ ! -f "$METAL_LIB" ]; then
    echo ""
    echo "WARNING: Metal shader library not found at $METAL_LIB"
    echo "If you built with Metal support, make sure the .metallib file is copied to the app bundle."
    echo ""
fi

# Step 1: Prepare for macdeployqt and run it
echo ""
echo "Step 1: Preparing app bundle and bundling Qt frameworks..."

# Clean up any previous packaging attempts (Frameworks might have read-only files)
if [ -d "$FRAMEWORKS_DIR" ]; then
    echo "  Cleaning up previous Frameworks directory..."
    chmod -R u+w "$FRAMEWORKS_DIR" 2>/dev/null || true
    rm -rf "$FRAMEWORKS_DIR"
fi

# macdeployqt expects the main executable to match the app name (liveview)
# But our build creates liveview-bin (actual binary) and liveview (shell wrapper)
# Temporarily swap them so macdeployqt processes the correct binary
WRAPPER="$APP_BUNDLE/Contents/MacOS/liveview"
BINARY="$APP_BUNDLE/Contents/MacOS/liveview-bin"

if [ -f "$WRAPPER" ] && [ -f "$BINARY" ]; then
    # Check if wrapper is actually a script (not binary)
    if ! file "$WRAPPER" | grep -q "Mach-O"; then
        echo "  Temporarily renaming wrapper for macdeployqt..."
        mv "$WRAPPER" "$WRAPPER.tmp"
        cp "$BINARY" "$WRAPPER"  # Copy, don't move, so both exist
        RESTORE_WRAPPER=true
    else
        RESTORE_WRAPPER=false
    fi
else
    RESTORE_WRAPPER=false
fi

echo "  Running macdeployqt..."
# Use -always-overwrite to ensure fresh deployment and include all Qt modules
$BREW_PREFIX/opt/qt@5/bin/macdeployqt "$APP_BUNDLE" -always-overwrite -verbose=0

MACDEPLOYQT_STATUS=$?

# Restore the original wrapper if we moved it
if [ "$RESTORE_WRAPPER" = true ]; then
    echo "  Restoring launcher wrapper..."
    rm -f "$WRAPPER"  # Remove the copy we made
    mv "$WRAPPER.tmp" "$WRAPPER"
fi

if [ $MACDEPLOYQT_STATUS -ne 0 ]; then
    echo "ERROR: macdeployqt failed"
    exit 1
fi

# Make all Qt frameworks writable (macdeployqt sometimes makes them read-only)
if [ -d "$FRAMEWORKS_DIR" ]; then
    chmod -R u+w "$FRAMEWORKS_DIR" 2>/dev/null || true
fi

# Verify Qt platform plugins were installed
PLUGINS_DIR="$APP_BUNDLE/Contents/PlugIns"
PLATFORMS_DIR="$PLUGINS_DIR/platforms"

if [ ! -d "$PLATFORMS_DIR" ] || [ ! -f "$PLATFORMS_DIR/libqcocoa.dylib" ]; then
    echo "  WARNING: Qt platform plugins missing, manually copying..."
    mkdir -p "$PLATFORMS_DIR"
    
    # Find Qt plugins directory
    QT_PLUGINS_DIR="$BREW_PREFIX/opt/qt@5/plugins"
    if [ -d "$QT_PLUGINS_DIR/platforms" ]; then
        cp -v "$QT_PLUGINS_DIR/platforms/libqcocoa.dylib" "$PLATFORMS_DIR/"
        
        # Also copy other useful plugins
        mkdir -p "$PLUGINS_DIR/imageformats"
        mkdir -p "$PLUGINS_DIR/styles"
        cp -v "$QT_PLUGINS_DIR"/imageformats/*.dylib "$PLUGINS_DIR/imageformats/" 2>/dev/null || true
        cp -v "$QT_PLUGINS_DIR"/styles/*.dylib "$PLUGINS_DIR/styles/" 2>/dev/null || true
    fi
fi

# Verify Qt frameworks needed by platform plugin (QtDBus, etc.)
echo "  Checking Qt framework dependencies for plugins..."
QT_LIB_DIR="$BREW_PREFIX/Cellar/qt@5/5.15.18/lib"
if [ ! -d "$QT_LIB_DIR" ]; then
    # Try to find Qt lib directory dynamically
    QT_LIB_DIR=$(ls -d "$BREW_PREFIX"/Cellar/qt@5/*/lib 2>/dev/null | head -1)
fi

# Copy QtDBus if missing (needed by libqcocoa.dylib)
if [ ! -d "$FRAMEWORKS_DIR/QtDBus.framework" ] && [ -d "$QT_LIB_DIR/QtDBus.framework" ]; then
    echo "  Copying QtDBus.framework (required by cocoa plugin)..."
    cp -R "$QT_LIB_DIR/QtDBus.framework" "$FRAMEWORKS_DIR/"
fi

# Copy libfreetype if missing (needed by QtGui/QtWidgets/plugins)
FREETYPE_LIB=$(ls "$BREW_PREFIX"/lib/libfreetype.*.dylib 2>/dev/null | head -1)
if [ -n "$FREETYPE_LIB" ] && [ ! -f "$FRAMEWORKS_DIR/$(basename $FREETYPE_LIB)" ]; then
    echo "  Copying libfreetype (required by Qt)..."
    cp "$FREETYPE_LIB" "$FRAMEWORKS_DIR/"
fi

# Create qt.conf to tell Qt where to find plugins
QT_CONF="$APP_BUNDLE/Contents/Resources/qt.conf"
cat > "$QT_CONF" << 'EOF'
[Paths]
Plugins = PlugIns
EOF

echo "✓ Qt frameworks bundled"

# Step 2: Bundle non-Qt libraries
echo ""
echo "Step 2: Copying non-Qt libraries to app bundle..."
mkdir -p "$FRAMEWORKS_DIR"

# Function to copy library and handle symlinks
copy_lib() {
    local lib_path="$1"
    local target_dir="$2"
    
    # Resolve symlinks to find the real file
    if [ -L "$lib_path" ]; then
        local real_lib=$(readlink "$lib_path")
        # Handle relative symlinks
        if [[ "$real_lib" != /* ]]; then
            real_lib="$(dirname "$lib_path")/$real_lib"
        fi
        lib_path="$real_lib"
    fi
    
    if [ -f "$lib_path" ]; then
        local basename_lib=$(basename "$lib_path")
        
        # Check if file already exists in target
        if [ -f "$target_dir/$basename_lib" ]; then
            echo "  Skipping $(basename $lib_path) (already exists)..."
            return 0
        fi
        
        echo "  Copying $(basename $lib_path)..."
        cp "$lib_path" "$target_dir/"
        
        # Make sure we own the file and can modify it later
        chmod u+w "$target_dir/$basename_lib" 2>/dev/null || true
        
        return 0
    else
        echo "  WARNING: Library not found: $lib_path"
        return 1
    fi
}

# Copy Boost libraries (try both -mt and non-mt versions)
echo "Boost libraries:"
for boostlib in thread filesystem system atomic; do
    if [ -f "$BREW_PREFIX/lib/libboost_${boostlib}-mt.dylib" ]; then
        copy_lib "$BREW_PREFIX/lib/libboost_${boostlib}-mt.dylib" "$FRAMEWORKS_DIR"
    elif [ -f "$BREW_PREFIX/lib/libboost_${boostlib}.dylib" ]; then
        copy_lib "$BREW_PREFIX/lib/libboost_${boostlib}.dylib" "$FRAMEWORKS_DIR"
    fi
done

# Copy GSL (try versioned first, then fall back to generic symlink)
echo "GSL libraries:"
# Find the actual versioned GSL library
GSL_LIB=$(ls "$BREW_PREFIX"/lib/libgsl.[0-9]*.dylib 2>/dev/null | head -1)
if [ -n "$GSL_LIB" ]; then
    copy_lib "$GSL_LIB" "$FRAMEWORKS_DIR"
else
    copy_lib "$BREW_PREFIX/lib/libgsl.dylib" "$FRAMEWORKS_DIR"
fi

GSLCBLAS_LIB=$(ls "$BREW_PREFIX"/lib/libgslcblas.[0-9]*.dylib 2>/dev/null | head -1)
if [ -n "$GSLCBLAS_LIB" ]; then
    copy_lib "$GSLCBLAS_LIB" "$FRAMEWORKS_DIR"
else
    copy_lib "$BREW_PREFIX/lib/libgslcblas.dylib" "$FRAMEWORKS_DIR"
fi

# Copy ZeroMQ
echo "ZeroMQ libraries:"
ZMQ_LIB=$(ls "$BREW_PREFIX"/lib/libzmq.[0-9]*.dylib 2>/dev/null | head -1)
if [ -n "$ZMQ_LIB" ]; then
    copy_lib "$ZMQ_LIB" "$FRAMEWORKS_DIR"
else
    copy_lib "$BREW_PREFIX/lib/libzmq.dylib" "$FRAMEWORKS_DIR"
fi

# Copy Exiv2
echo "Exiv2 libraries:"
EXIV2_LIB=$(ls "$BREW_PREFIX"/lib/libexiv2.[0-9]*.dylib 2>/dev/null | head -1)
if [ -n "$EXIV2_LIB" ]; then
    copy_lib "$EXIV2_LIB" "$FRAMEWORKS_DIR"
else
    copy_lib "$BREW_PREFIX/lib/libexiv2.dylib" "$FRAMEWORKS_DIR"
fi

# Copy OpenMP
echo "OpenMP libraries:"
copy_lib "$BREW_PREFIX/opt/libomp/lib/libomp.dylib" "$FRAMEWORKS_DIR"

# Copy GStreamer libraries
echo "GStreamer libraries:"
for gstlib in gstreamer-1.0 gstapp-1.0 gstbase-1.0; do
    VERSIONED=$(ls "$BREW_PREFIX"/lib/lib${gstlib}.[0-9]*.dylib 2>/dev/null | head -1)
    if [ -n "$VERSIONED" ]; then
        copy_lib "$VERSIONED" "$FRAMEWORKS_DIR"
    else
        copy_lib "$BREW_PREFIX/lib/lib${gstlib}.dylib" "$FRAMEWORKS_DIR"
    fi
done

# Copy GLib/GObject (GStreamer dependencies)
echo "GLib/GObject libraries:"
for glib in glib-2.0 gobject-2.0 gio-2.0 gmodule-2.0; do
    VERSIONED=$(ls "$BREW_PREFIX"/lib/lib${glib}.[0-9]*.dylib 2>/dev/null | head -1)
    if [ -n "$VERSIONED" ]; then
        copy_lib "$VERSIONED" "$FRAMEWORKS_DIR"
    else
        copy_lib "$BREW_PREFIX/lib/lib${glib}.dylib" "$FRAMEWORKS_DIR"
    fi
done

# Copy common dependencies
echo "Common dependencies:"
for commonlib in intl pcre2-8; do
    VERSIONED=$(ls "$BREW_PREFIX"/lib/lib${commonlib}.[0-9]*.dylib 2>/dev/null | head -1)
    if [ -n "$VERSIONED" ]; then
        copy_lib "$VERSIONED" "$FRAMEWORKS_DIR"
    else
        copy_lib "$BREW_PREFIX/lib/lib${commonlib}.dylib" "$FRAMEWORKS_DIR"
    fi
done

# Copy Exiv2 dependencies
if [ -f "$BREW_PREFIX/lib/libinih.dylib" ]; then
    copy_lib "$BREW_PREFIX/lib/libinih.dylib" "$FRAMEWORKS_DIR"
fi
if [ -f "$BREW_PREFIX/lib/libexpat.dylib" ]; then
    copy_lib "$BREW_PREFIX/lib/libexpat.dylib" "$FRAMEWORKS_DIR"
fi

# Copy libsodium (ZeroMQ dependency)
if [ -f "$BREW_PREFIX/lib/libsodium.dylib" ]; then
    copy_lib "$BREW_PREFIX/lib/libsodium.dylib" "$FRAMEWORKS_DIR"
fi

echo "✓ Libraries copied"

# Step 3: Fix library paths
echo ""
echo "Step 3: Fixing library paths to use @rpath..."

# Add rpath to executable (if not already present)
install_name_tool -add_rpath "@executable_path/../Frameworks" "$EXECUTABLE" 2>/dev/null || true

# Function to fix paths in a binary
fix_paths() {
    local binary="$1"
    local binary_name=$(basename "$binary")
    
    # Get all dependencies using otool
    local deps=$(otool -L "$binary" 2>/dev/null | tail -n +2 | awk '{print $1}')
    
    for dep in $deps; do
        # Skip system libraries and already-fixed paths
        if [[ "$dep" == /System/* ]] || [[ "$dep" == /usr/lib/* ]] || [[ "$dep" == @* ]]; then
            continue
        fi
        
        # Handle Qt Frameworks (e.g., /path/to/QtCore.framework/Versions/5/QtCore)
        if [[ "$dep" == *.framework/* ]]; then
            # Extract framework name (e.g., QtCore from QtCore.framework/Versions/5/QtCore)
            local framework_name=$(echo "$dep" | sed -E 's|.*/([^/]+)\.framework/.*|\1|')
            local framework_path="$dep"
            
            # Check if this framework exists in our bundle
            if [ -d "$FRAMEWORKS_DIR/${framework_name}.framework" ]; then
                # Get the framework's internal structure (Versions/5/FrameworkName)
                local framework_internal=$(echo "$dep" | sed -E 's|.*/([^/]+\.framework/.*)|\1|')
                local new_path="@rpath/$framework_internal"
                
                install_name_tool -change "$dep" "$new_path" "$binary" 2>/dev/null || true
            fi
            continue
        fi
        
        # Handle regular dylibs
        local libname=$(basename "$dep")
        
        # Check if this library exists in our Frameworks directory
        if [ -f "$FRAMEWORKS_DIR/$libname" ]; then
            # Change the path to use @rpath
            install_name_tool -change "$dep" "@rpath/$libname" "$binary" 2>/dev/null || true
        fi
    done
}

# Fix executable
echo "  Fixing executable: liveview-bin"
fix_paths "$EXECUTABLE"

# Fix all libraries
echo "  Fixing libraries:"
for lib in "$FRAMEWORKS_DIR"/*.dylib; do
    if [ -f "$lib" ] && [ ! -L "$lib" ]; then
        libname=$(basename "$lib")
        echo "    - $libname"
        
        # Update library's own install name to use @rpath
        install_name_tool -id "@rpath/$libname" "$lib" 2>/dev/null || true
        
        # Fix this library's dependencies
        fix_paths "$lib"
    fi
done

# Fix Qt plugins
echo "  Fixing Qt plugins:"
if [ -d "$PLUGINS_DIR" ]; then
    for plugin in "$PLUGINS_DIR"/**/*.dylib; do
        if [ -f "$plugin" ]; then
            pluginname=$(basename "$plugin")
            echo "    - $pluginname"
            
            # Add rpath to plugins so they can find frameworks
            install_name_tool -add_rpath "@loader_path/../../Frameworks" "$plugin" 2>/dev/null || true
            
            # Fix plugin dependencies
            fix_paths "$plugin"
        fi
    done
fi

echo "✓ Library paths updated"

# Step 4: Iteratively find and copy missing dependencies
echo ""
echo "Step 4: Scanning for transitive dependencies..."

MAX_ITERATIONS=5
iteration=0
found_new_deps=true

while [ "$found_new_deps" = true ] && [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    echo "  Iteration $iteration..."
    found_new_deps=false
    
    # Check all dylibs in Frameworks
    for lib in "$FRAMEWORKS_DIR"/*.dylib "$EXECUTABLE"; do
        if [ -f "$lib" ] && [ ! -L "$lib" ]; then
            # Get dependencies - look for Homebrew paths AND @loader_path/@rpath references
            all_deps=$(otool -L "$lib" 2>/dev/null | tail -n +2 | awk '{print $1}' || true)
            
            for dep in $all_deps; do
                # Extract library name from various path formats
                if [[ "$dep" == @loader_path/* ]] || [[ "$dep" == @rpath/* ]]; then
                    # Handle @loader_path and @rpath references
                    libname=$(basename "$dep")
                elif [[ "$dep" == /opt/homebrew/* ]] || [[ "$dep" == /usr/local/* ]]; then
                    # Handle absolute Homebrew paths
                    libname=$(basename "$dep")
                else
                    # Skip system libraries
                    continue
                fi
                
                # Check if we already have this exact library or a similar version
                # (e.g., libexiv2.28.dylib is similar to libexiv2.0.28.7.dylib)
                if [ -f "$FRAMEWORKS_DIR/$libname" ]; then
                    continue  # Already have exact match
                fi
                
                # Check for similar library names (different version suffix)
                base_lib=$(echo "$libname" | sed 's/\.[0-9][0-9.]*\.dylib/.dylib/')
                similar_found=false
                for existing in "$FRAMEWORKS_DIR"/$base_lib* "$FRAMEWORKS_DIR"/$(echo "$base_lib" | sed 's/\.dylib$//').[0-9]*.dylib; do
                    if [ -f "$existing" ]; then
                        similar_found=true
                        break
                    fi
                done
                
                if [ "$similar_found" = false ]; then
                    echo "    Found missing dependency: $libname"
                    
                    # Try to find and copy it
                    if [ -f "$dep" ]; then
                        copy_lib "$dep" "$FRAMEWORKS_DIR"
                        found_new_deps=true
                    else
                        # Try common locations
                        if [ -f "$BREW_PREFIX/lib/$libname" ]; then
                            copy_lib "$BREW_PREFIX/lib/$libname" "$FRAMEWORKS_DIR"
                            found_new_deps=true
                        fi
                    fi
                fi
            done
        fi
    done
    
    # If we found new dependencies, fix their paths
    if [ "$found_new_deps" = true ]; then
        echo "    Fixing paths for new libraries..."
        for lib in "$FRAMEWORKS_DIR"/*.dylib; do
            if [ -f "$lib" ] && [ ! -L "$lib" ]; then
                libname=$(basename "$lib")
                install_name_tool -id "@rpath/$libname" "$lib" 2>/dev/null || true
                fix_paths "$lib"
            fi
        done
    fi
done

if [ $iteration -ge $MAX_ITERATIONS ]; then
    echo "  WARNING: Reached maximum iterations. Some dependencies may still be missing."
else
    echo "✓ All transitive dependencies resolved"
fi

# Step 5: Final verification
echo ""
echo "Step 5: Verifying bundle is self-contained..."

check_deps() {
    local binary="$1"
    local binary_name=$(basename "$binary")
    local external_deps=$(otool -L "$binary" 2>/dev/null | grep -E "(/usr/local|/opt/homebrew)" || true)
    
    if [ -n "$external_deps" ]; then
        echo "  ⚠ External dependencies in $binary_name:"
        echo "$external_deps" | sed 's/^/      /'
        return 1
    fi
    return 0
}

all_good=true

# Check executable
if ! check_deps "$EXECUTABLE"; then
    all_good=false
fi

# Check all bundled libraries
for lib in "$FRAMEWORKS_DIR"/*.dylib; do
    if [ -f "$lib" ] && [ ! -L "$lib" ]; then
        if ! check_deps "$lib"; then
            all_good=false
        fi
    fi
done

if [ "$all_good" = true ]; then
    echo "✓ Bundle is fully self-contained!"
else
    echo ""
    echo "⚠ WARNING: Some external dependencies remain."
    echo "The app may not work on systems without Homebrew."
    echo "You may need to manually copy additional libraries."
    echo ""
fi

# Step 6: Code signing
echo ""
echo "Step 6: Code signing the app bundle..."
codesign --force --deep --sign - "$APP_BUNDLE" 2>&1 | grep -v "replacing existing signature" || true
echo "✓ App bundle signed (ad-hoc signature)"

# Step 7: Create DMG
echo ""
echo "Step 7: Creating distributable DMG..."

DMG_NAME="FlightView-macOS-$(uname -m)"  # Includes architecture (arm64 or x86_64)
DMG_DIR="FlightView-dist"

rm -rf "$DMG_DIR" "$DMG_NAME.dmg"
mkdir "$DMG_DIR"
cp -R "$APP_BUNDLE" "$DMG_DIR/"

# Create README
cat > "$DMG_DIR/README.txt" << 'EOF'
FlightView for macOS

================================================================================
INSTALLATION
================================================================================

1. Drag FlightView.app to your Applications folder
2. Double-click FlightView.app to launch with default settings
3. If you see a security warning, go to:
   System Settings > Privacy & Security > Allow apps downloaded from...

================================================================================
CONFIGURATION
================================================================================

To customize launch arguments, edit:
   FlightView.app/Contents/Resources/launch_config.txt

Default configuration:
   --rtpcam --rtpnextgen --rtpheight 328 --rtpwidth 1280 
   --rtpinterface lo0 --rtpport 5004 --datastoragelocation /tmp

================================================================================
SYSTEM REQUIREMENTS
================================================================================

• macOS 10.13 (High Sierra) or later
• Metal-capable GPU (for GPU acceleration)
• 8 GB RAM recommended
• Apple Silicon (M1/M2/M3) or Intel processor

================================================================================
GPU ACCELERATION
================================================================================

This build includes Metal GPU support for standard deviation calculations.
The Metal shader is automatically used if available.

================================================================================
DOCUMENTATION
================================================================================

For complete documentation, visit:
   https://github.com/nasa-jpl/LiveView

Build date: $(date +"%Y-%m-%d")
Architecture: $(uname -m)

================================================================================
EOF

# Create DMG
echo "  Creating $DMG_NAME.dmg..."
hdiutil create -volname "FlightView" \
    -srcfolder "$DMG_DIR" \
    -ov -format UDZO \
    "$DMG_NAME.dmg" > /dev/null 2>&1

rm -rf "$DMG_DIR"

echo "✓ DMG created: $DMG_NAME.dmg"

# Final summary
echo ""
echo "======================================"
echo "✓ Packaging Complete!"
echo "======================================"
echo ""
echo "Created: $DMG_NAME.dmg"
echo "Size: $(du -h "$DMG_NAME.dmg" | awk '{print $1}')"
echo ""
echo "DISTRIBUTION:"
echo "  • Share the DMG file with your colleagues"
echo "  • No Homebrew or dependencies required on their systems"
echo "  • Works on any Mac with the same architecture ($(uname -m))"
echo ""
echo "TESTING:"
echo "  1. Mount the DMG and run the app directly"
echo "  2. Or copy to /Applications and test"
echo "  3. Verify all features work correctly"
echo ""
echo "VERIFICATION COMMANDS:"
echo "  # Check what's bundled:"
echo "  ls -lh $APP_BUNDLE/Contents/Frameworks/"
echo ""
echo "  # Verify no external dependencies:"
echo "  otool -L $EXECUTABLE | grep -v '@rpath\\|/usr/lib\\|/System'"
echo ""
