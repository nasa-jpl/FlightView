# Packaging Script Fixes - Jan 28, 2026

## Issues Fixed

### 1. Binary Name Detection Issue
**Problem:** `macdeployqt` was trying to analyze the shell script `liveview` instead of the actual binary `liveview-bin`, causing:
```
ERROR: Could not parse otool output: "... is not an object file"
```

**Solution:** Added smart binary detection that:
- First looks for `liveview-bin` (correct name after build)
- Falls back to `liveview` if needed, but verifies it's a Mach-O binary
- Exits with a clear error if neither is found
- Reports which executable is being used

### 2. Boost Library Naming
**Problem:** Script expected `-mt` suffix (e.g., `libboost_thread-mt.dylib`) but newer Homebrew Boost doesn't use this suffix.

**Solution:** Added fallback logic that tries both:
```bash
# Try -mt version first
if [ -f "$BREW_PREFIX/lib/libboost_thread-mt.dylib" ]; then
    copy_lib "libboost_thread-mt.dylib"
# Fall back to non-mt version
elif [ -f "$BREW_PREFIX/lib/libboost_thread.dylib" ]; then
    copy_lib "libboost_thread.dylib"
fi
```

### 3. Library Version Hardcoding
**Problem:** Script hardcoded specific versions:
- `libgsl.27.dylib` (but system has `libgsl.28.dylib`)
- `libzmq.5.dylib`
- `libexiv2.28.dylib`
- GStreamer library versions

**Solution:** Dynamic version detection using glob patterns:
```bash
GSL_LIB=$(ls "$BREW_PREFIX"/lib/libgsl.[0-9]*.dylib 2>/dev/null | head -1)
if [ -n "$GSL_LIB" ]; then
    copy_lib "$GSL_LIB"
else
    copy_lib "$BREW_PREFIX/lib/libgsl.dylib"  # Fallback to symlink
fi
```

This works across different Homebrew versions and library updates.

### 4. Build Script Integration
**Added:** Automatic copying of `package_macos_app.sh` to build directory at end of `build_with_metal.sh`:

```bash
# Step 5: Copy packaging script to build directory for convenience
echo "Step 5: Copying packaging script to build directory..."
if [ -f "$SCRIPT_DIR/package_macos_app.sh" ]; then
    cp "$SCRIPT_DIR/package_macos_app.sh" "$BUILD_DIR/"
    chmod +x "$BUILD_DIR/package_macos_app.sh"
    echo "Packaging script copied to: $BUILD_DIR/package_macos_app.sh"
fi
```

### 5. Duplicate Library Version Conflicts
**Problem:** During transitive dependency scanning, the script tried to copy libraries with slightly different version names, causing permission errors:
```
Found missing dependency: libexiv2.28.dylib
Found missing dependency: libexiv2.0.28.7.dylib  # Same library, different name!
cp: Permission denied
```

**Solution:** 
1. **Skip existing files** - Check if file already exists before copying
2. **Detect similar versions** - Recognize that `libexiv2.28.dylib` and `libexiv2.0.28.7.dylib` are the same library
3. **Make files writable** - Ensure copied files have write permissions for later operations

```bash
# Check if file already exists
if [ -f "$target_dir/$basename_lib" ]; then
    echo "Skipping $basename_lib (already exists)..."
    return 0
fi

# Check for similar library names (different version suffix)
base_lib=$(echo "$libname" | sed 's/\.[0-9][0-9.]*\.dylib/.dylib/')
for existing in "$FRAMEWORKS_DIR"/$base_lib*; do
    if [ -f "$existing" ]; then
        similar_found=true
        break
    fi
done
```

## Changes Made to Files

1. **`build_with_metal.sh`** - Added Step 5 to copy packaging script
2. **`package_macos_app.sh`** - Major improvements:
   - Smart binary detection
   - Flexible Boost library naming
   - Dynamic version detection for all libraries
   - Skip existing files during copy
   - Detect similar library versions
   - Better error messages and file permissions

## Testing

The updated script now handles:
- ✅ Different Boost library naming conventions
- ✅ Any GSL version (27, 28, etc.)
- ✅ Any ZeroMQ version
- ✅ Any Exiv2 version
- ✅ Binary detection for both `liveview` and `liveview-bin`
- ✅ Duplicate library versions (e.g., libexiv2.28.dylib vs libexiv2.0.28.7.dylib)
- ✅ File permission issues when overwriting
- ✅ Clear error messages when things go wrong

## Usage

From your build directory (e.g., `/Users/eliggett/Documents/liveview/20260126/m3`):

```bash
# The script is now automatically copied there after building
./package_macos_app.sh
```

Expected output:
```
======================================
FlightView macOS Packaging Script
======================================
Detected Apple Silicon Mac (Homebrew at /opt/homebrew)
Using executable: lv_release/liveview.app/Contents/MacOS/liveview-bin

Step 1: Bundling Qt frameworks with macdeployqt...
✓ Qt frameworks bundled

Step 2: Copying non-Qt libraries to app bundle...
Boost libraries:
  Copying libboost_thread.dylib...
  Copying libboost_filesystem.dylib...
GSL libraries:
  Copying libgsl.28.dylib...
  Copying libgslcblas.0.dylib...
[... continues with all libraries ...]

✓ Libraries copied
✓ Library paths updated
✓ All transitive dependencies resolved
✓ Bundle is fully self-contained!
✓ App bundle signed
✓ DMG created: FlightView-macOS-arm64.dmg

====================================
✓ Packaging Complete!
====================================
```

## Notes

- The script now works across different Homebrew installations and library versions
- Future library updates won't break the packaging process
- The script provides clear feedback at each step
- All errors include actionable information
