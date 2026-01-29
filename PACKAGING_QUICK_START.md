# Quick Start: Packaging FlightView for macOS Distribution

## TL;DR

Yes, you can package FlightView as a self-contained app bundle that requires **no Homebrew or dependencies** on target Macs.

```bash
# 1. Build your app (with Metal support)
./build_with_metal.sh

# 2. Run the packaging script
./package_macos_app.sh

# 3. Share the resulting DMG file
# FlightView-macOS-arm64.dmg or FlightView-macOS-x86_64.dmg
```

That's it! The DMG contains everything needed to run on any Mac of the same architecture.

---

## How It Works

The packaging process:

1. **Bundles Qt frameworks** - Uses `macdeployqt` to copy Qt libraries into the `.app` bundle
2. **Copies all dependencies** - Boost, GSL, GStreamer, ZeroMQ, Exiv2, OpenMP, etc.
3. **Fixes library paths** - Changes absolute paths to `@rpath` (relative to bundle)
4. **Resolves transitive deps** - Recursively finds and bundles dependencies of dependencies
5. **Code signs the bundle** - Prevents macOS security warnings
6. **Creates a DMG** - Standard macOS disk image for easy distribution

## What Gets Bundled

Inside `FlightView.app/Contents/Frameworks/`:
- Qt frameworks (QtCore, QtGui, QtWidgets, QtNetwork, QtSvg, etc.)
- Boost libraries (thread, filesystem)
- GSL (GNU Scientific Library)
- GStreamer (video/streaming)
- ZeroMQ (messaging)
- Exiv2 (image metadata)
- OpenMP (multi-threading)
- All transitive dependencies

Plus:
- Metal shader: `std_dev_filter.metallib` (GPU acceleration)
- App resources and icons
- Launch configuration

**Total bundle size:** Approximately 100-150 MB

---

## Requirements

### On Your Build Machine (the Mac where you build)

- Homebrew with all dependencies installed (see `BUILD_MACOS.md`)
- Qt 5, Boost, GSL, GStreamer, ZeroMQ, Exiv2, OpenMP
- Built FlightView app in `lv_release/liveview.app`

### On Colleague's Machines (where they run the app)

- **Nothing!** Just macOS 10.13+ and a compatible architecture
- No Homebrew
- No dependencies
- No build tools

---

## Architecture Considerations

macOS apps are architecture-specific:

| Build Machine | Output Architecture | Runs On |
|---------------|---------------------|---------|
| Apple Silicon (M1/M2/M3) | `arm64` | Apple Silicon Macs only |
| Intel Mac | `x86_64` | Intel Macs only* |

*Universal binaries (both architectures) are possible but require building twice and using `lipo` to combine them.

**Recommendation:** Build on the same architecture as your colleagues' Macs.

---

## Distribution Methods

### Method 1: DMG (Recommended)

```bash
./package_macos_app.sh
# Creates: FlightView-macOS-arm64.dmg

# Share via:
# - Email (if < 25 MB after compression)
# - File sharing service (Dropbox, Google Drive, etc.)
# - Internal network share
```

**Pros:**
- Standard macOS format
- Can include README and documentation
- Mounts as a disk for easy drag-and-drop install

**Cons:**
- Slightly larger than ZIP

### Method 2: ZIP Archive

```bash
cd lv_release
zip -r ../FlightView.zip liveview.app
```

**Pros:**
- Smaller file size
- Easy to share

**Cons:**
- Less "Mac-like" experience
- No opportunity for install instructions in the package

### Method 3: App Store (Advanced)

For wider distribution, you can:
1. Get an Apple Developer account ($99/year)
2. Code sign with a Developer ID certificate
3. Notarize the app with Apple
4. Distribute via App Store or direct download

This removes all security warnings and enables automatic updates.

---

## Verification

Test the bundle is self-contained:

```bash
# Check executable dependencies
otool -L lv_release/liveview.app/Contents/MacOS/liveview-bin

# Should ONLY show:
# - @rpath/... (bundled libraries)
# - /usr/lib/... (system libraries)
# - /System/... (system frameworks)

# No /opt/homebrew or /usr/local paths!
```

Test on a clean system:
1. Create a new user account (no Homebrew)
2. Mount the DMG
3. Drag app to Applications
4. Launch and verify all features work

---

## Troubleshooting

### "App is damaged and can't be opened"

**Cause:** Gatekeeper security check failed (unsigned app)

**Solution:**
```bash
# On the user's Mac:
xattr -cr /Applications/FlightView.app
```

Or: Right-click > Open (instead of double-clicking)

### "Library not found" errors at runtime

**Cause:** Missing dependency or incorrect library path

**Solution:**
```bash
# Check what's missing:
otool -L lv_release/liveview.app/Contents/MacOS/liveview-bin | grep -E "(homebrew|local)"

# Re-run packaging script:
./package_macos_app.sh
```

### App bundle too large

**Cause:** Unnecessary files included

**Solution:** The script already excludes debug symbols. To reduce further:
- Strip binaries: `strip -x liveview.app/Contents/MacOS/liveview-bin`
- Remove unused Qt plugins
- Compress the DMG more aggressively

### Different architectures

**Problem:** You built on M1 (arm64) but colleague has Intel (x86_64)

**Solutions:**
1. Build on the target architecture
2. Use Rosetta 2 (arm64 apps can run on Intel via emulation) - NOT recommended
3. Create a universal binary (advanced)

---

## Advanced: Universal Binary

To create a single app that works on both architectures:

```bash
# 1. Build on Intel Mac
cd build-intel
qmake CONFIG+=release
make -j8

# 2. Build on Apple Silicon Mac
cd build-arm64
qmake CONFIG+=release
make -j8

# 3. Combine with lipo
lipo -create \
    build-intel/lv_release/liveview-bin \
    build-arm64/lv_release/liveview-bin \
    -output liveview-universal

# 4. Replace binary and package
```

This requires access to both architectures and is complex. Usually better to just build for the target architecture.

---

## Summary

**✓ YES** - You can create a fully self-contained macOS app bundle  
**✓ YES** - Colleagues don't need Homebrew or any dependencies  
**✓ YES** - All dependencies reside inside the `.app` bundle  
**✓ YES** - Distribution via DMG is standard and professional  

**Use:** `./package_macos_app.sh` after building

**Share:** The resulting `.dmg` file

**Size:** ~100-150 MB (one-time download)

**Compatibility:** Same architecture only (arm64 or x86_64)

---

## References

- **Full Documentation:** `documentation/MACOS_PACKAGING.md`
- **Build Instructions:** `documentation/BUILD_MACOS.md`
- **Build Script:** `build_with_metal.sh`
- **Packaging Script:** `package_macos_app.sh`
- **Apple Documentation:** [Bundle Programming Guide](https://developer.apple.com/library/archive/documentation/CoreFoundation/Conceptual/CFBundles/Introduction/Introduction.html)

---

**Questions?** Check the full `MACOS_PACKAGING.md` documentation for detailed technical information.
