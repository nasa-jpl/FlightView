#!/bin/bash
# Convert liveview.png to liveview.icns for macOS app bundle
# This script creates an .icns file with multiple resolutions

set -e

PNG_FILE="liveview.png"
ICONSET_DIR="liveview.iconset"

if [ ! -f "$PNG_FILE" ]; then
    echo "Error: $PNG_FILE not found!"
    exit 1
fi

echo "Creating iconset directory..."
mkdir -p "$ICONSET_DIR"

echo "Generating icon at multiple resolutions..."
# Generate all required sizes for macOS icons
sips -z 16 16     "$PNG_FILE" --out "$ICONSET_DIR/icon_16x16.png"
sips -z 32 32     "$PNG_FILE" --out "$ICONSET_DIR/icon_16x16@2x.png"
sips -z 32 32     "$PNG_FILE" --out "$ICONSET_DIR/icon_32x32.png"
sips -z 64 64     "$PNG_FILE" --out "$ICONSET_DIR/icon_32x32@2x.png"
sips -z 128 128   "$PNG_FILE" --out "$ICONSET_DIR/icon_128x128.png"
sips -z 256 256   "$PNG_FILE" --out "$ICONSET_DIR/icon_128x128@2x.png"
sips -z 256 256   "$PNG_FILE" --out "$ICONSET_DIR/icon_256x256.png"
sips -z 512 512   "$PNG_FILE" --out "$ICONSET_DIR/icon_256x256@2x.png"
sips -z 512 512   "$PNG_FILE" --out "$ICONSET_DIR/icon_512x512.png"
sips -z 1024 1024 "$PNG_FILE" --out "$ICONSET_DIR/icon_512x512@2x.png"

echo "Converting iconset to icns..."
iconutil -c icns "$ICONSET_DIR" -o liveview.icns

echo "Cleaning up..."
rm -rf "$ICONSET_DIR"

echo "✓ Successfully created liveview.icns"
ls -lh liveview.icns
