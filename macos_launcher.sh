#!/bin/bash
# FlightView macOS Launcher Script
# This script reads default arguments from a config file and launches the actual binary

# Get the directory where this script lives
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES="$DIR/../Resources"
CONFIG="$RESOURCES/launch_config.txt"

# Read default args from config file (single line)
if [ -f "$CONFIG" ]; then
    DEFAULT_ARGS=$(cat "$CONFIG")
else
    # Fallback if config file doesn't exist
    DEFAULT_ARGS=""
fi

# Launch the actual binary with defaults + any additional args passed to this script
exec "$DIR/liveview-bin" $DEFAULT_ARGS "$@"
