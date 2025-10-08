#!/usr/bin/env python3
"""
Shared Memory Frame Monitor
Reads and displays metadata and frame statistics from /liveview_image shared memory segment
"""

import sys
import time
import struct
import ctypes
import mmap
from posix_ipc import SharedMemory
import numpy as np

# Constants from shm_image.h
SHM_FRAME_BUFFER_SIZE = 10
SHM_FILENAME_BUFFER_SIZE = 256

# Status byte values
SHM_STATUS_READY = 31
SHM_STATUS_WAITING = 28
SHM_STATUS_INITALIZING = 26
SHM_STATUS_CLOSED = 24
SHM_STATUS_ERROR = 13

STATUS_NAMES = {
    31: "READY",
    28: "WAITING",
    26: "INITIALIZING",
    24: "CLOSED",
    13: "ERROR"
}


class ShmSharedDataStruct(ctypes.Structure):
    """
    Python representation of the C struct shmSharedDataStruct
    Must match the memory layout of the C struct exactly
    """
    _fields_ = [
        ("statusByte", ctypes.c_char),
        ("fps", ctypes.c_float),
        ("recordingDataToFile", ctypes.c_bool),
        ("counter", ctypes.c_uint16),
        ("writingFrameNum", ctypes.c_int),
        ("bufferSizeFrames", ctypes.c_int),
        ("frameWidth", ctypes.c_int),
        ("frameHeight", ctypes.c_int),
        ("takingDark", ctypes.c_bool),
        ("frameTime", ctypes.c_uint64 * SHM_FRAME_BUFFER_SIZE),
        ("lastFilename", ctypes.c_char * SHM_FILENAME_BUFFER_SIZE),
    ]


class ShmFrameMonitor:
    def __init__(self, shm_name="/liveview_image"):
        self.shm_name = shm_name
        self.shm = None
        self.mmap_obj = None
        self.metadata = None
        self.frame_buffer_offset = None
        self.last_frame_time = None  # Track previous frame time for age calculation
        
    def connect(self):
        """Connect to the shared memory segment"""
        try:
            # Open the shared memory object
            self.shm = SharedMemory(self.shm_name)
            print(f"Connected to shared memory: {self.shm_name}")
            
            # Memory map the shared memory
            self.mmap_obj = mmap.mmap(self.shm.fd, 0)
            print(f"Shared memory size: {len(self.mmap_obj)} bytes")
            
            # Map the metadata struct
            self.metadata = ShmSharedDataStruct.from_buffer(self.mmap_obj)
            
            # Calculate frame buffer offset (starts after metadata struct)
            self.frame_buffer_offset = ctypes.sizeof(ShmSharedDataStruct)
            
            return True
        except Exception as e:
            print(f"Error connecting to shared memory: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False
    
    def disconnect(self):
        """Disconnect from shared memory"""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None
        if self.shm:
            self.shm.close_fd()
            self.shm = None
            print("Disconnected from shared memory")
    
    def get_metadata_dict(self):
        """Extract metadata as a dictionary"""
        if not self.metadata:
            return None
        
        status_val = ord(self.metadata.statusByte)
        status_name = STATUS_NAMES.get(status_val, f"UNKNOWN({status_val})")
        
        # Decode filename, stopping at null terminator
        filename = self.metadata.lastFilename.decode('utf-8', errors='ignore')
        filename = filename.split('\x00')[0]  # Stop at null terminator
        
        return {
            "statusByte": status_name,
            "fps": self.metadata.fps,
            "recordingDataToFile": self.metadata.recordingDataToFile,
            "counter": self.metadata.counter,
            "writingFrameNum": self.metadata.writingFrameNum,
            "bufferSizeFrames": self.metadata.bufferSizeFrames,
            "frameWidth": self.metadata.frameWidth,
            "frameHeight": self.metadata.frameHeight,
            "takingDark": self.metadata.takingDark,
            "lastFilename": filename if filename else "(none)"
        }
    
    def get_frame(self, frame_index):
        """
        Retrieve a specific frame from the buffer as a numpy array
        
        Args:
            frame_index: Index of frame to retrieve (0 to bufferSizeFrames-1)
            
        Returns:
            numpy array of shape (frameHeight, frameWidth) with dtype uint16
        """
        if not self.metadata:
            return None
        
        frame_width = self.metadata.frameWidth
        frame_height = self.metadata.frameHeight
        
        if frame_index < 0 or frame_index >= self.metadata.bufferSizeFrames:
            print(f"Invalid frame index: {frame_index}", file=sys.stderr)
            return None
        
        # Calculate the byte offset for this frame
        frame_size_pixels = frame_width * frame_height
        frame_size_bytes = frame_size_pixels * 2  # uint16_t = 2 bytes
        frame_offset = self.frame_buffer_offset + (frame_index * frame_size_bytes)
        
        # Extract frame data as numpy array
        try:
            # Create a numpy array view directly from mmap buffer
            frame_data = np.frombuffer(
                self.mmap_obj,
                dtype=np.uint16,
                count=frame_size_pixels,
                offset=frame_offset
            )
            
            # Reshape to 2D array (create a copy to avoid mmap issues)
            frame = frame_data.reshape((frame_height, frame_width)).copy()
            return frame
        except Exception as e:
            print(f"Error reading frame {frame_index}: {e}", file=sys.stderr)
            return None
    
    def get_latest_frame(self):
        """
        Get the most recent complete frame (not the one being written)
        
        Returns:
            numpy array of the latest frame
        """
        if not self.metadata:
            return None
        
        writing_frame = self.metadata.writingFrameNum
        buffer_size = self.metadata.bufferSizeFrames
        
        # Read the frame before the one being written
        latest_frame_index = (writing_frame - 1) % buffer_size
        
        return self.get_frame(latest_frame_index)
    
    def print_metadata(self):
        """Print formatted metadata information"""
        meta = self.get_metadata_dict()
        if not meta:
            print("No metadata available")
            return
        
        print("\n" + "="*60)
        print(f"Status: {meta['statusByte']}")
        print(f"FPS: {meta['fps']:.2f}")
        print(f"Frame Counter: {meta['counter']}")
        print(f"Writing Frame #: {meta['writingFrameNum']}")
        print(f"Buffer Size: {meta['bufferSizeFrames']} frames")
        print(f"Frame Dimensions: {meta['frameWidth']} x {meta['frameHeight']}")
        print(f"Recording: {'YES' if meta['recordingDataToFile'] else 'NO'}")
        print(f"Taking Dark: {'YES' if meta['takingDark'] else 'NO'}")
        print(f"Last Filename: {meta['lastFilename']}")
        
        # Display frame times for available frames
        print(f"\nFrame Times:")
        
        # Find the most recent frame (highest timestamp)
        max_time = 0
        for i in range(meta['bufferSizeFrames']):
            frame_time = self.metadata.frameTime[i]
            if frame_time > max_time:
                max_time = frame_time
        
        # Calculate age relative to most recent frame
        for i in range(meta['bufferSizeFrames']):
            frame_time = self.metadata.frameTime[i]
            if frame_time > 0:
                age_ms = max_time - frame_time
                writing_marker = " <-- WRITING" if i == meta['writingFrameNum'] else ""
                freshness = "LATEST" if frame_time == max_time else f"{age_ms:6d} ms old"
                print(f"  Frame {i}: {freshness:20s} (timestamp: {frame_time}){writing_marker}")
        print("="*60)
    
    def analyze_latest_frame(self):
        """Analyze and print statistics for the latest frame"""
        frame = self.get_latest_frame()
        
        if frame is None:
            print("Could not retrieve latest frame")
            return
        
        # Calculate statistics
        mean_val = np.mean(frame)
        std_val = np.std(frame)
        min_val = np.min(frame)
        max_val = np.max(frame)
        
        print(f"\nLatest Frame Statistics:")
        print(f"  Mean pixel value: {mean_val:.2f}")
        print(f"  Std deviation: {std_val:.2f}")
        print(f"  Min value: {min_val}")
        print(f"  Max value: {max_val}")


def main():
    monitor = ShmFrameMonitor()
    
    if not monitor.connect():
        print("Failed to connect to shared memory. Is the producer running?")
        return 1
    
    try:
        print("\nMonitoring shared memory segment...")
        print("Press Ctrl+C to exit\n")
        
        while True:
            # Print metadata
            monitor.print_metadata()
            
            # Analyze latest frame
            monitor.analyze_latest_frame()
            
            # Wait 1 second before next update
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError during monitoring: {e}", file=sys.stderr)
        return 1
    finally:
        monitor.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
