# FlightView
2014-2024 Jet Propulsion Laboratory, AVIRIS lab

Author/Maintainer: Elliott Liggett

Prior Authors: Jackie Ryan and Noah Levy

This version of LiveView uses NVIDIA's CUDA toolkit for real-time processing, which is covered in the following IEEE publication: 

[LiveView: A new utility for real-time calibration of focal plane arrays using commodity hardware](https://ieeexplore.ieee.org/abstract/document/7500690)

## Overview
*LiveView* (aka *FlightView*) is graphical program which displays real-time data from imaging spectrometers. Liveview supports high data rate (in excess of 3 gigabits/sec) acquisition over Ethernet (typically 10G fiber optic) with the RTP protocol, as well as over CameraLink (EDT cards are supported). 

Standard deviation is computed in real-time using the GPU via CUDA. 

Plots are implemented using the [QCustomPlot](http://www.qcustomplot.com) library, which generates live color maps, bar graphs, and line graphs within the Qt C++ environment.

A real-time feed of the current frame data is made available over a Shared Memory Segment. 

A TCP/IP socket allows for rudimentry control such as saving, dark averaging, and so on. 

The AVIRIS-III flight system uses an Atlans A7 for GNSS/IMU data, and LiveView connects to this device to save the binary data and present it to the operator for verification. 

FlightView is used or has been used on the following instruments for I&T and/or aircraft Flight Operations:

- GAO (Global Airborne Observatory, formerly CAO)
- NGIS NIS-1, NIS-2, and NIS-3
- AVIRIS Next Gen
- AVIRIS-III (*)
- AVIRIS-IV
- EMIT
- CPM

(*) = Flight Operations

## Installation

Please see the documentation folder for the latest detailed directions.

## System Requirements
### Minimum Requirements:
- Fast, modern, multi-core CPU. 
- NVIDIA Graphics Card with 512MB VRAM, i.e, GTX 560 (adjust the size of GPU_BUFFER_SIZE in constants.h of cuda_take and make a clean compile to change this setting)
- Linux OS such as Debian, Ubuntu, or Mint
- Proprietary NVIDIA CUDA-capable driver version 10.1 or greater (>=12.1 preferred). 

## Contact Us!

Elliott Liggett:

JPL email: [Elliott.H.Liggett@jpl.nasa.gov](mailto:Elliott.H.Liggett@jpl.nasa.gov)
