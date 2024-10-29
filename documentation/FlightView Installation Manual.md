*Last Updated: 2024-10-29*

Installing FlightView
=======================

### Table of Contents
* [Prerequisites](#part-1-prerequisites)
  * [OS](#os)
  * [Dependencies](#dependencies)
    * [Patching gstreamer](#patching-gstreamer)
* [GPU Drivers](#gpu-drivers)
* [Camera Link Drivers](#camera-link-drivers)
* [FlightView Installation](#part-2-building-and-installing-flightview)
  * [Obtain Source Code](#obtaining-the-source-code)
  * [Building the Backend](#building-the-backend-cuda-take)
  * [Building the Frontend](#building-the-frontend)

Part 1: Prerequisites
------------------------

<details open>
    <summary></summary>

### OS

FlightView has been tested on Ubuntu, Linux Mint, and Pop!OS. Most development has taken place using Linux Mint 20.2 with kernel 5.4.0-74-generic. Some drivers required by FlightView require the installed version of Linux to be running a kernel no later than 5.x. FlightView has been tested 5.4.x and 5.15.x. Ensure that the kernel version used is compatible with the hardware in use. For instance, some network devices and video cards don't have drivers available on older versions of the kernel. 

### Dependencies

To install the GPU drivers, CameraLink drivers, and FlightView software, some dependencies need to be installed first. Instructions are provided for Debian-based distributions, and the commands will need to be adjusted accordingly for non-Debian distributions. Package names may differ slightly between various operating systems.

#### Build Tools & Libraries:

```bash
sudo apt-get -y install build-essential git wget libc-dev libc6-dev gcc g++ linux-headers-$(uname -r)
sudo apt-get -y install qt5-default qt5-qmake qt5-qmake-bin libqt5svg5 libqt5svg5-dev qtcreator
sudo apt-get -y install libboost-all-dev libgsl-dev libgsl23 libgslcblas0
```
*Note: It may be necessary to reinstall the kernel headers if the OS kernel is updated.*

<details open>
<summary><strong>Using gstreamer vs. rtpnextgen</strong></summary>
RTP support is provided via two libraries: gstreamer, and rtpnextgen. If using gstreamer, additional steps are requierd to patch it for 16-big grayscale support. If using rtpnextgen, gstreamer still needs to be installed, but can be installed via apt without a patch.

gstreamer features:
* Supports IPv4 & IPv6
* Requires a patch for 16-bit grayscale

rtpnextgen features:
* Supports IPv4
* Supports faster frame rates (>3 gigabits/sec)
* Less picky about timestamps and other data
* Native 16-bit grayscale

To install required libraries **without a patch**:
```bash
sudo apt-get install -y libgstreamer1.0-0 libgstreamer1.0-dev
sudo apt-get install -y gstreamer-1.0-plugins-base gstreamer-1.0-plugins-godd gstreamer-1.0-plugins-good-dev
# gstreamer-1.0-plugins-good-dev may not be able to be located
```

#### Patching gstreamer
```bash
#Prepare the apt system for source installs

sudo sed -i '/deb-src/s/^# //' /etc/apt/sources.list
sudo apt-get update
sudo apt-get install -y apt-src


# Install the source for gstreamer1.0-plugins-good
mkdir ~/Downloads/gst
cd ~/Downloads/gst
apt-src install gstreamer1.0-plugins-good


# Patch the library
cd gst-pluginsgood1.0-16.3/gst/rtp
cp gstrtpvrawdepay.c gstrtpvrawdepay.c-orig

# Get the patch file from the flightview git repository
wget https://raw.githubusercontent.com/nasa-jpl/FlightView/refs/heads/master/utils/gst/gstrtpvrawdepay.c.patch

patch gstrtpvrawdepay.c gstrtpvrawdepay.c.patch

# Build and install the plugin
cd ~/Downloads/gst
apt-src build gstreamer1.0-plugins-good
sudo dpkg --install ./gstreamer1.0*.deb
```

</details>

#### SSH Server
It is **strongly recommended** to install and enable an SSH Server so that, if the graphics don't come up, an SSH connection can be made for diagnostics:

```bash
sudo apt-get -y install openssh-server
sudo systemctl enable openssh-server
sudo systemctl start openssh-server
```



</details>

GPU Drivers
-----------
<details open>
    <summary>
        <b>Installing the drivers</b>
    </summary>

The computer bust be using the proprietary Nvidia graphics drivers as well as a matching CUDA driver. Additionally, the CUDA Development Kit must be installed, which includes the nvcc Nvidia CUDA compiler and associated libraries. All these can be installed by using a single download. 

**Please use CUDA Development Kit version 11.6 or greater for the smoothest installation of FlightView**

While the RPM or DEB installation options will likely work, **it is recommended to use the Runfile installer for the installation**, as it will not be affected by OS updates. Runfile install instructions are found in section 5 of the linked documentation.

### Remove prior version(s) of CUDA and Nvidia Drivers

*Described in greater detail in section 2.6 in the linked Nvidia documentation*.

```bash
# Uninstalling DEB-based install
sudo apt-get --purge remove <package_name>

# Uninstalling .run file-based install
sudo /usr/local/cuda-X.Y/bin/cuda-uninstaller
```

Instructions and links to appropriate installers can be found at: [Nvidia CUDA Linux Installation Documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

While the Runfile installer provides all appropriate options, ensure that, regardless of installation method, the CUDA Development Kit, Proprietary Video Driver, and CUDA Driver are installed. Also ensure that non-proprietary drivers are blacklisted from loading.

### Blacklisting non-proprietary drivers

```bash
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo update-initramfs -u
sudo update-grub
```
</details>

Camera Link Drivers
-------------------
*FlightView currently only supports EDT PDV software prior to version 6.0*



<details open>
    <summary>
        <b>Install EDT PDV software</b>
    </summary>

Download and install the EDT PDV software from [here](https://edt.com/file-category/pdv/), following the installation instructions provided on the site. As of this writing, the only supported Linux installer for versions prior to 6.0 is for version 5.6.8.0 and is a Runfile installer. Accept the default options in the installer.

*Typically, the installer will attempt to build all the EDTpdv utilities. Some of these may fail, but generally the most important ones will build without too much trouble.*

Recent EDT drivers need these additional steps:
```bash
sudo chmod -R o+r /opt/EDTpdv # Allows global read of all files in the directory
sudo mv /opt/EDTpdv/version /opt/EDTpdv/version-orig
```

The drivers include a kernel module, which can be manually re-built with the following commands:
```bash
cd /opt/EDTpdv/module
sudo make # make includes install
sudo update-initramfs -u
sudo update-grub
```
*This is necessary after updating the OS if the kernel is updated. Ensure that the correct linux-headers package is installed after OS udpates before rebuilding this kernel module.*

### Initialize the Camera Link Port
To initialize the camera link port, the `initcam` program is used with a camera link config file. By default, a large set of camera link config files are provided in `/opt/EDTpdv/camera_config`. These plaintext files may be used as a starting point for building a custom config file.

Here is an example for a 640x481 14-bit camera:
```json
camera_class:       "Imaging Spectrometer"
camera_model:       "Camera Link 14-bit"
camera_info:        "640x481 HSI"
width:              640
height:             481
depth:              14
extdepth:           14
CL_CFG_NORM:        02
continuous:         1
MODE_CNTL_NORM:     00
```

Here is an example for a 1280x328 16-bit camera:
```json
camera_class:       "Imaging Spectrometer"
camera_model:       "FPA"
camera_info:        "1280x480 16-bit HSI"
width:              1280
height:             328
depth:              16
extdepth:           16
rbtfile:            aiagcl.bit
continuous:         1
CL_CFG_NORM:        00
htaps:              4
serial_baud:        19200
```

A Camera Link port can be initialized like this:
```bash
/opt/EDTpdv/initcam -f /opt/EDTpdv/camera_config/spectrometer.cfg
```

It is recommended to build a shell script for this task and then call it from a desktop launcher, with the terminal opening to show any error messages.

Example shell script file `init_ngis.sh`:
```bash
#!/bin/bash
CFGFILE=/opt/EDTpdv/camera_config/NGIS.cfg
echo "Running $0 for config file $CFGFILE ..."
/opt/EDTpdv/initcam -f $CFGFILE
echo "Done! Closing window in 2 seconds."
sleep 2;
```

Example desktop file `init_camera.desktop`:

```ini
[Desktop Entry]
Name=initcam NGIS
Exec=/home/username/bin/init_ngis.sh
Comment=640x481
Terminal=true
Icon=cinnamon-panel-launcher
Type=Application
```

To verify the port was successfully initialized, run `/opt/EDTpdv/take`, which attempts to acquire one frame from the camera link port. The command will typically return with one of the following:
* `EDT /dev/pdv0 open failed. pdv_open(pdv0_0): No such device or address`
  * This means that either the kernel module did not load, or the card has a problem or isn't physically installed. Check the output of `lsmod` and `dmesg` for details.
* `pdv0: Invalid image size. Make sure device has been initialized.`
  * Run `initcam`
* `1 images 1 timeouts 0 overruns 0 bytes`
  * The port was initialized, but the camera was not running
* `1 images 0 timeouts 0 overruns`
  * The port was initialized, and the camera returned a frame without timing out. This means the camera link system is ready for use.

*The return value is generally `1` upon error and `0` upon success.*

</details>

Part 2: Building and Installing FlightView
------------------------------------------

### Obtaining the Source Code
```bash
cd ~/Documents/
git clone https://github.com/nasa-jpl/FlightView.git # Optionally, use the flightview branch for a more stable release

cd ~/FlightView
git submodule init
git submodule update
```

### Building the backend (cuda-take)
Verify that `nvcc` is found with `which`. If it is not found, add the appropriate bin directory to the path before proceeding. 
```bash
# Example output
which nvcc
 /usr/local/cuda/bin/nvcc

# To add the nvcc path to $PATH:
# For runfile-based installs:
export PATH=$PATH:/usr/local/cuda/bin # cuda may also have a version number, such as cuda-11.6

# For apt-based installs:
export PATH=$PATH:/usr/cuda/bin
```

If `nvcc`'s is version is <11.6 (check with `nvcc -V`), extra steps are required:
* Edit the Makefile at `cuda_take/Makefile` and uncomment & edit an NVCCFLAGS line which matches the appropriate card. Card type can be queried with `nvidia-smi -L`. 
  * To find the matching specification for the card, use [this](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) site.

After verifying nvcc, build cuda_take:
```bash
cd cuda_take # Current directory should be ~/Documents/FlightView/cuda_take
make -j
```

Common errors:
* Unknown architecture
  * Refer to above instructions regarding NVCCFLAGS in the Makefile
* nvcc: not found
  * Refer to the above instructions regarding adding NVCC to $PATH

### Building the frontend
```bash
cd ~/Documents/FlightView
mkdir build
cd build

# Run qmake to configure the Makefile
# For a release build (for deployment):
qmake ../liveview.pro

# For a debug build (for testing):
qmake CONFIG+=debug ../liveview.pro

# Build FlightView:
make -j
```

After the build completes, a folder within `build` named `lv_release` will be created. Within that directory is the build binary, a sample `.desktop` file, and a `liveview.png` to be used for the desktop file icon. Move these to the following locations:
```bash
cd lv_release
sudo cp ./liveview /usr/local/bin/liveview && sudo chmod +x /usr/local/bin/liveview
sudo cp ./liveview_icon.png /usr/share/pixmaps/liveview.png
sudo cp ./liveview.desktop /usr/share/applications/liveview.desktop
```
*liveview.dekstop may need to have execute permissions set by right-clicking and selecting `Allow Launching` or a similar option. This is different from setting the execute bit on a file.*