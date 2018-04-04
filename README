
# LIVEVIEW 2.8
2014-2018 Jet Propulsion Laboratory, AVIRIS lab
@authors: Jackie Ryan, and Elliott Liggett, with past work from Noah Levy

## Contact Us!
Jackie's contact info:
JPL email: [Jacqueline.Ryan@jpl.nasa.gov](mailto:Jacqueline.Ryan@jpl.nasa.gov)
For urgent inquiries: please email for phone number.

Elliott Liggett (secondary contact):
JPL email: [Elliott.Liggett@jpl.nasa.gov](mailto:Elliott.Liggett@jpl.nasa.gov)

## Overview
*liveview2* is a Qt frontend GUI for cuda_take, it displays focal plane data and basic analysis (such as the std. dev, dark subtraction, FFT, Spectral Profile, and Video Savant*-like* playback). Plots are implemented using the [QCustomPlot](http://www.qcustomplot.com) library, which generates live color maps, bar graphs, and line graphs within the Qt C++ environment.

## System Requirements
### Minimum Requirements:
Quad-core processor (x86 Architecture)
8GB RAM
NVIDIA Graphics Card with 512MB VRAM, i.e, GTX 560 (adjust the size of GPU_BUFFER_SIZE in constants.h of cuda_take and make a clean compile to change this setting)
Linux OS

### Recommended Settings:
32-core CPU array
16GB RAM (depending on the imaging application, memory allocation will increase)
2 SSLI NVIDIA Graphics Card with 1GB VRAM (or one high-end GPU)
Server-based RAID array (for large data saves)

## Installation
*cuda_take* must be installed in the same parent directory as *liveview2*. The build folder for LiveView is contained within the source folder itself. Although Qmake does not support this feature, we have ways >:)

1. A good practice is to make a "projects" folder within your home folder:
```
	you@computer: ~$ mkdir projects
	you@computer: ~$ cd projects
```
*Perform cuda_take installation now. Please refer to the instructions in the readme for [cuda_take](https://github.com/NASA-JPL/cuda_take)*

2. Clone a copy of *liveview2* into the "projects" folder:
```
	you@computer: ~/projects$ git clone https://github.com/NASA-JPL/liveview
```

3. If you have completed both clones successfully, entering "ls -l" now will show the presence of both *liveview2* and *cuda_take* directories within the same "projects" directory
```
	you@computer: ~/projects$ ls -l
```

4. Enter the *liveview2* directory and either open the .pro file in qtcreator and build it with the qt IDE or use:
```
	you@computer: ~/projects$ cd liveview2
	you@computer: ~/projects/liveview2$ qmake
	you@computer: ~/projects/liveview2$ make -j
```

5. (Optional - for developers) Developers may sometimes need to delete the build directory and recompile if changes to certain headers are not being recognized at compile time. This is vey important when reconfiguring cuda_take to work for different types of hardware.

## Design
*liveview2* is designed to be somewhat modular; the backend is queried by a single class, frame_worker, which is run outside of the event thread. All of the widgets for viewing data are implemented as their own classes and are embedded in the mainwindow class via QTabWidget.

Each widget has its own renderTimer which emits a local signal to replot the data at a framerate defined in settings.h. The queuing thread that runs frame_worker does not periodically send signals to the event thread, this is because this had terrible performance in real life.

## Keyboard Controls
### General
* p - Toggle the Precision Slider
* m - Toggle the Dark Subtraction Mask (if one is present)
* , - Begin recording Dark Frames
* . - Stop recording dark frames

### For frame views (raw image, dark subtraction, standard deviation)
* left click - profile the data at the specified coordinate
* esc - reset the crosshairs
* d - Toggle display of the crosshairs

### For the Histogram Widget
* r - reset the range of the display. Zooming may make it difficult to return to the original scale of the plot.

### For the Playback Widget
* *drag and drop onto the viewing window* - load the selected file. WARNING: Any filetype is accepted. This means if the filetype is not data, garbage will be displayed in the viewing window. This is done by design.
* s - Stop playback and return to the first frame
return - Play/Pause
* f - Fast Forward. Multiple presses increase the fast forward multiplier up to 64x faster.
* r - Rewind. Multple presses inreas the rewind multiplier up to 64x faster.
* a - Move back one frame. Only works when playback is paused.
* d - Move forward one frame. Only works when playback is paused.

## Components of this repository
* **Main -> main.cpp**: The version information is displayed, then the splash screen (which also has version info), and then the backend and mainwindow are launched.

* **MainWindow -> mainwindow.cpp**: The main window for the Qt Application is established in here. The GUI for liveview2 is separated into two main parts - the current view widget, and the controls box. Certain controls are shared between all widgets, with minor changes depending on the context of the data; these buttons are contained within the ControlsBox GUI class. All the widgets have controls which specific to themselves. These widgets are entered into the QTabWidget. Additionally, all keyboard controls for liveview2 are mapped in mainwindow.cpp 

* **Settings -> settings.h**, **Frame Meta -> frame_c_meta.h**, **Image Types -> image_type.h**:
These headers establish various constants used by the software.
The frame skip, target (ideal) framerate, and frame display period can be set from settings.h
Some types from the backend frame struct are established as metatypes in frame_c_meta.h
The image_type macro is defined in image_type.h; this macro is used by the GUI to recognize the type of data currently being displayed. In the current version, some values of this type are deprecated. To avoid compiler warnings when using a switch statement with image_type, add "default: break;"

* **FrameWorker -> frame_worker.cpp**: The FrameWorker class contains backend and analysis information that must be shared between classes in the GUI. Structurally, frameWorker is a worker object tied to a QThread started in main. The main event loop in this object may therefore be considered the function handleNewFrame() which periodically grabs a frame from the backend and checks for asynchronous signals from the filters about the status of the data processing. A frame may still be displayed even if analysis functions associated with it time out for the loop. In general, the frameWorker is the only object with a copy of the take_object and should exclusively handle communication with cuda_take.

* **ConrolsBox -> controlsbox.cpp**: The ControlsBox is a wrapper GUI class which contains the (mostly) static controls between widgets.

*Detailed information for developers*: After establishing the buttons in the constructor, the class will call the function tabSlotChanged(int index) to establish widget-specific controls and settings. For instance, all profile widgets and FFTs make use of the Lines To Average slider rather than the disabled Std Dev N slider. As Qt does not support a pure virtual interface for widgets, each widget must make a connection to its own version of updateCeiling(int c), updateFloor(int f), and any other widget specfic action within its case in tabSlotChanged. The beginning of this function specifies the behavior for when tabs are exited - all connections made must be disconnected.

* **SaveServer -> saveserver.cpp**: liveview2 offers support for saving frames from a remote client. A server is established when the software is loaded that listens on the address of the current host, at port 65000. This information is displayed in the bottom left corner of the window. Clients may connect to the server at using "telnet $IP_ADDRESS 65000" and issuing commands, the old fashioned insecure way. This functionality was developed on secure internal networks, so no security concerns were implemented in the design. It is highly recommended, however, to use a Qt client to connect and issue commands to the server instead. A python client called "LiveView_client.py" has been included in the utils folder of this repository, which provides an API for accessing the basic functionalities of the frame saving. 

* **PreferenceWindow -> pref_window.cpp**: The Preference Window offers control over the hardware conditions of the current camera. Many of the command options in this window directly affect the raw data as it is received at the back end. Chroma Pixel Remapping may be turned on or off for cameras with chroma geometry (1280x480 resolution). On other cameras, this option is disabled. Options for 14- and 16-bit bright-dark swapping are also included. All data arriving on the data bus will be inverted by the specified factor. As a sanity check, the expected data range is displayed above this option. The assumed camera type and geometry are listed at the top of the window. Additionally, the first or last row data in the raw image may be excluded from the image. This option only applies to linear profiles. Log files are not currently an implemented feature.

* **frameview_widget -> frameview_widget.cpp**: The frame views diplay color maps of the live data as it comes from the camera. Live View displays the raw image without any analysis (other than harware specific remappings). Dark Subtraction displays the live image with a mask of recorded dark frames applied. You will need to record dark frames before you are able to see the dark subtraction. The standard deviation view displays the pixel-wise standard deviation over the specified window, with a maximum of 499 frames.

* **histogram_widget -> histogram_widget.cpp**: The histogram displays the spacial frequency of the standard deviations calculated for the standard deviation frame. Each bar represents a range of sigma (in DN) that the individual pixel sigmas are binned within. These data are plotted with a logarithmic x-axis. Scrolling on the mouse wheel will zoom in and out. The standard viewing scale omits dead pixel bars, but these can be viewed by zooming out.

* **profile_widget -> profile_widget.cpp**: The profile widget displays a horizontal or vertical "slice" of image data. In the Vertical Mean and Horizontal Mean, all columns or rows, respectively, are averaged and plotted as a single line for each frame. In the Vertical and Horizontal Crosshair Profiles, the number of rows or columns to be averaged may be selected using the slider in the controls box. 1 line represents a plot of only the row or column specified in the plot title. The maximum number of lines represents a plot of the repective mean profile. Additionally, there is an option to select and view the value of data points on charts by double-clicking. The y-value of that point on the plot will be displayed with respect to an x-coordinate.

* **fft_widget -> fft_widget.cpp**:  The FFT Widget displays the Fourier Transform of three different types of data in a bar graph format. There are three sampling types which may be selected using the buttons underneath the plot. For information on the analysis, please see the cuda_take documentation on the mean filter and FFT. In vertical crosshair FFTs, the number of columns to be binned may be selected using the slider in the controls box. In the tap profile, the tap that is being sampled may be selected using the numbered display next to the button.

* **playback_widget -> playback_widget.cpp**:  The Playback Widget offers self-contained playback of data. It is assumed that the data being viewed has the same geometry as the currently loaded camera hardware. Playback is handled using VCR-like controls. Users may drag and drop data onto the viewing window to load a file, or use the folder icon on the left side of the window.


## Known Bugs & Issues
* When selecting a save file, I had no way of checking permissions. If the user selects a file that they cannot write to, the backend will hang and the program will crash.

* When viewing the Plane Mean FFT, there will occasionally be discontinuities in the buffer that introduce a strong sinc^2 signal. Wait about 5 seconds for the discontinuity to filter out. This is caused by large changes to the frame data or mean calculation during runtime.

* Playback Mode contains an option to drag and drop files onto the viewing screen when loading. However, it does not check if the file is of a valid MIME type. If invalid data files are loaded in, the widget will display garbage data.
