///////////////////////////////////////
///									///
///		LIVEVIEW UTILITIES			///
///									///
///////////////////////////////////////


Contents of this folder:

	doc = The documentation for LiveView
	doc_latex = Latex version of the documentation
	saveClient = prototype C++ GUI client for LiveView networking
	liveview_client.py = Python API for LiveView networking


How to use doc:

	Inside the doc folder, search for and open a file named index.html
	This html document contains the doxygen output. When writing new doxygen-style
	comments into the code, the changes must be updated to the documentation
	using the command:

	doxygen livedox

	-or-

	doxygen cuda_dox (when using cuda_take)


saveClient:

	saveClient contains a small GUI application which contains the option to specify 
	a file path for saving frames. It will automatically attempt to save 10 frames in
	LiveView, connecting on the address of AVIRIS-CAL at port 65000. This code is currently
	in alpha and will need to be edited for use as an API.

liveview_client.py:

	The python client code offers a much more modular coding experience for creating applications
	which can run tests using the LiveView software. To use this code, copy the client code file
	into the same folder as the application code and use "from liveview_client import saveClient".
	Next, create an instance of the object using "xxx = saveClient("ip_address", port_number)" where
	xxx is any name, and "ipAddress" is the address of the computer LiveView is listening on and
	portNumber is the port. This information is listed in the lower left-hand corner of the
	LiveView window.

	The following methods are defined in the API:

	requestFrames(self, file_name, num_frames)
	Requests LiveView to save a number of frames equal to num_frames into
	the file named file_name. When specifying file_name, please specify the
	path and extension in the name as well, e.g, "/home/user/test_file.raw"
	LiveView will not warn about overwriting files when accepting network
	commands, so users must ensure that their code does not unintentionally
	overwrite files.

	requestStatus(self)
	returns: (frames_left, fps)
	Requests LiveView to send the status of the current frame save operation.
	The return values are frames_left which specifies the number of frames left to save in
	the current operation, and the current backend framerate, fps.

How to change harware types (proprietary code):

	To switch to different type of data link, the following actions must be taken:

	1. Go into the cuda_take/include/ folder, and open the file take_object.hpp
	2. Search for the Harware Macros section, uncomment the macro to be used and uncomment the
		other macro. E.g;
			#define EDT
			//#define OPALKELLY
	3. Go to the Makfile in cuda_take, and change the HARDWARE macro to specify which type of harware to compile for.
		Currently special instructions are provided for Opal Kelly devices but all other data connections (EDT) have default behavior.
	4. Recompile a clean version of cuda_take while in the cuda_take directory using the following commands:
			make clean
			make -j
	5. In the parent directory, delete the build directory for liveview. (The "liveview-build-desktop..." directory)
	6. Re-build the liveview project in qtcreator.

