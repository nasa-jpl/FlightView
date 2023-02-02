#ifndef TAKEOBJECT_HPP_
#define TAKEOBJECT_HPP_

//standard includes
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

//multithreading includes
#include <atomic>
#include <boost/shared_array.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <pthread.h>
#include <mutex>
#include <gsl/gsl_statistics_uint.h>
#include <gsl/gsl_statistics.h>
//#include <boost/atomic.hpp>

//custom includes
#include "frame_c.hpp"
#include "std_dev_filter.hpp"
#include "chroma_translate_filter.hpp"
#include "dark_subtraction_filter.hpp"
#include "mean_filter.hpp"
#include "camera_types.h"
#include "cameramodel.h"
#include "xiocamera.h"
#include "constants.h"
#include "safestringset.h"
#include "takeoptions.h"
#include "fileformats.h"
#include "rtpcamera.hpp"

//** Harware Macros ** These Macros set the hardware type that take_object will use to collect data
#define EDT


//** Debug Macros **
//#define RESET_GPUS // will reset the GPU hardware on closing the program
//#define VERBOSE  // sets whether or not to compile the program with some debugging diagnostics

//Handles a corner case for the lines that display the version author. If no HOST or UNAME can be
// provided by the OS at compile time, this will display instead.
#ifndef HOST
#define HOST "unknown location"
#endif
#ifndef UNAME
#define UNAME "unknown person"
#endif

using std::string;

static const bool CHECK_FOR_MISSED_FRAMES_6604A = false; // toggles the presence or absence of the "WARNING: MISSED FRAME X" line

#define meanDeltaSize (20)

class take_object {
    PdvDev * pdv_p = NULL;
    unsigned int channel;
    unsigned int numbufs;
    unsigned int filter_refresh_rate;


    bool closing = false;
    bool grabbing = true;
    bool runStdDev = true;

    boost::thread cam_thread; // this thread controls the data collection
    boost::thread reading_thread; // this is used for file reading in the XIO camera.
    boost::thread::native_handle_type cam_thread_handler;
    boost::thread::native_handle_type reading_thread_handler;

    boost::thread rtpAcquireThread; // copy from RTP stream into buffer
    boost::thread rtpCopyThread; // copy from buffer into currFrame
    boost::thread::native_handle_type rtpAcquireThreadHandler;
    boost::thread::native_handle_type rtpCopyThreadHandler;

    int pdv_thread_run = 0;
    bool cam_thread_start_complete=false; // added by Michael Bernas 2016

	unsigned int size;
    int lastfc;

    //frame dimensions
    frame_c* curFrame;
    unsigned int dataHeight;
    unsigned int frHeight;
    unsigned int frWidth;

    //Filter-specific variables
	int std_dev_filter_N;

    std_dev_filter* sdvf;
    int meanStartRow, meanHeight, meanStartCol, meanWidth; // dimensions used by the mean filter
    int lh_start, lh_end, cent_start, cent_end, rh_start, rh_end; // VERT_OVERLAY

    //frame saving variables
    boost::thread saving_thread; // this thread handles the frame saving, as saving frames should not cause data collection to suspend
	//unsigned int save_count;
    bool do_raw_save;
	bool saveFrameAvailable;
	uint16_t * raw_save_ptr;

public:
    take_object(int channel_num = 0, int number_of_buffers = 64,
                int filter_refresh_rate = 10, bool runStdDev = true);
    take_object(takeOptionsType options, int channel_num = 0, int number_of_buffers = 64,
                int filter_refresh_rate = 10, bool runStdDev = true);
    virtual ~take_object();
    void initialSetup(int channel_num = 0, int number_of_buffers = 64,
                      int filter_refresh_rate = 10, bool runStdDev = true);
    void start();
    void changeOptions(takeOptionsType options);
    void setReadDirectory(const char* directory);
    camControlType* getCamControl();
    dark_subtraction_filter* dsf;
    camera_t cam_type;
    frame_c * frame_ring_buffer;
    unsigned long count = 0; // running frame counter
    int xioCount = 0; // counter for each set of xio files.
    uint16_t* prior_temp_frame = NULL;
    int getMicroSecondsPerFrame();

    //Frame filters that affect everything at the raw data level
    void setInversion(bool checked, unsigned int factor);
    void paraPixRemap(bool checked);

    //DSF mask functions
	void startCapturingDSFMask();
	void finishCapturingDSFMask();
	void loadDSFMask(std::string file_name);
    void loadDSFMaskFromFramesU16(std::string file_name, fileFormat_t format);
    bool dsfMaskCollected;
    bool useDSF = false;

    // Std Dev Filter functions
    void setStdDev_N(int s);
    void toggleStdDevCalculation(bool enabled);

    // Mean filter functions
    void updateVertRange(int br, int er);
    void updateHorizRange(int bc, int ec);
    void updateVertOverlayParams(int lh_start, int lh_end,\
                                 int cent_start, int cent_end,\
                                 int rh_start, int rh_end);
    void changeFFTtype(FFT_t t);

    // Frame saving functions
    void startSavingRaws(std::string raw_file_name, unsigned int frames_to_save, unsigned int num_avgs_save);
	void stopSavingRaws();
    //void panicSave(std::string);
    std::list<uint16_t *> saving_list;
	std::atomic <uint_fast32_t> save_framenum;
	std::atomic <uint_fast32_t> save_count;
	unsigned int save_num_avgs;

    //Getter functions / variables
    unsigned int getDataHeight();
    unsigned int getFrameHeight();
    unsigned int getFrameWidth();
    bool std_dev_ready();
    std::vector<float> * getHistogramBins();
    FFT_t getFFTtype();

private:
    void pdv_loop();
    void fileImageCopyLoop();
    void fileImageReadingLoop();
    void prepareFileReading();
    void prepareRTPCamera();
    void rtpStreamLoop(); // acquire from RTP network source
    void rtpConsumeFrames(); // copy into take object.
    CameraModel *Camera = NULL;
    bool fileReadingLoopRun = false;
    bool rtpConsumerRun = false;
    camControlType cameraController;
    CameraModel::camStatusEnum camStatus;

    void savingLoop(std::string, unsigned int num_avgs, unsigned int num_frames);
    std::mutex savingMutex;
    bool savingData = false;

    takeOptionsType options;

    float deltaT_micros = 100.0;
    int measuredDelta_micros_final = 0;
    int meanDeltaArrayPos = 0;
    int meanDeltaArray[meanDeltaSize] = {10}; // = {10,10,10,10,10,10,10,10,10,10};

    void markFrameForChecking(uint16_t * frame);
    bool checkFrame(uint16_t *Frame);
    void clearAllRingBuffer();

    void errorMessage(const char* message);
    void warningMessage(const char* message);
    void statusMessage(const char* message);
    void errorMessage(const string message);
    void warningMessage(const string message);
    void statusMessage(const string message);
    void statusMessage(std::ostringstream &message);

    // variables needed by the Raw Filters
    unsigned int invFactor; // inversion factor as determined by the maximum possible pixel magnitude
    bool inverted = false;
    bool pixRemap = false; // Enable Parallel Pixel Mapping (Chroma Translate filter)
    bool continuousRecording = false; // flag to enable continuous recording
    FFT_t whichFFT;
};

#endif /* TAKEOBJECT_HPP_ */
