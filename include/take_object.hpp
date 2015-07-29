/*
 * takeobject.hpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#ifndef TAKEOBJECT_HPP_
#define TAKEOBJECT_HPP_

//standard includes
#include <cstdint>
#include <cstdio>
#include <ostream>
#include <string>

//multithreading includes
#include <atomic>
#include <boost/shared_array.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
//#include <boost/atomic.hpp>

//custom includes
#include "frame_c.hpp"
#include "std_dev_filter.hpp"
#include "chroma_translate_filter.hpp"
#include "dark_subtraction_filter.hpp"
#include "mean_filter.hpp"
#include "camera_types.h"
#include "constants.h"

//These Macros set the hardware type that take_object will use to collect data
#define EDT
//#define OPALKELLY


#ifdef OPALKELLY
//OpalKelly Device Support
#include "okFrontPanelDLL.h"
#include "ok_addresses.h"

// the location of the OpalKelly bit file which configures the  FPGA
#define FPGA_CONFIG_FILE "/home/jryan/NGIS_DATA/jryan/projects/cuda_take/top4ch.bit"
#endif

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

static const bool CHECK_FOR_MISSED_FRAMES_6604A = false; // toggles the presence or absence of the "WARNING: MISSED FRAME X" line

class take_object {
#ifdef EDT
    PdvDev * pdv_p;
    unsigned int channel;
    unsigned int numbufs;
    unsigned int filter_refresh_rate;
#endif
#ifdef OPALKELLY
    okCFrontPanel* xem;
    unsigned long clock_div = CLOCK_DIV;
    unsigned long clock_delay = CLOCK_DLY;
    int blocklen;
    long framelen;
#endif

    boost::thread pdv_thread; // this thread controls the data collection
    int pdv_thread_run = 0;

	unsigned int size;
    int lastfc;

    //frame dimensions
    frame_c* curFrame;
    unsigned int dataHeight;
    unsigned int frHeight;
    unsigned int frWidth;

    //Filter-specific variables
	int std_dev_filter_N;
    dark_subtraction_filter* dsf;
    std_dev_filter* sdvf;
    int meanStartRow, meanHeight, meanStartCol, meanWidth; // dimensions used by the mean filter

    //frame saving variables
    boost::thread saving_thread; // this thread handles the frame saving, as saving frames should not cause data collection to suspend
	unsigned int save_count;
    bool do_raw_save;
	bool saveFrameAvailable;
	uint16_t * raw_save_ptr;

public:
    take_object(int channel_num = 0, int number_of_buffers = 64, int filter_refresh_rate = 10);
    virtual ~take_object();
	void start();

    camera_t cam_type;
    frame_c * frame_ring_buffer;
    unsigned long count = 0; // frame count

    //Frame filters that affect everything - at the raw data level
    void setInversion( bool, unsigned int );
    void chromaPixRemap( bool );

    //DSF mask functions
	void startCapturingDSFMask();
	void finishCapturingDSFMask();
	void loadDSFMask(std::string file_name);
    bool dsfMaskCollected;
    bool useDSF = false;

    //Std Dev Filter functions
    void setStdDev_N( int s );

    //Mean filter functions
    void updateVertRange( int, int );
    void updateHorizRange( int, int );
    void update_start_row( int );
    void update_end_row( int );
    void changeFFTtype( int );

    //Frame saving functions
    void startSavingRaws( std::string, unsigned int );
	void stopSavingRaws();
    //void panicSave( std::string );
    std::list<uint16_t *> saving_list;
	std::atomic <uint_fast32_t> save_framenum;

    //Getter functions / variablessdf
    unsigned int getDataHeight();
    unsigned int getFrameHeight();
    unsigned int getFrameWidth();
    bool std_dev_ready();
    std::vector<float> * getHistogramBins();
    int getFFTtype();

private:
	void pdv_loop();
    void savingLoop(std::string);
    //void saveFramesInBuffer();
    /* This function will save all the frames currently in the frame_ring_buffer
     * to a pre-specified raw file. For the moment, it stops the take_object loop
     * until it has finished saving. Not fully implemented. */

#ifdef OPALKELLY
    okCFrontPanel* initializeFPGA();
    void ok_init_pipe();
    void ok_read_frame(unsigned char*);
#endif

    // variables needed by the Raw Filters
    unsigned int invFactor; // inversion factor as determined by the maximum possible pixel magnitude
    bool inverted = false;
    bool chromaPix = false; // Enable Chroma Pixel Mapping (Chroma Translate filter)

    int whichFFT;
};

#endif /* TAKEOBJECT_HPP_ */
