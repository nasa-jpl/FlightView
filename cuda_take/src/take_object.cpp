#include "take_object.hpp"
#include "fft.hpp"


take_object::take_object(takeOptionsType options, int channel_num, int number_of_buffers,
                         int filter_refresh_rate, bool runStdDev)
{
    changeOptions(options);
    initialSetup(channel_num, number_of_buffers,
                 filter_refresh_rate, runStdDev);
}

take_object::take_object(int channel_num, int number_of_buffers,
                         int frf, bool runStdDev)
{
    statusMessage("Starting take_object with default options.");
    takeOptionsType options;
    options.theseAreDefault = true;
    options.xioCam = false;
    changeOptions(options);
    initialSetup(channel_num, number_of_buffers,
                 frf, runStdDev);
}

void take_object::initialSetup(int channel_num, int number_of_buffers,
                               int filter_refresh_rate, bool runStdDev)
{
    coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(coutbuf);

    closing = false;
    this->channel = channel_num;
    this->numbufs = number_of_buffers;
    this->filter_refresh_rate = filter_refresh_rate;

    frame_ring_buffer = new frame_c[CPU_FRAME_BUFFER_SIZE];

    //For the filters
    dsfMaskCollected = false;
    this->std_dev_filter_N = 400;
    this->runStdDev = runStdDev;
    whichFFT = PLANE_MEAN;

    // for the overlay, zap everything to zero:
    this->lh_start = 0;
    this->lh_start = 0;
    this->lh_end = 0;
    this->cent_start = 0;
    this->cent_end = 0;
    this->rh_start = 0;
    this->rh_end = 0;

    //For the frame saving
    this->do_raw_save = false;
    savingData = false;
    continuousRecording = false;
    save_framenum = 0;
    save_count=0;
    save_num_avgs=1;
    saving_list.clear();

    camStatus = CameraModel::camUnknown;
}

take_object::~take_object()
{
    closing = true;
    rtpConsumerRun = false;

    while(grabbing)
    {
        // wait here for last frame to complete
        usleep(1000);
    }
    if(pdv_thread_run != 0) {
        pdv_thread_run = 0;

        int dummy;
        if(pdv_p)
        {
            pdv_wait_last_image(pdv_p,&dummy); //Collect the last frame to avoid core dump
            pdv_close(pdv_p);
        }
        if(Camera)
        {
            LOG << "Deleting camera.";
            usleep(100000);
            delete Camera;
            usleep(100000);
            LOG << "Done deleting camera.";
        }

#ifdef VERBOSE
        printf("about to delete filters!\n");
#endif

        delete dsf;
        delete sdvf;
    }

    delete[] frame_ring_buffer;

#ifdef RESET_GPUS
    printf("reseting GPUs!\n");
    int count;
    cudaGetDeviceCount(&count);
    for(int i = 0; i < count; i++) {
        printf("resetting GPU#%i",i);
        cudaSetDevice(i);
        cudaDeviceReset(); //Dump all the bad stuff from each of our GPUs.
    }
#endif
}

//public functions
void take_object::changeOptions(takeOptionsType optionsIn)
{
    this->options = optionsIn;

    if(optionsIn.xioDirSet)
    {
        if(optionsIn.xioDirectory == NULL)
        {
            errorMessage("Cannot have set directory that is null.");
            abort();
        } else {
             safeStringSet(options.xioDirectory, optionsIn.xioDirectory);
            }
   } else {
        // TODO: Wait safely for a directory, and do not try reading yet.
        if(optionsIn.xioCam) {
            statusMessage("xio directory not set. Use interface to specify directory.");
        }
    }

    if(!options.theseAreDefault) {
        statusMessage(std::string("Accepted startup options. Target FPS: ") + std::to_string(options.targetFPS));
        if(options.xioDirSet && options.xioCam)
        {
            statusMessage(std::string("XIO directory: ") + *options.xioDirectory);
        }

        if(options.rtpCam)
        {
            statusMessage("RTP Camera enabled.");
            statusMessage(std::string("RTP Height: ") + std::to_string(options.rtpHeight));
            statusMessage(std::string("RTP Width:  ") + std::to_string(options.rtpWidth));
        } else {
            statusMessage("RTP Camera disabled.");
        }
        if(options.rtpNextGen) {
            statusMessage("RTP Camera is NextGen model");
        }
        if((!options.rtpCam) && (!options.xioCam)) {
            statusMessage("CameraLink enabled.");
        }
    }

    // Recalculate the frame-to-frame delay:
    deltaT_micros = 1000000.0 / options.targetFPS;
}

void take_object::shmSetup()
{
    statusMessage("Preparing shared memory segment for images.");

    if(shmValid) {
        // unexpected to be valid already
        warningMessage("SHM already valid");
        return;
    }

    size_t shmLen = sizeof(struct shmSharedDataStruct);
    shmFd = shm_open("/liveview_image", O_RDWR | O_CREAT ,S_IRUSR | S_IWUSR);

    if(shmFd == -1) {
        errorMessage("Could not open shared memory segment.");
        shmValid = false;
        return;
    } else {
        statusMessage("Created shared memory segment /liveview_image");
    }

    char trunmessage[128];
    if(ftruncate(shmFd, shmLen) == -1) {
        sprintf(trunmessage, "Could not truncate shared memory segment to %zu bytes.", shmLen);
        errorMessage(trunmessage);
        shmValid = false;
        goto cleanup;
    } else {
        sprintf(trunmessage, "Truncated shared memory segment to %zu bytes.", shmLen);
        statusMessage(trunmessage);
    }

    shm = (shmSharedDataStruct*)mmap (0, shmLen, PROT_WRITE, MAP_SHARED, shmFd, 0);
    if( (shm == NULL) || (shm==MAP_FAILED) ) {
        errorMessage("Could not map memory shared memory segment to a local variable.");
        shmValid = false;
        goto cleanup;
    }

    shm->statusByte = SHM_STATUS_INITALIZING;
    shm->recordingDataToFile = false;
    shm->fps = 0.0;
    shm->counter = 0;
    shm->writingFrameNum = 0;
    shm->bufferSizeFrames = shmFrameBufferSize;
    shm->frameHeight = this->frHeight;
    shm->frameWidth = this->frWidth;
    shm->takingDark = false;

    for(int i=0; i < shmFilenameBufferSize; i++) {
        shm->lastFilename[i] = '\0';
    }

    for(int i=0; i < shmFrameBufferSize; i++) {
        shm->frameTime[i] = 0;
    }

    for(int f=0; f < shmFrameBufferSize; f++) {
        for(int p=0; p < shmHeight*shmWidth; p++) {
            shm->frameBuffer[f][p] = 0;
        }
    }
    shm->statusByte = SHM_STATUS_WAITING;
    shmValid = true;
    goto cleanup;

    cleanup:
    if( (shmFd != -1) && (shmFd != 0) ) {
        close(shmFd);
        return;
    }
}

void take_object::start()
{
    pdv_thread_run = 1;

    std::cout << "This version of cuda_take was compiled on " << __DATE__ << " at " << __TIME__ << " using gcc " << __GNUC__ << std::endl;
    std::cout << "The compilation was perfromed by " << UNAME << " @ " << HOST << std::endl;

    pthread_setname_np(pthread_self(), "TAKE");

    this->pdv_p = NULL;

    if(options.xioCam)
    {
        if(!options.heightWidthSet)
        {
            options.xioHeight = 481;
            options.xioWidth = 640;
            warningMessage("Warning: XIO Height and Width not specified. Assuming 640x481 geometry.");
        }
        frWidth = options.xioWidth;
        frHeight = options.xioHeight;
        dataHeight = options.xioHeight;
        size = frWidth * frHeight * sizeof(uint16_t);
        statusMessage("start() running with with XIO camera settings.");
    } else if (options.rtpCam)
    {
        if(!options.heightWidthSet)
        {
            options.rtpHeight = 481;
            options.rtpWidth= 640;
            warningMessage("Warning: RTP Height and Width not specified. Assuming 640x481 geometry.");
        }
        frWidth = options.rtpWidth;
        frHeight = options.rtpHeight;
        dataHeight = options.rtpHeight;
        size = frWidth * frHeight * sizeof(uint16_t);
        statusMessage("start() running with with RTP camera settings.");
        std::cout << "Height: " << frHeight << ", width: " << frWidth << std::endl;
        if(options.rtpAddress != NULL)
        {
            std::cout << "rtpAddress: " << options.rtpAddress << std::endl;
        }
        if(options.rtpInterface != NULL)
        {
            std::cout << "rtpInterface: " << options.rtpInterface << std::endl;
        }
    } else {
        this->pdv_p = pdv_open_channel(EDT_INTERFACE,0,this->channel);
        if(pdv_p == NULL) {
            std::cerr << "Could not open device channel. Is one connected?" << std::endl;
            return;
        }
        size = pdv_get_dmasize(pdv_p); // this size is only used to determine the camera type
        // actual grabbing of the dimensions
        frWidth = pdv_get_width(pdv_p);
        dataHeight = pdv_get_height(pdv_p);
        frHeight = dataHeight;
    }

    switch(size) {
    case 481*640*sizeof(uint16_t): cam_type = CL_6604A; break;
    case 285*640*sizeof(uint16_t): cam_type = CL_6604A; break;
    case 480*640*sizeof(uint16_t): cam_type = CL_6604B; pixRemap = true; break;
    default: cam_type = CL_6604B; pixRemap = true; break;
    }
	setup_filter(cam_type);
    setup_filter(frHeight, frWidth);
	if(pixRemap) {
		std::cout << "2s compliment filter ENABLED" << std::endl;
	} else {
		std::cout << "2s compliment filter DISABLED" << std::endl;
	}


    //frHeight = cam_type == CL_6604A ? dataHeight - 1 : dataHeight;
    frHeight = dataHeight;

#ifdef VERBOSE
    std::cout << "Camera Type: " << cam_type << ". Frame Width: " << frWidth << \
                 " Data Height: " << dataHeight << " Frame Height: " << frHeight << std::endl;
    std::cout << "About to start threads..." << std::endl;
#endif

    // Initialize the filters
    dsf = new dark_subtraction_filter(frWidth,frHeight);
    sdvf = new std_dev_filter(frWidth,frHeight);

    // Initial dimensions for calculating the mean that can be updated later
    meanStartRow = 0;
    meanStartCol = 0;
    meanHeight = frHeight;
    meanWidth = frWidth;

#ifdef USE_SHM
    // Get the shared memory segment for images ready:
    if(options.useSHM) {
        shmSetup();
    }
#else
    statusMessage("Not initializing shared memory segment.");
#endif


    numbufs = 16;
    int rtnval = 0;
    if(options.xioCam)
    {
        cam_thread_start_complete = false;
        statusMessage("Creating an XIO camera take_object.");
        prepareFileReading(); // make a camera
        statusMessage("Creating an XIO camera thread inside take_object.");
        cam_thread = boost::thread(&take_object::fileImageCopyLoop, this);
        cam_thread_handler = cam_thread.native_handle();
        pthread_setname_np(cam_thread_handler, "XIOCAM");
        statusMessage("Created thread.");
        while(!cam_thread_start_complete)
            usleep(100);
        // The idea is to hold off on doing anything else until some setup is finished.

        statusMessage("Creating XIO File reading thread reading_thread.");
        reading_thread = boost::thread(&take_object::fileImageReadingLoop, this);
        reading_thread_handler = reading_thread.native_handle();
        pthread_setname_np(reading_thread_handler, "READING");
        statusMessage("Done creating XIO File reading thread reading_thread.");

        char threadinfo[16];
        statusMessage("Thread Information: ");
        std::ostringstream info;

        pthread_getname_np(cam_thread_handler, threadinfo, 16);
        info << "Cam thread name: " << threadinfo;
        statusMessage(info);

        info.str("");

        pthread_getname_np(reading_thread_handler, threadinfo, 16);
        info << "Reading thread name: " << string(threadinfo);
        statusMessage(info);

        info.str("");

        pthread_getname_np(pthread_self(), threadinfo, 16);
        info << "Self thread name: " << string(threadinfo);
        statusMessage(info);


    } else if (options.rtpNextGen) {
        statusMessage("Starting RTP NextGen camera in take object.");
        cam_thread_start_complete = false;
        statusMessage("Preparing RTP NextGen camera");
        std::cout.rdbuf(coutbuf); // restore cout
        prepareRTPNGCamera();
        std::cout.rdbuf(coutbuf);

        statusMessage("Creating boost thread for RTP NextGen camera streamLoop()");
        rtpAcquireThread = boost::thread(&take_object::rtpNGStreamLoop, this);
        rtpAcquireThreadHandler = rtpAcquireThread.native_handle();
        pthread_setname_np(rtpAcquireThreadHandler, "RTPNG Stream");
        statusMessage("Created RTP NextGen streamLoop() thread.");

        // At this point, the RTP camera is initialized and now it is running.
        // Data is being acquired if the stream source is emitting data,
        // and data is being copied into the guarenteed frame buffer of the RTPCamera.

        // These functions get the data into the rest of take object:
        rtpConsumerRun = true;
        statusMessage("Creating RTP NextGen consumer thread to copy data into take_object");
        rtpCopyThread = boost::thread(&take_object::rtpConsumeFrames, this);
        rtpCopyThreadHandler = rtpCopyThread.native_handle();
        pthread_setname_np(rtpCopyThreadHandler, "RTPNG Consume");
        statusMessage("Created RTP NextGen consumer thread.");

    } else if (options.rtpCam) {
        statusMessage("Starting RTP camera in take object.");
        cam_thread_start_complete = false;
        statusMessage("Preparing RTP camera");
        prepareRTPCamera();

        statusMessage("Creating boost thread for camera streamLoop()");
        rtpAcquireThread = boost::thread(&take_object::rtpStreamLoop, this);
        rtpAcquireThreadHandler = rtpAcquireThread.native_handle();
        pthread_setname_np(rtpAcquireThreadHandler, "RTP Stream");
        statusMessage("Created RTP streamLoop() thread.");
        // At this point, the RTP camera is initialized and now it is running.
        // Data is being acquired if the stream source is emitting data,
        // and data is being copied into the guarenteed frame buffer of the RTPCamera.

        rtpConsumerRun = true;
        statusMessage("Creating RTP consumer thread to copy data into take_object");
        rtpCopyThread = boost::thread(&take_object::rtpConsumeFrames, this);
        rtpCopyThreadHandler = rtpCopyThread.native_handle();
        pthread_setname_np(rtpCopyThreadHandler, "RTP Consume");
        statusMessage("Created RTP consumer thread.");

    } else {
        statusMessage("Creating CameraLink multibuf.");
        if(pdv_p != NULL)
            rtnval = pdv_multibuf(pdv_p,this->numbufs);
        if(rtnval != 0)
        {
            std::cerr << "Error, could not initialize camera link multibuffer." << std::endl;
            std::cerr << "Make sure the camera link driver is loaded and that the camera link port has been initialized using initcam." << std::endl;
            system("xmessage \"Error, please initialize the camera link frame grabber first.\"");
            abort();
        }


        pdv_start_images(pdv_p,numbufs); //Before looping, emit requests to fill the pdv ring buffer
        cam_thread = boost::thread(&take_object::pdv_loop, this);
        cam_thread_handler = cam_thread.native_handle();
        pthread_setname_np(cam_thread_handler, "PDVCAM");
        //usleep(350000);
        while(!cam_thread_start_complete) usleep(1); // Added by Michael Bernas 2016. Used to prevent thread error when starting without a camera
    }
    statusMessage("Finished creating threads.");
}
void take_object::setInversion(bool checked, unsigned int factor)
{
    inverted = checked;
    invFactor = factor;
}
void take_object::paraPixRemap(bool checked )
{
    pixRemap = checked;
    std::cout << "2s Compliment Filter ";
    if(pixRemap) {
        std::cout << "ENABLED" << std::endl;
    } else {
        std::cout << "DISABLED" << std::endl;
    }
}

void take_object::enableDarkStatusPixelWrite(bool writeValues) {
    setDarkStatusInFrame = writeValues;
}

void take_object::startCapturingDSFMask()
{
    dsfMaskCollected = false;

    dsf->start_mask_collection();
    if(shmValid) {
        shm->takingDark = true;
    }

    darkStatusPixelVal = obcStatusDark1;
}
void take_object::finishCapturingDSFMask()
{
    dsf->mask_mutex.lock();
    dsf->finish_mask_collection();
    dsf->mask_mutex.unlock();
    dsfMaskCollected = true;
    if(shmValid) {
        shm->takingDark = false;
    }
    darkStatusPixelVal = obcStatusScience;
}
void take_object::loadDSFMaskFromFramesU16(std::string file_name, fileFormat_t format)
{
    // Creates a mask from a file containing multiple frames
    // The frames are expected to be the same geometry as the
    // frame source, and the pixels are expected to be 16-bit unsigned int.

    // This function was largly copied from the main.cpp file of
    // the included "statscli" program found under "utils".

    std::ostringstream message;

    float * mean_frame = NULL;
    uint16_t * frames = NULL;
    unsigned int * input_array = NULL;

    unsigned int frame_size_numel = frHeight*frWidth;
    unsigned int nframes = 0;
    unsigned int pixel_size = sizeof(uint16_t);
    size_t items_read = 0;
    (void)format;

    FILE * file = fopen(file_name.c_str(), "r");
    if(file == NULL)
    {
        message << "Error, could not load DSF file " << file_name;
        statusMessage(message);
        return;
    }

    fseek(file, 0, SEEK_END);
    long int filesize = ftell(file);
    nframes = filesize / pixel_size / (frHeight * frWidth);
    fseek(file, 0, SEEK_SET);
    frames = (uint16_t *) malloc(filesize * pixel_size);
    if(frames == NULL)
    {
        errorMessage("Did not successfully allocate frames for dark subtraction file");
        abort();
    }

    items_read = fread(frames, sizeof(uint16_t), filesize/pixel_size, file);

    message << "DSF Load: Read      " << items_read << " pixels from " << file_name;
    statusMessage(message); message.str("");

    fclose(file);

    mean_frame = (float *) malloc(sizeof(float) * frame_size_numel);

    // The input_array is where the data are initially loaded.
    // These data can be type-converted after loading.
    input_array = (unsigned int *) malloc(sizeof(unsigned int) * nframes * frame_size_numel); // native size
    if(input_array == NULL)
    {
        errorMessage("Did not successfully allocate input_array for dark subtraction file");
        abort();
    }

    if(format == fmt_uint16_2s)
    {
        // Convert the data first:
        for(unsigned int nth_element = 0; nth_element < frame_size_numel * nframes; nth_element++)
        {
            input_array[nth_element] = (unsigned int)( frames[nth_element] ^ (1<<15) );
        }
        // Process:
        #pragma omp parallel for
        for(unsigned int nth_frame_el = 0; nth_frame_el < frame_size_numel; nth_frame_el++)
        {
            // iterate over each pixel in a frame
            mean_frame[nth_frame_el] = (float)gsl_stats_uint_mean(input_array+nth_frame_el, frame_size_numel, nframes);
        }
    } else {
        // Convert uint16_t to unsigned int for GSL:
        // TODO: consider loading it in this way
        for(unsigned int nth_element = 0; nth_element < frame_size_numel * nframes; nth_element++)
        {
            input_array[nth_element] = (unsigned int)frames[nth_element];
        }
        // Process:
        #pragma omp parallel for
        for(unsigned int nth_frame_el = 0; nth_frame_el < frame_size_numel; nth_frame_el++)
        {
            // iterate over each pixel in a frame
            mean_frame[nth_frame_el] = (float)gsl_stats_uint_mean(input_array+nth_frame_el, frame_size_numel, nframes);
        }
    }

    dsf->load_mask(mean_frame); // memcopy to stack variable
    dsfMaskCollected = true;

    if(frames)
        free(frames);
    if(mean_frame)
        free(mean_frame);
}

void take_object::loadDSFMask(std::string file_name)
{
    // Loads a file containing a single 32-bit float frame.
    float *mask_in = new float[frWidth*frHeight];
    FILE *pFile;
    unsigned long size = 0;
    pFile  = fopen(file_name.c_str(), "rb");
    if(pFile == NULL) std::cerr << "error opening raw file" << std::endl;
    else
    {
        fseek (pFile, 0, SEEK_END); // non-portable
        size = ftell(pFile);
        if(size != (frWidth*frHeight*sizeof(float)))
        {
            std::cerr << "Error: mask file does not match image size" << std::endl;
            fclose (pFile);
            return;
        }
        rewind(pFile);   // go back to beginning
        fread(mask_in,sizeof(float),frWidth * frHeight,pFile);
        fclose (pFile);
#ifdef VERBOSE
        std::cout << file_name << " read in "<< size << " bytes successfully " <<  std::endl;
#endif
    }
    dsf->load_mask(mask_in); // memcopy to stack variable
    delete mask_in;
}
void take_object::setStdDev_N(int s)
{
    this->std_dev_filter_N = s;
}

void take_object::toggleStdDevCalculation(bool enabled)
{
    this->runStdDev = enabled;
}

void take_object::updateVertOverlayParams(int lh_start_in, int lh_end_in,
                                          int cent_start_in, int cent_end_in,
                                          int rh_start_in, int rh_end_in)
{
    this->lh_start = lh_start_in;
    this->lh_start = lh_start_in;
    this->lh_end = lh_end_in;
    this->cent_start = cent_start_in;
    this->cent_end = cent_end_in;
    this->rh_start = rh_start_in;
    this->rh_end = rh_end_in;

    /*
    // Debug, remove later:
    std::cout << "----- In take_object::updateVertOverlayParams\n";
    std::cout << "->lh_start:   " << lh_start <<   ", lh_end:   " << lh_end << std::endl;
    std::cout << "->rh_start:   " << rh_start <<   ", rh_end:   " << rh_end << std::endl;
    std::cout << "->cent_start: " << cent_start << ", cent_end: " << cent_end << std::endl;
    std::cout << "----- end take_object::updateVertOverlayParams -----\n";
    */
}

void take_object::updateVertRange(int br, int er)
{
    meanStartRow = br;
    meanHeight = er;
#ifdef VERBOSE
    std::cout << "meanStartRow: " << meanStartRow << " meanHeight: " << meanHeight << std::endl;
#endif
}
void take_object::updateHorizRange(int bc, int ec)
{
    meanStartCol = bc;
    meanWidth = ec;
#ifdef VERBOSE
    std::cout << "meanStartCol: " << meanStartCol << " meanWidth: " << meanWidth << std::endl;
#endif
}
void take_object::changeFFTtype(FFT_t t)
{
    whichFFT = t;
}
void take_object::startSavingRaws(std::string raw_file_name, unsigned int frames_to_save, unsigned int num_avgs_save)
{
    if(frames_to_save==0)
    {
        continuousRecording = true;
    } else {
        continuousRecording = false;
    }
    
    save_framenum.store(0, std::memory_order_seq_cst);
    save_count.store(0, std::memory_order_seq_cst);
#ifdef VERBOSE
    printf("ssr called\n");
#endif
    while(!saving_list.empty())
    {
#ifdef VERBOSE
        printf("Waiting for empty saving list...\n");
#endif
    }
    save_framenum.store(frames_to_save,std::memory_order_seq_cst);
    save_count.store(0, std::memory_order_seq_cst);
    save_num_avgs=num_avgs_save;
#ifdef VERBOSE
    printf("Begin frame save! @ %s\n", raw_file_name.c_str());
#endif
    if(shmValid) {
        strncpy(shm->lastFilename, raw_file_name.c_str(), shmFilenameBufferSize-1);
        shm->recordingDataToFile = true;
    }
    saving_thread = boost::thread(&take_object::savingLoop,this,raw_file_name,num_avgs_save,frames_to_save);
}
void take_object::stopSavingRaws()
{
    continuousRecording = false;
    save_framenum.store(0,std::memory_order_relaxed);
    save_count.store(0,std::memory_order_relaxed);
    save_num_avgs=1;
    if(shmValid) {
        shm->recordingDataToFile = false;
    }

#ifdef VERBOSE
    printf("Stop Saving Raws!");
#endif
}
unsigned int take_object::getDataHeight()
{
    return dataHeight;
}
unsigned int take_object::getFrameHeight()
{
    return frHeight;
}
unsigned int take_object::getFrameWidth()
{
    return frWidth;
}
bool take_object::std_dev_ready()
{
    return sdvf->outputReady();
}
std::vector<float> * take_object::getHistogramBins()
{
    return sdvf->getHistogramBins();
}
FFT_t take_object::getFFTtype()
{
    return whichFFT;
}

// private functions

void take_object::prepareFileReading()
{
    // Makes an XIO file reading camera

    if(Camera == NULL)
    {
        Camera = new XIOCamera(frWidth,
                               frHeight,
                               frHeight);
        this->Camera->setCamControlPtr(&this->cameraController);
        if(Camera == NULL)
        {
            errorMessage("XIO Camera could not be created, was NULL.");
        } else {
            statusMessage(string("XIO Camera was made"));
        }
    } else {
        errorMessage("XIO Camera should be NULL at start but isn't");
    }

    bool cam_started = Camera->start();
    if(cam_started)
    {
        statusMessage("XIO Camera started.");
    } else {
        errorMessage("XIO Camera not started");
    }
}

void take_object::prepareRTPCamera()
{
    // Makes an RTP gstreamer pipeline and related objects

    if(Camera == NULL)
    {
        // TODO: add parameters to startup options
//        Camera = new RTPCamera(frWidth,
//                               frHeight,
//                               5004, "lo");
        Camera = new RTPCamera(options);
        this->Camera->setCamControlPtr(&this->cameraController);
        if(Camera == NULL)
        {
            errorMessage("RTP Camera could not be created, was NULL.");
        } else {
            statusMessage("RTP Camera was made");
        }
    } else {
        errorMessage("RTP Camera should be NULL at start but isn't");
    }
}

void take_object::prepareRTPNGCamera() {
    if(Camera == NULL) {
        Camera = new rtpnextgen(options);
        if(Camera == NULL) {
            errorMessage("RTP NextGen camera was NULL");
        } else {
            statusMessage("RTP NextGen camera created.");
        }
        this->Camera->setCamControlPtr(&this->cameraController);
    } else {
        // re-create camera?
        errorMessage("RTP NextGen camera was expected to be NULL but was not!");
    }
}

void take_object::fileImageReadingLoop()
{
    // This thread makes the camera keep reading files
    // readLoop() runs readFile() inside.

    if(Camera)
    {
        statusMessage(std::string("Starting XIO Camera readLoop() function. Initial closing value: ") + std::string(closing?"true":"false"));
        // TODO: Come up with a switchable condition here
        // One-shot mode:
        while(!closing)
        {
            Camera->readLoop();
            //statusMessage("Completed readLoop(), pausing and then running again.");
            usleep(100000);
        }
        statusMessage("completed XIO Camera readLoop() while function. No more files can be read once completed. ");
    } else {
        errorMessage("XIO Camera is NULL, cannot readLoop().");
    }
}

void take_object::markFrameForChecking(uint16_t *frame)
{
    // This function overrides some data in the top three rows of the frame.
    // This is only to be used for debugging.

    // Pattern:
    // X 0 X 0 X X 0 X 0 X
    // X 0 X 0 X X 0 X 0 X
    // X 0 X 0 X X 0 X 0 X

    frame[0] = (uint16_t)0xffff;
    frame[1] = (uint16_t)0x0000;
    frame[2] = (uint16_t)0xffff;
    frame[3] = (uint16_t)0x0000;
    frame[4] = (uint16_t)0xffff;
    frame[5] = (uint16_t)0xffff;
    frame[6] = (uint16_t)0x0000;
    frame[7] = (uint16_t)0xffff;
    frame[8] = (uint16_t)0x0000;
    frame[9] = (uint16_t)0xffff;

    frame[0+640] = (uint16_t)0xffff;
    frame[1+640] = (uint16_t)0x0000;
    frame[2+640] = (uint16_t)0xffff;
    frame[3+640] = (uint16_t)0x0000;
    frame[4+640] = (uint16_t)0xffff;
    frame[5+640] = (uint16_t)0xffff;
    frame[6+640] = (uint16_t)0x0000;
    frame[7+640] = (uint16_t)0xffff;
    frame[8+640] = (uint16_t)0x0000;
    frame[9+640] = (uint16_t)0xffff;

    frame[0+640+640] = (uint16_t)0xffff;
    frame[1+640+640] = (uint16_t)0x0000;
    frame[2+640+640] = (uint16_t)0xffff;
    frame[3+640+640] = (uint16_t)0x0000;
    frame[4+640+640] = (uint16_t)0xffff;
    frame[5+640+640] = (uint16_t)0xffff;
    frame[6+640+640] = (uint16_t)0x0000;
    frame[7+640+640] = (uint16_t)0xffff;
    frame[8+640+640] = (uint16_t)0x0000;
    frame[9+640+640] = (uint16_t)0xffff;
}

bool take_object::checkFrame(uint16_t* Frame)
{
    bool ok = true;
    ok &= Frame[1] == (uint16_t)0x0000;
    ok &= Frame[2] == (uint16_t)0xffff;
    ok &= Frame[3] == (uint16_t)0x0000;
    ok &= Frame[4] == (uint16_t)0xffff;
    ok &= Frame[5] == (uint16_t)0xffff;
    ok &= Frame[6] == (uint16_t)0x0000;
    ok &= Frame[7] == (uint16_t)0xffff;
    ok &= Frame[8] == (uint16_t)0x0000;
    ok &= Frame[9] == (uint16_t)0xffff;
    statusMessage(std::string("Frame check result (1 of 3): ") + std::string(ok?"GOOD":"BAD"));

    // Test for bad data:
    // Frame[4+640] = (uint16_t)0xABCD; // intentional

    ok &= Frame[0+640] == (uint16_t)0xffff;
    ok &= Frame[1+640] == (uint16_t)0x0000;
    ok &= Frame[2+640] == (uint16_t)0xffff;
    ok &= Frame[3+640] == (uint16_t)0x0000;
    ok &= Frame[4+640] == (uint16_t)0xffff;
    ok &= Frame[5+640] == (uint16_t)0xffff;
    ok &= Frame[6+640] == (uint16_t)0x0000;
    ok &= Frame[7+640] == (uint16_t)0xffff;
    ok &= Frame[8+640] == (uint16_t)0x0000;
    ok &= Frame[9+640] == (uint16_t)0xffff;
    statusMessage(std::string("Frame check result: (2 of 3): ") + std::string(ok?"GOOD":"BAD"));

    ok &= Frame[0+640+640] == (uint16_t)0xffff;
    ok &= Frame[1+640+640] == (uint16_t)0x0000;
    ok &= Frame[2+640+640] == (uint16_t)0xffff;
    ok &= Frame[3+640+640] == (uint16_t)0x0000;
    ok &= Frame[4+640+640] == (uint16_t)0xffff;
    ok &= Frame[5+640+640] == (uint16_t)0xffff;
    ok &= Frame[6+640+640] == (uint16_t)0x0000;
    ok &= Frame[7+640+640] == (uint16_t)0xffff;
    ok &= Frame[8+640+640] == (uint16_t)0x0000;
    ok &= Frame[9+640+640] == (uint16_t)0xffff;

    statusMessage(std::string("Frame check result: (3 of 3): ") + std::string(ok?"GOOD":"BAD"));

    return ok;
}

void take_object::clearAllRingBuffer()
{
    frame_c *curFrame = NULL;
    uint16_t *zeroFrame = NULL;
    zeroFrame = (uint16_t*)calloc(frWidth*dataHeight , sizeof(uint16_t));
    if(zeroFrame == NULL)
    {
        errorMessage("Zero-frame could not be established.");
        abort();
    }

    for(size_t f=0; f < CPU_FRAME_BUFFER_SIZE; f++)
    {
        curFrame = &frame_ring_buffer[f];
        curFrame->reset();
        memcpy(curFrame->raw_data_ptr,zeroFrame,frWidth*dataHeight);
    }
    statusMessage("Done zero-setting memory in frame_ring_buffer");
}

void take_object::fileImageCopyLoop()
{
    // This thread copies data from the XIO Camera's buffer
    // and into curFrane of take_object. It is the "consumer"
    // thread in a way.

    bool good = false;
    uint16_t *zeroFrame = NULL;
    zeroFrame = (uint16_t*)calloc(frWidth*dataHeight , sizeof(uint16_t));
    if(zeroFrame == NULL)
    {
        errorMessage("Zero-frame could not be established. You may be out of memory.");
        abort();
    }

    // Verify our frame data stability:
    markFrameForChecking(zeroFrame); // adds special data to the frame which can be checked for later.

    bool goodResult = checkFrame(zeroFrame);
    if(goodResult == false)
    {
        errorMessage("ERROR, BAD data detected");
        abort();
    } else {
        statusMessage("Initial data check passed.");
    }
    // End verification.

    volatile bool hasBeenNull = false;
    (void)hasBeenNull;
    if(Camera)
    {
        count = 0;
        uint16_t framecount = 1;
        uint16_t last_framecount = 0;
        (void)last_framecount; // use count

        mean_filter * mf = new mean_filter(curFrame,count,meanStartCol,meanWidth,\
                                           meanStartRow,meanHeight,frWidth,useDSF,\
                                           whichFFT, lh_start, lh_end,\
                                           cent_start, cent_end,\
                                           rh_start, rh_end);
        setup_filter(frHeight, frWidth);

        if(options.targetFPS == 0.0)
            options.targetFPS = 100.0;

        deltaT_micros = 1000000.0 / options.targetFPS;
        int measuredDelta_micros = 0;
        fileReadingLoopRun = true;

        std::chrono::steady_clock::time_point begintp;
        std::chrono::steady_clock::time_point endtp;
        std::chrono::steady_clock::time_point finaltp;

        xioCount = 0;
        int ngFrameCount = 0;
        bool wasPaused = false;
        bool wasTestPattern = false;
        bool wasDone = false;

        while(fileReadingLoopRun && (!closing))
        {
            begintp = std::chrono::steady_clock::now();

            grabbing = true;
            curFrame = &frame_ring_buffer[count % CPU_FRAME_BUFFER_SIZE];
            curFrame->reset();

            if(closing)
            {
                fileReadingLoopRun = false;
                break;
            } else {
                // start image collection on the camera

            }
            cam_thread_start_complete=true;

            uint16_t* temp_frame = Camera->getFrame(&this->camStatus);

            if(camStatus==CameraModel::camPlaying)
            {
                xioCount++;
                if(wasPaused)
                {
                    ngFrameCount = 0;
                    wasPaused = false;
                }
                if(wasDone)
                {
                    wasDone = false;
                    xioCount = 0;
                    ngFrameCount = 0;
                }
                if(wasTestPattern)
                {
                    wasTestPattern = false;
                    xioCount = 0;
                    ngFrameCount = 0;
                }
            } else if (camStatus==CameraModel::camPaused) {
                // Generally this happens when we are out of frames to read.
                wasPaused = true;
                ngFrameCount++;
            } else if (camStatus==CameraModel::camDone) {
                wasDone = true;
                ngFrameCount++;
            } else if (camStatus==CameraModel::camTestPattern)
            {
                ngFrameCount++;
                wasTestPattern = true;
            }



            prior_temp_frame = temp_frame; // store the old address for comparison

            if(temp_frame)
            {
                memcpy(curFrame->raw_data_ptr,temp_frame,frWidth*dataHeight*2);
            } else {
                hasBeenNull = true;
                errorMessage("Frame was NULL!");
                memcpy(curFrame->raw_data_ptr,zeroFrame,frWidth*dataHeight*2);
            }

            // From here on out, the code should be
            // very similar to the EDT frame grabber code.


            if(pixRemap)
            {
                apply_chroma_translate_filter(curFrame->raw_data_ptr);
                curFrame->image_data_ptr = curFrame->raw_data_ptr;
            }

//            if(cam_type == CL_6604A)
//                curFrame->image_data_ptr = curFrame->raw_data_ptr + frWidth;
//            else
            curFrame->image_data_ptr = curFrame->raw_data_ptr;
            if(inverted)
            { // record the data from high to low. Store the pixel buffer in INVERTED order from the camera link
                for(uint i = 0; i < frHeight*frWidth; i++ )
                    curFrame->image_data_ptr[i] = invFactor - curFrame->image_data_ptr[i];
            }


            // Calculating the filters for this frame
            if(runStdDev)
            {
                sdvf->update_GPU_buffer(curFrame,std_dev_filter_N);
            }
            dsf->update(curFrame->raw_data_ptr,curFrame->dark_subtracted_data);
            mf->update(curFrame,count,meanStartCol,meanWidth,\
                       meanStartRow,meanHeight,frWidth,useDSF,\
                       whichFFT, lh_start, lh_end,\
                                               cent_start, cent_end,\
                                               rh_start, rh_end);

            mf->start_mean();

            if((save_framenum > 0) || continuousRecording)
            {
                uint16_t * raw_copy = new uint16_t[frWidth*dataHeight];
                memcpy(raw_copy,curFrame->raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
                saving_list.push_front(raw_copy);
                save_framenum--;
            }

            framecount = *(curFrame->raw_data_ptr + 160); // The framecount is stored 160 bytes offset from the beginning of the data
            /*
            if(CHECK_FOR_MISSED_FRAMES_6604A && cam_type == CL_6604A)
            {
                if( (framecount - 1 != last_framecount) && (last_framecount != UINT16_MAX) )
                {
                    std::cerr << "WARNING: MISSED FRAME " << framecount << std::endl;
                }
            }
            */

            last_framecount = framecount;
            count++;
            grabbing = false;
            if(closing)
            {
                fileReadingLoopRun = false;
                break;
            }


            // Forced FPS
            endtp = std::chrono::steady_clock::now();
            measuredDelta_micros = std::chrono::duration_cast<std::chrono::microseconds>(endtp-begintp).count();
            if(measuredDelta_micros < deltaT_micros)
            {
                // wait
                //statusMessage(std::string("Waiting additional ") + std::to_string(deltaT_micros - measuredDelta_micros) + std::string(" microseconds."));
                usleep(deltaT_micros - measuredDelta_micros);
            } else {
                //warningMessage("Cannot guarentee requested frame rate. Frame rate is too fast or computation is too slow.");
                //warningMessage(std::string("Requested deltaT: ") + std::to_string(deltaT_micros) + std::string(", measured delta microseconds: ") + std::to_string(measuredDelta_micros));
            }
            finaltp = std::chrono::steady_clock::now();
            measuredDelta_micros_final = std::chrono::duration_cast<std::chrono::microseconds>(finaltp-begintp).count();
            meanDeltaArray[(++meanDeltaArrayPos)%meanDeltaSize] = measuredDelta_micros_final;
        }
        statusMessage("Done providing frames");
    } else {
        errorMessage("Camera was NULL!");
        abort();
    }
    if(zeroFrame != NULL)
        free(zeroFrame);
}

int take_object::getMicroSecondsPerFrame()
{
    int max = (meanDeltaArrayPos < meanDeltaSize)?meanDeltaArrayPos:meanDeltaSize;
    // If max is 0, we have not actually taken a reading yet.
    if(max == 0)
        return 0;

    int sum = 0;
    for(int i=0; i < max; i++)
    {
        sum += meanDeltaArray[i];
    }
    return sum / max;
}

void take_object::setReadDirectory(const char *directory)
{
    if(directory == NULL)
    {
        errorMessage("directory is empty string or NULL, cannot set directory.");
        return;
    }

    if(Camera == NULL)
    {
        errorMessage("Camera is NULL! Cannot set directory (yet).");
        return;
    }

    if(sizeof(directory) != 0)
    {
        statusMessage(string("Setting directory to: ") + directory);
        Camera->setDir(directory);
    } else {
        errorMessage("Cannot set directory to zero-length string.");
    }
}

camControlType* take_object::getCamControl()
{
    return &cameraController;
}

void take_object::rtpStreamLoop()
{
    LOG << "Entering streamLoop";
    Camera->streamLoop();
}

void take_object::rtpNGStreamLoop() {
    LOG << "Entering streamLoop";
    Camera->streamLoop();
}

void take_object::rtpConsumeFrames()
{
    // This thread copies frames from the RTP Stream Loop
    // guarenteed buffer into the take object.
    // The frames are copied using Camera->getFrameWait
    // which waits for new frames.

    // Initializers just in case:
    save_framenum = 0;
    continuousRecording = false;

    mean_filter * mf = new mean_filter(curFrame,count,meanStartCol,meanWidth,\
                                       meanStartRow,meanHeight,frWidth,useDSF,\
                                       whichFFT, lh_start, lh_end,\
                                       cent_start, cent_end,\
                                       rh_start, rh_end);

    std::chrono::steady_clock::time_point begintp;
    std::chrono::steady_clock::time_point finaltp;

    int framecount = 0;
    int last_framecount __attribute__((unused)) = 0;
    uint16_t *temp_frame = NULL;
    int lastFrameNumber = 0;
    count = 0;

    if(shmValid) {
        shm->statusByte = SHM_STATUS_READY;
        shmBufferPosition = 0;
        shmBufferPositionPrior = 0;
    }

    while(rtpConsumerRun)
    {
        begintp = std::chrono::steady_clock::now();
        grabbing = true;
        curFrame = &frame_ring_buffer[count % CPU_FRAME_BUFFER_SIZE];
        curFrame->reset();
        temp_frame = Camera->getFrameWait(lastFrameNumber, &this->camStatus);
        memcpy(curFrame->raw_data_ptr,temp_frame,frWidth*dataHeight*2);


        if(pixRemap)
        {
            apply_chroma_translate_filter(curFrame->raw_data_ptr);
            //curFrame->image_data_ptr = curFrame->raw_data_ptr;
        }


        curFrame->image_data_ptr = curFrame->raw_data_ptr;
        if(inverted)
        { // record the data from high to low. Store the pixel buffer in INVERTED order from the camera link
            for(uint i = 0; i < frHeight*frWidth; i++ )
                curFrame->image_data_ptr[i] = invFactor - curFrame->image_data_ptr[i];
        }

        if(setDarkStatusInFrame) {
            curFrame->image_data_ptr[obcStatusPixel] = darkStatusPixelVal;
        }

        shmBufferPosition = (shmBufferPositionPrior + 1)%shmFrameBufferSize;
        if(shmValid) {
            shm->writingFrameNum = shmBufferPosition;
            memcpy(shm->frameBuffer[shmBufferPosition],curFrame->raw_data_ptr, frHeight*frWidth);
        }


        // Calculating the filters for this frame
        if(!options.noGPU) {
            if(runStdDev)
            {
                sdvf->update_GPU_buffer(curFrame,std_dev_filter_N);
            }
            dsf->update(curFrame->raw_data_ptr,curFrame->dark_subtracted_data);
            mf->update(curFrame,count,meanStartCol,meanWidth,\
                       meanStartRow,meanHeight,frWidth,useDSF,\
                       whichFFT, lh_start, lh_end,\
                       cent_start, cent_end,\
                       rh_start, rh_end);

            mf->start_mean();
        }

        if((save_framenum > 0) || continuousRecording)
        {
            uint16_t * raw_copy = new uint16_t[frWidth*dataHeight];
            memcpy(raw_copy,curFrame->raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
            saving_list.push_front(raw_copy);
            save_framenum--;
        }

        framecount = *(curFrame->raw_data_ptr + 160); // The framecount is stored 160 bytes offset from the beginning of the data
        /*
        if(CHECK_FOR_MISSED_FRAMES_6604A && cam_type == CL_6604A)
        {
            if( (framecount - 1 != last_framecount) && (last_framecount != UINT16_MAX) )
            {
                std::cerr << "WARNING: MISSED FRAME " << framecount << std::endl;
            }
        }
        */

        finaltp = std::chrono::steady_clock::now();
        measuredDelta_micros_final = std::chrono::duration_cast<std::chrono::microseconds>(finaltp-begintp).count();
        meanDeltaArray[(++meanDeltaArrayPos)%meanDeltaSize] = measuredDelta_micros_final;

        if(shmValid) {
            if(measuredDelta_micros_final != 0)
                shm->fps = 1E6/measuredDelta_micros_final;
            shm->frameTime[shmBufferPosition] = finaltp.time_since_epoch() / std::chrono::milliseconds(1);
            shm->counter = count;
        }
        shmBufferPositionPrior = shmBufferPosition;


        last_framecount = framecount;
        count++;
        grabbing = false;
    }
    statusMessage("RTP Consumer Loop is done providing frames");
}

void take_object::pdv_loop() //Producer Thread (pdv_thread)
{
	count = 0;

    uint16_t framecount = 1;
    uint16_t last_framecount = 0;
	unsigned char* wait_ptr;


    mean_filter * mf = new mean_filter(curFrame,count,meanStartCol,meanWidth,\
                                       meanStartRow,meanHeight,frWidth,useDSF,\
                                       whichFFT, lh_start, lh_end,\
                                       cent_start, cent_end,\
                                       rh_start, rh_end);

    std::chrono::steady_clock::time_point finaltp;
    std::chrono::steady_clock::time_point begintp;

    if(shmValid) {
        shm->statusByte = SHM_STATUS_READY;
    }
    while(pdv_thread_run == 1)
    {	
        grabbing = true;
        begintp = std::chrono::steady_clock::now();
        curFrame = &frame_ring_buffer[count % CPU_FRAME_BUFFER_SIZE];
        curFrame->reset();
        if(closing)
        {
            pdv_thread_run = 0;
            break;

        } else {
            pdv_start_image(pdv_p); //Start another
            // Have seen Segmentation faults here on closing liveview:
            if(!closing) wait_ptr = pdv_wait_image(pdv_p);
        }
        cam_thread_start_complete=true;

        /* In this section of the code, after we have copied the memory from the camera link
         * buffer into the raw_data_ptr, we will check various parameters to see if we need to
         * modify the data based on our hardware.
         *
         * First, the data is stored differently depending on the type of camera, 6604A or B.
         *
         * Second, we may have to apply a filter to pixels which remaps the image based on the
         * way information is sent by some detectors.
         *
         * Third, we may need to invert the data range if a cable is inverting the magnitudes
         * that arrive from the ADC. This feature is also modified from the preference window.
         */
        memcpy(curFrame->raw_data_ptr,wait_ptr,frWidth*dataHeight*sizeof(uint16_t));
        if(pixRemap)
            apply_chroma_translate_filter(curFrame->raw_data_ptr);

        curFrame->image_data_ptr = curFrame->raw_data_ptr;
        if(inverted)
        { // record the data from high to low. Store the pixel buffer in INVERTED order from the camera link
            for(uint i = 0; i < frHeight*frWidth; i++ )
                curFrame->image_data_ptr[i] = invFactor - curFrame->image_data_ptr[i];
        }

        if(setDarkStatusInFrame) {
            curFrame->image_data_ptr[obcStatusPixel] = darkStatusPixelVal;
        }

        shmBufferPosition = (shmBufferPositionPrior + 1)%shmFrameBufferSize;
        if(shmValid) {
            shm->writingFrameNum = shmBufferPosition;
            memcpy(shm->frameBuffer[shmBufferPosition],curFrame->raw_data_ptr, frHeight*frWidth);
        }

        // Calculating the filters for this frame
        if(!options.noGPU) {

            if(runStdDev)
            {
                sdvf->update_GPU_buffer(curFrame,std_dev_filter_N);
            }
            dsf->update(curFrame->raw_data_ptr,curFrame->dark_subtracted_data);
            mf->update(curFrame,count,meanStartCol,meanWidth,\
                       meanStartRow,meanHeight,frWidth,useDSF,\
                       whichFFT, lh_start, lh_end,\
                       cent_start, cent_end,\
                       rh_start, rh_end);

            mf->start_mean();
        }

        if((save_framenum > 0) || continuousRecording)
        {
            uint16_t * raw_copy = new uint16_t[frWidth*dataHeight];
            memcpy(raw_copy,curFrame->raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
            saving_list.push_front(raw_copy);
            save_framenum--;
        }

        framecount = *(curFrame->raw_data_ptr + 160); // The framecount is stored 160 bytes offset from the beginning of the data
        if(CHECK_FOR_MISSED_FRAMES_6604A && cam_type == CL_6604A)
        {
            if( (framecount - 1 != last_framecount) && (last_framecount != UINT16_MAX) )
            {
                std::cerr << "WARNING: MISSED FRAME " << framecount << std::endl;
            }
        }
        last_framecount = framecount;
        count++;

        finaltp = std::chrono::steady_clock::now();
        measuredDelta_micros_final = std::chrono::duration_cast<std::chrono::microseconds>(finaltp-begintp).count();
        meanDeltaArray[(++meanDeltaArrayPos)%meanDeltaSize] = measuredDelta_micros_final;

        if(shmValid) {
            if(measuredDelta_micros_final != 0)
                shm->fps = 1E6/measuredDelta_micros_final;
            shm->frameTime[shmBufferPosition] = finaltp.time_since_epoch() / std::chrono::milliseconds(1);
            shm->counter = count;
        }
        shmBufferPositionPrior = shmBufferPosition;


        grabbing = false;
        if(closing)
        {
            pdv_thread_run = 0;
            break;
        }
    }
}
void take_object::savingLoop(std::string fname, unsigned int num_avgs, unsigned int num_frames) 
{
    // Frame Save Thread (saving_thread)

    // The main loop (pdvLoop, etc) of take_object will place frames into save_list,
    // and this thread will remove frames in save_list. While the data are being taken,
    // this thread will not empty the list.

    // This thread ends when the file finished being written to and the buffer is empty.

    std::ostringstream ss;
    ss << "Starting saveLoop. Thread ID: " << boost::this_thread::get_id();

    statusMessage(ss);

    if(options.debug) {
        if(num_avgs > 1) {
            statusMessage("Saving mode: averaging (float)");
        } else {
            statusMessage("Saving mode: uint16");
        }
    }


    if(savingData)
    {
        errorMessage("Saving loop hit but already saving data! Not saving this data!");
        return;
    } else {
        savingData = true;
    }

    savingMutex.lock();

    // if there is ".raw" already, then the hdr_fname shall be the same thing just without the .raw.
    // if there is not ".raw" then we just add ".hdr"
    std::string hdr_fname;
    if(fname.find(".")!=std::string::npos)
    {
        // The filename has ".", likely ".raw"
        //fname.replace(fname.find("."),std::string::npos,".raw");
        hdr_fname = fname.substr(0,fname.size()-3) + "hdr";
    }
    else
    {
        // The filename does not have "."
        hdr_fname=fname+".hdr";
    }

    FILE * file_target = fopen(fname.c_str(), "wb");
    int sv_count = 0;

    while(  (save_framenum != 0) || continuousRecording)
    {
        if(saving_list.size() > 2)
        {
            if(num_avgs == 1)
            {
                // This is our not-averaging save, where most saves go:
                if(saving_list.size() > 2) {
                    // We refuse to take the last item off the list.
                    // it can wait until we are completely done recording.
                    // This way the list remains valid in memory.
                    uint16_t * data = saving_list.back();
                    saving_list.pop_back();
                    fwrite(data,sizeof(uint16_t),frWidth*dataHeight,file_target); //It is ok if this blocks
                    delete[] data;
                    sv_count++;
                    if(sv_count == 1) {
                        save_count.store(1, std::memory_order_seq_cst);
                    }
                    else {
                        save_count++;
                    }
                }
            }
            else if(saving_list.size() >= num_avgs && num_avgs != 1)
            {
                float * data = new float[frWidth*dataHeight];
                for(unsigned int i2 = 0; i2 < num_avgs; i2++)
                {
                    uint16_t * data2 = saving_list.back();
                    saving_list.pop_back();
                    if(i2 == 0)
                    {
                        for(unsigned int i = 0; i < frWidth*dataHeight; i++)
                        {
                            data[i] = (float)data2[i];
                        }
                    }
                    else if(i2 == num_avgs-1)
                    {
                        for(unsigned int i = 0; i < frWidth*dataHeight; i++)
                        {
                            data[i] = (data[i] + (float)data2[i])/num_avgs;
                        }
                    }
                    else
                    {
                        for(unsigned int i = 0; i < frWidth*dataHeight; i++)
                        {
                            data[i] += (float)data2[i];
                        }
                    }
                    delete[] data2;
                }
                fwrite(data,sizeof(float),frWidth*dataHeight,file_target); //It is ok if this blocks
                delete[] data;
                sv_count++;
                if(sv_count == 1) {
                    save_count.store(1, std::memory_order_seq_cst);
                }
                else {
                    save_count++;
                }
                //std::cout << "save_count: " << std::to_string(save_count) << "\n";
                //std::cout << "list size: " << std::to_string(saving_list.size() ) << "\n";
                //std::cout << "save_framenum: " << std::to_string(save_framenum) << "\n";
            }
            else if(save_framenum == 0 && saving_list.size() < num_avgs)
            {
                warningMessage("Erasing saving_list");
                saving_list.erase(saving_list.begin(),saving_list.end());
            }
            else
            {
                //We're waiting for data to get added to the list...
                usleep(250);
            }
        }
        else
        {
            //We're waiting for data to get added to the list...
            usleep(250);
        }
    }

    // Almost done, let's take care of anything left in the buffer.

    statusMessage("Finished primary saving loop.");
    char message[128];
    sprintf(message, "Size of buffer: %ld", saving_list.size());
    statusMessage(message);
    if( (num_avgs==1) || (num_avgs==0)) {
        statusMessage("Finishing write...");
        while(saving_list.size() > 0) {
            statusMessage("Writing additional frame");
            uint16_t * data = saving_list.back();
            if(saving_list.size() > 0)
                saving_list.pop_back();
            fwrite(data,sizeof(uint16_t),frWidth*dataHeight,file_target);
            sv_count++;
            delete[] data;
        }
        statusMessage("Done with write.");
    } else {
        while(saving_list.size() > 0) {
            //statusMessage("Dropping additional frame at end that does not meet average interval.");
            uint16_t * data = saving_list.back();
            if(saving_list.size() > 0)
                saving_list.pop_back();
            // Since averaging is typically many frames (>100),
            // we cannot really average the last two or three frames
            // in a meaningfull way. Writing the data out will just
            // confuse people about the scale of the last few frames.
            //fwrite(data,sizeof(float),frWidth*dataHeight,file_target);
            delete[] data;
        }
    }

    fclose(file_target);
    std::string hdr_text;
    if( (num_avgs !=0) && (num_avgs !=1) )
    {
        hdr_text = "ENVI\ndescription = {LIVEVIEW raw export file, " + std::to_string(num_avgs) + " frames mean per line}\n";
    } else {
        hdr_text = "ENVI\ndescription = {LIVEVIEW raw export file}\n";
    }

    hdr_text= hdr_text + "samples = " + std::to_string(frWidth) +"\n";
    hdr_text= hdr_text + "lines   = " + std::to_string(sv_count) +"\n"; // save count, ie, number of frames in the file
    hdr_text= hdr_text + "bands   = " + std::to_string(dataHeight) +"\n";
    hdr_text+= "header offset = 0\n";
    hdr_text+= "file type = ENVI Standard\n";
    if((num_avgs != 1) && (num_avgs != 0))
    {
        hdr_text+= "data type = 4\n";
    }
    else
    {
        hdr_text+= "data type = 12\n";
    }
    hdr_text+= "interleave = bil\n";
    hdr_text+="sensor type = Unknown\n";
    hdr_text+= "byte order = 0\n";
    hdr_text+= "wavelength units = Unknown\n";
    //std::cout << hdr_text;
    std::ofstream hdr_target(hdr_fname);
    hdr_target << hdr_text;
    hdr_target.close();
    // What does this usleep do? --EHL
    if(sv_count == 1)
        usleep(500000);
    save_count.store(0, std::memory_order_seq_cst);
    statusMessage("Saving complete.");
    savingMutex.unlock();
    savingData = false;
}

void take_object::errorMessage(const char *message)
{
    if((!options.rtpCam) || (options.rtpNextGen))
    {
        std::cerr << "take_object: ERROR: " << message << std::endl;
    } else {
        g_critical("take_object: ERROR: %s", message);
    }
}

void take_object::warningMessage(const char *message)
{
    if((!options.rtpCam) || (options.rtpNextGen))
    {
        std::cout << "take_object: WARNING: " << message << std::endl;
    } else {
        g_message("take_object: WARNING: %s", message);
    }
}

void take_object::statusMessage(const char *message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "take_object: STATUS: " << message << std::endl;
    } else {
        g_message("take_object: STATUS: %s", message);
    }
}

void take_object::errorMessage(const string message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cerr << "take_object: ERROR: " << message << std::endl;
    } else {
        g_error("take_object: ERROR: %s", message.c_str());
    }
}

void take_object::warningMessage(const string message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "take_object: WARNING: " << message << std::endl;
    } else {
        g_message("take_object: WARNING: %s", message.c_str());
    }
}

void take_object::statusMessage(const string message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "take_object: STATUS: " << message << std::endl;
    } else {
        g_message("take_object: STATUS: %s", message.c_str());
    }
}

void take_object::statusMessage(std::ostringstream &message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "take_object: STATUS: " << message.str() << std::endl;
    } else {
        g_message("take_object: STATUS: %s", message.str().c_str());
    }
}
