#include "acquire.hpp"
#include "fft.hpp"

// macOS pthread_setname_np compatibility
#ifdef __APPLE__
#define pthread_setname_np_compat(thread, name) pthread_setname_np(name)
#else
#define pthread_setname_np_compat(thread, name) pthread_setname_np(thread, name)
#endif


acquire::acquire(takeOptionsType options, int channel_num, int number_of_buffers,
                         int filter_refresh_rate, bool runStdDev)
{
    dsfMaskCollected = new bool(false);
    wrMaskCollected = new bool(false);
    changeOptions(options);
    initialSetup(channel_num, number_of_buffers,
                 filter_refresh_rate, runStdDev);
}

acquire::acquire(int channel_num, int number_of_buffers,
                         int frf, bool runStdDev)
{
    statusMessage("Starting acquire with default options.");
    dsfMaskCollected = new bool(false);
    wrMaskCollected = new bool(false);
    takeOptionsType options;
    options.theseAreDefault = true;
    options.xioCam = false;
    changeOptions(options);
    initialSetup(channel_num, number_of_buffers,
                 frf, runStdDev);
}

void acquire::initialSetup(int channel_num, int number_of_buffers,
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
    *dsfMaskCollected = false;
    *wrMaskCollected = false;
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
    continuousRecording.store(false, std::memory_order_seq_cst);
    save_framenum.store(0, std::memory_order_seq_cst);
    save_count=0;
    save_num_avgs=1;
    //saving_list.clear();

    camStatus = CameraModel::camUnknown;
}

acquire::~acquire()
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

#ifdef CAMERALINK
        int dummy;
        if(pdv_p)
        {
            pdv_wait_last_image(pdv_p,&dummy); //Collect the last frame to avoid core dump
            pdv_close(pdv_p);
        }
#endif
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
void acquire::changeOptions(takeOptionsType optionsIn)
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
        if(options.stdDevNSet) {
            this->std_dev_filter_N = options.stdDevN;
            statusMessage(std::string("Standard deviation buffer size (N) set to: ") + std::to_string(options.stdDevN));
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
        if(options.rtpCam) {
            if ( options.rtpHeight*options.rtpWidth > MAX_SIZE ) {
                errorMessage("This geometry is not supported, must increase MAX_SIZE in backend/include/constants.h");
                abort();
            }
        }
    }

    // Recalculate the frame-to-frame delay:
    deltaT_micros = 1000000.0 / options.targetFPS;
}

void acquire::acceptGPSDataPtr(basicGPS_t *basicGPSDataIn) {
    if(basicGPSDataIn != NULL) {
        this->basicGPSData = basicGPSDataIn;
        this->haveGPSDataPointer = true;
    } else {
        this->haveGPSDataPointer = false;
    }
}

void acquire::shmSetup()
{
    statusMessage("Preparing shared memory segment for images.");

    if(shmValid) {
        warningMessage("SHM already valid");
        return;
    }

    // 1. Get the dynamic dimensions from the object's members
    const int frameWidth = this->frWidth;
    const int frameHeight = this->frHeight;

    // 2. Calculate the required total size dynamically
    size_t metadataSize = sizeof(struct shmSharedDataStruct);

    // Calculate size of all frames in the buffer:
    // shmFrameBufferSize * (Width * Height * sizeof(uint16_t))
    size_t framePixelCount = (size_t)frameWidth * (size_t)frameHeight;
    size_t frameSizeInBytes = framePixelCount * sizeof(uint16_t);
    size_t bufferDataSize = (size_t)shmFrameBufferSize * frameSizeInBytes;

    // Total shared memory size
    size_t shmLen = metadataSize + bufferDataSize;

    // --- Standard SHM opening process (unchanged) ---
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
        snprintf(trunmessage, sizeof(trunmessage), "Could not truncate shared memory segment to %zu bytes.", shmLen);
        errorMessage(trunmessage);
        shmValid = false;
        goto cleanup;
    } else {
        snprintf(trunmessage, sizeof(trunmessage), "Truncated shared memory segment to %zu bytes (Metadata: %zu, Buffer: %zu).", shmLen, metadataSize, bufferDataSize);
        statusMessage(trunmessage);
    }

    shm = (shmSharedDataStruct*)mmap (0, shmLen, PROT_WRITE, MAP_SHARED, shmFd, 0);
    if( (shm == NULL) || (shm==MAP_FAILED) ) {
        errorMessage("Could not map memory shared memory segment to a local variable.");
        shmValid = false;
        goto cleanup;
    }

    // --- Initialization (Updated with dynamic size) ---
    shm->statusByte = SHM_STATUS_INITALIZING;
    shm->recordingDataToFile = false;
    shm->fps = 0.0;
    shm->counter = 0;
    shm->writingFrameNum = 0;

    // Set dynamic size metadata
    shm->bufferSizeFrames = shmFrameBufferSize;
    shm->frameHeight = frameHeight; // Dynamic height
    shm->frameWidth = frameWidth;   // Dynamic width
    shm->takingDark = false;

    // Clear fixed-size fields
    for(int i=0; i < shmFilenameBufferSize; i++) {
        shm->lastFilename[i] = '\0';
    }

    for(int i=0; i < shmFrameBufferSize; i++) {
        shm->frameTime[i] = 0;
    }

    // Zero out the entire variable-sized frame buffer block
    if (bufferDataSize > 0) {
        // Use memset to efficiently zero out the entire data buffer section.
        // The buffer starts exactly after the metadata struct.
        memset(SHM_FRAME_BUFFER_START(shm), 0, bufferDataSize);
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

void acquire::start()
{
    pdv_thread_run = 1;

    std::cout << "This version of backend was compiled on " << __DATE__ << " at " << __TIME__ << " using gcc " << __GNUC__ << std::endl;
    std::cout << "The compilation was perfromed by " << UNAME << " @ " << HOST << std::endl;

    pthread_setname_np_compat(pthread_self(), "TAKE");

#ifdef CAMERALINK
    this->pdv_p = NULL;
#endif
#ifdef USE_CUDA
    if(!options.noGPU) {

        size_t cudamem[10] = {0};
        size_t maxMemFound = 0;
        int bestDevice = 0;
        for(int i=0; i < getDeviceCount(); i++) {
            cudaGetDeviceProperties(&cdev, i);
            cudamem[i] = cdev.totalGlobalMem;
            printf("Device %d has name %s and memory %ld MiB\n",
                   i, cdev.name, cudamem[i]/1024/1024);
            if(cudamem[i] > maxMemFound) {
                maxMemFound = cudamem[i];
                bestDevice = i;
            }
        }

        cudaDevNumber = bestDevice;
        cudaDeviceNumberStatic = cudaDevNumber;

        printf("TAKE_OBJECT: Setting device number to %d\n", cudaDevNumber);
        cudaSetDevice(cudaDevNumber);
        int cudaDevNumCheck = -1;
        cudaGetDevice(&cudaDevNumCheck);
        printf("TAKE_OBJECT: Current device is: %d\n", cudaDevNumCheck);
        cudaGetDeviceProperties(&cdev, cudaDevNumCheck);
        printf("TAKE_OBJECT: CUDA device name: %s\n", cdev.name);
    } else {
        printf("TAKE_OBJECT: CUDA support compiled in but --noGPU specified, running CPU-only mode\n");
    }
#else
    printf("TAKE_OBJECT: Built without CUDA support, running CPU-only mode\n");
#endif

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
#ifndef CAMERALINK
        errorMessage("This version of FlightView does not have support for Camera Link.");
        errorMessage("See liveview.pro and enable the CAMERALINK define");
        abort();
#else
        this->pdv_p = pdv_open_channel(EDT_INTERFACE,0,this->channel);
        if(pdv_p == NULL) {
            std::cerr << "Could not open device channel. Is one connected?" << std::endl;
            return;
        }
        size = pdv_get_dmasize(pdv_p); // this size is only used to determine the camera type
        // actual grabbing of the dimensions
        if(options.rotate) {
            frWidth = pdv_get_height(pdv_p);
            dataHeight = pdv_get_width(pdv_p);
            frHeight = dataHeight;
        } else {
            frWidth = pdv_get_width(pdv_p);
            dataHeight = pdv_get_height(pdv_p);
            frHeight = dataHeight;
        }
#endif
    }

    switch(size) {
    case 481*640*sizeof(uint16_t): cam_type = CL_6604A; break;
    case 285*640*sizeof(uint16_t): cam_type = CL_6604A; break;
    case 480*640*sizeof(uint16_t): cam_type = CL_6604B; break;
    default: cam_type = CL_6604B; break;
    }
	setup_filter(cam_type);
    setup_filter(frHeight, frWidth);
    if(twoscomp) {
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
    dsf = new dark_subtraction_filter(frWidth,frHeight, dsfMaskCollected);
    wrf= new white_ref_filter(frWidth, frHeight, wrMaskCollected, dsf);
    sdvf = new std_dev_filter(frWidth,frHeight, cudaDevNumber);

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
        statusMessage("Creating an XIO camera acquire.");
        prepareFileReading(); // make a camera
        statusMessage("Creating an XIO camera thread inside acquire.");
        cam_thread = boost::thread(&acquire::fileImageCopyLoop, this);
        cam_thread_handler = cam_thread.native_handle();
        pthread_setname_np_compat(cam_thread_handler, "XIOCAM");
        statusMessage("Created thread.");
        while(!cam_thread_start_complete)
            usleep(100);
        // The idea is to hold off on doing anything else until some setup is finished.

        statusMessage("Creating XIO File reading thread reading_thread.");
        reading_thread = boost::thread(&acquire::fileImageReadingLoop, this);
        reading_thread_handler = reading_thread.native_handle();
        pthread_setname_np_compat(reading_thread_handler, "READING");
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
        rtpAcquireThread = boost::thread(&acquire::rtpNGStreamLoop, this);
        rtpAcquireThreadHandler = rtpAcquireThread.native_handle();
        pthread_setname_np_compat(rtpAcquireThreadHandler, "RTPNG Stream");
        statusMessage("Created RTP NextGen streamLoop() thread.");

        // At this point, the RTP camera is initialized and now it is running.
        // Data is being acquired if the stream source is emitting data,
        // and data is being copied into the guarenteed frame buffer of the RTPCamera.

        // These functions get the data into the rest of acquire:
        rtpConsumerRun = true;
        statusMessage("Creating RTP NextGen consumer thread to copy data into acquire");
        rtpCopyThread = boost::thread(&acquire::rtpConsumeFrames, this);
        rtpCopyThreadHandler = rtpCopyThread.native_handle();
        pthread_setname_np_compat(rtpCopyThreadHandler, "RTPNG Consume");
        statusMessage("Created RTP NextGen consumer thread.");

    } else if (options.rtpCam) {
        statusMessage("Starting RTP camera in take object.");
        cam_thread_start_complete = false;
        statusMessage("Preparing RTP camera");
        prepareRTPCamera();

        statusMessage("Creating boost thread for camera streamLoop()");
        rtpAcquireThread = boost::thread(&acquire::rtpStreamLoop, this);
        rtpAcquireThreadHandler = rtpAcquireThread.native_handle();
        pthread_setname_np_compat(rtpAcquireThreadHandler, "RTP Stream");
        statusMessage("Created RTP streamLoop() thread.");
        // At this point, the RTP camera is initialized and now it is running.
        // Data is being acquired if the stream source is emitting data,
        // and data is being copied into the guarenteed frame buffer of the RTPCamera.

        rtpConsumerRun = true;
        statusMessage("Creating RTP consumer thread to copy data into acquire");
        rtpCopyThread = boost::thread(&acquire::rtpConsumeFrames, this);
        rtpCopyThreadHandler = rtpCopyThread.native_handle();
        pthread_setname_np_compat(rtpCopyThreadHandler, "RTP Consume");
        statusMessage("Created RTP consumer thread.");

    } else {
#ifndef CAMERALINK
        errorMessage("Cameralink support was not compiled into this version of FlightView.");
        errorMessage("Please see the liveview.pro file for a #define CAMERALINK");
#else
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
        cam_thread = boost::thread(&acquire::pdv_loop, this);
        cam_thread_handler = cam_thread.native_handle();
        pthread_setname_np_compat(cam_thread_handler, "PDVCAM");
        //usleep(350000);
        while(!cam_thread_start_complete) usleep(1); // Added by Michael Bernas 2016. Used to prevent thread error when starting without a camera
#endif
    }
    statusMessage("Finished creating threads.");
}

void acquire::setInversion(bool checked, unsigned int factor)
{
    inverted = checked;
    invFactor = factor;
}
void acquire::set_twoscomp(bool checked )
{
    // This function was, unfortunately, briefly called paraPixRemap
    twoscomp = checked;
    std::cout << "2s Compliment Filter ";
    if(twoscomp) {
        std::cout << "ENABLED" << std::endl;
    } else {
        std::cout << "DISABLED" << std::endl;
    }
}

void acquire::enableDarkStatusPixelWrite(bool writeValues) {
    setDarkStatusInFrame = writeValues;
}

void acquire::acceptFrameHealthPtr(flightAppStatus_t *p) {
    frameHealth = p;
}

void acquire::startCapturingWR() {
    *wrMaskCollected = false;
    takingWR = true;
    wrf->start_mask_collection();
}

void acquire::setNDStatus(bool useND) {
    this->useND = useND;
    if(shmValid) {
        shm->usingNDFilter = useND;
    }
}

void acquire::finishCapturingWR() {
    //wrf->mask_mutex.lock();
    takingWR= false; // it's ok that the processing happens now. The point of this variable is to stop collecting additional WR frames.
    wrf_liveMean_thread = boost::thread( boost::bind(&white_ref_filter::finish_mask_collection, wrf));
    wrf_liveMean_thread_handler = wrf_liveMean_thread.native_handle();
    pthread_setname_np_compat(wrf_liveMean_thread_handler, "WR_MEAN");
}

void acquire::loadWR_entry(std::string filename_s, fileFormat_t fmt) {
    switch (fmt) {
    case fmt_float32:
        mask_thread = boost::thread(boost::bind(&acquire::loadWR_float, this, filename_s));
        break;
    case fmt_uint16:
        mask_thread = boost::thread(boost::bind(&acquire::loadWR_uint16, this, filename_s));
        break;
    default:
        errorMessage("WR Filetype not available.");
        break;
    }
}

void acquire::loadWR_float(std::string file_name) {
    if(readingWRFile)
        return;

    readingWRFile = true;
    // Loads a file containing a single 32-bit float frame.
    float *mask_in = new float[frWidth*frHeight];
    FILE *pFile;
    unsigned long size = 0;
    pFile  = fopen(file_name.c_str(), "rb");
    if(pFile == NULL) {
        errorMessage("error opening float WR file");
    } else {
        fseek (pFile, 0, SEEK_END); // non-portable
        size = ftell(pFile);
        if(size != (frWidth*frHeight*sizeof(float)))
        {
            errorMessage("Error: WR mask file does not match image size");
            fclose (pFile);
            delete [] mask_in;
            readingWRFile = false;
            return;
        }
        rewind(pFile);   // go back to beginning
        fread(mask_in,sizeof(float),frWidth * frHeight,pFile);
        fclose (pFile);
#ifdef VERBOSE
        std::cout << file_name << " read in "<< size << " bytes successfully " <<  std::endl;
#endif
    }

    // Dark sub:
    if(dsf->maskReady()) {
        dsf->static_dark_subtract(mask_in, mask_in);
    } else {
        errorMessage("DSF not ready, not applying to White Reference.");
    }
    // Load:
    wrf->load_mask(mask_in); // memcopy to stack variable

    delete [] mask_in;
    statusMessage("Completed White Reference load from float32 type.");
    readingWRFile = false;
}

void acquire::loadWR_uint16(std::string file_name) {
    if(readingWRFile)
        return;

    readingWRFile = true;
    std::ostringstream message;


    float * mean_frame = NULL;
    uint16_t * frames = NULL;
    unsigned int * input_array = NULL;

    unsigned int frame_size_numel = frHeight*frWidth;
    unsigned int nframes = 0;
    unsigned int pixel_size = sizeof(uint16_t);
    size_t items_read = 0;

    FILE * file = fopen(file_name.c_str(), "r");
    if(file == NULL)
    {
        message << "Error, could not load WR file " << file_name;
        errorMessage(message);
        readingDSFFile = false;
        return;
    }

    fseek(file, 0, SEEK_END);
    long int filesize = ftell(file);
    nframes = filesize / pixel_size / (frHeight * frWidth);
    fseek(file, 0, SEEK_SET);
    frames = (uint16_t *) malloc(filesize * pixel_size);
    if(frames == NULL)
    {
        errorMessage("Did not successfully allocate frames for white reference file");
        readingDSFFile = false;
        abort();
    }

    items_read = fread(frames, sizeof(uint16_t), filesize/pixel_size, file);

    message << "WR Load: Read      " << items_read << " pixels from " << file_name;
    statusMessage(message); message.str("");

    fclose(file);
    mean_frame = (float *) malloc(sizeof(float) * frame_size_numel);

    // The input_array is where the data are initially loaded.
    // These data can be type-converted after loading.
    input_array = (unsigned int *) malloc(sizeof(unsigned int) * nframes * frame_size_numel); // native size
    if(input_array == NULL)
    {
        errorMessage("Did not successfully allocate input_array for white reference file");
        readingDSFFile = false;
        abort();
    }
    // Convert uint16_t to unsigned int for GSL:
    // TODO: consider loading it in this way
#pragma omp parallel for num_threads(8)
    for(unsigned int nth_element = 0; nth_element < frame_size_numel * nframes; nth_element++)
    {
        input_array[nth_element] = (unsigned int)frames[nth_element];
    }
    // Process (create mean):
#pragma omp parallel for num_threads(8)
    for(unsigned int nth_frame_el = 0; nth_frame_el < frame_size_numel; nth_frame_el++)
    {
        // iterate over each pixel in a frame
        mean_frame[nth_frame_el] = (float)gsl_stats_uint_mean(input_array+nth_frame_el, frame_size_numel, nframes);
    }
    if(dsf->maskReady()) {
        dsf->static_dark_subtract(mean_frame, mean_frame);
    } else {
        errorMessage("DSF was not ready for White Reference processing from White Reference uint16 raw file");
    }

    wrf->load_mask(mean_frame); // memcpy out
    *wrMaskCollected = true;

    if(frames)
        free(frames);
    if(mean_frame)
        free(mean_frame);
    if(input_array)
        free(input_array);

    statusMessage("Completed White Reference load from uint16 type.");
    readingWRFile = false;
}

void acquire::startCapturingDSFMask()
{
    *dsfMaskCollected = false;

    dsf->start_mask_collection();
    if(shmValid) {
        shm->takingDark = true;
    }

    darkStatusPixelVal = obcStatusDark1;
}
void acquire::finishCapturingDSFMask()
{
    //statusMessage("Entering finishCapturingDSFMask()");
    // No point in a mutex here because it only protects the setup,
    // not the actual processing.
    //dsf->mask_mutex.lock();

    // launch the thread to take the average:
#ifdef VERBOSE
    std::cout << "Launching thread to compute mean." << std::endl;
#endif
    mask_liveMean_thread = boost::thread( boost::bind(&dark_subtraction_filter::finish_mask_collection, dsf));
    mask_liveMean_thread_handler = mask_liveMean_thread.native_handle();
    pthread_setname_np_compat(mask_liveMean_thread_handler, "MASKMEAN");

    //dsf->mask_mutex.unlock();
    *dsfMaskCollected = true;
    if(shmValid) {
        shm->takingDark = false;
    }
    darkStatusPixelVal = obcStatusScience;
    //statusMessage("Exiting finishCapturingDSFMask()");
}

void acquire::loadDSFMaskFromFramesU16(std::string file_name, fileFormat_t format)
{
    // Creates a mask from a file containing multiple frames
    // The frames are expected to be the same geometry as the
    // frame source, and the pixels are expected to be 16-bit unsigned int.

    // This function was largly copied from the main.cpp file of
    // the included "statscli" program found under "utils".

    if(readingDSFFile)
        return;

    readingDSFFile = true;
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
        readingDSFFile = false;
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
        readingDSFFile = false;
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
        readingDSFFile = false;
        abort();
    }

    if(format == fmt_uint16_2s)
    {
        // Convert the data first:
#pragma omp parallel for num_threads(8)
        for(unsigned int nth_element = 0; nth_element < frame_size_numel * nframes; nth_element++)
        {
            input_array[nth_element] = (unsigned int)( frames[nth_element] ^ (1<<15) );
        }
        // Process:
#pragma omp parallel for num_threads(8)
        for(unsigned int nth_frame_el = 0; nth_frame_el < frame_size_numel; nth_frame_el++)
        {
            // iterate over each pixel in a frame
            mean_frame[nth_frame_el] = (float)gsl_stats_uint_mean(input_array+nth_frame_el, frame_size_numel, nframes);
        }
    } else {
        // Convert uint16_t to unsigned int for GSL:
        // TODO: consider loading it in this way
#pragma omp parallel for num_threads(8)
        for(unsigned int nth_element = 0; nth_element < frame_size_numel * nframes; nth_element++)
        {
            input_array[nth_element] = (unsigned int)frames[nth_element];
        }
        // Process:
#pragma omp parallel for num_threads(8)
        for(unsigned int nth_frame_el = 0; nth_frame_el < frame_size_numel; nth_frame_el++)
        {
            // iterate over each pixel in a frame
            mean_frame[nth_frame_el] = (float)gsl_stats_uint_mean(input_array+nth_frame_el, frame_size_numel, nframes);
        }
    }

    dsf->load_mask(mean_frame); // memcopy to stack variable
    *dsfMaskCollected = true;

    if(frames)
        free(frames);
    if(mean_frame)
        free(mean_frame);
    if(input_array)
        free(input_array);

    statusMessage("Completed DSF load from uint16 type.");
    readingDSFFile = false;
}

void acquire::loadDSFMask_entry(std::string filename_s, fileFormat_t fmt) {
    // TODO: Mutex or even lockout
    if(readingDSFFile) {
        // This flag is set and cleared within the load/average functions.
        errorMessage("Already reading DSF mask. Cannot load concurrently.");
        return;
    }
    switch(fmt) {
    case fmt_uint16:
        statusMessage("Loading uing16_t DSF mask");
        mask_thread = boost::thread( boost::bind(&acquire::loadDSFMaskFromFramesU16, this, filename_s, fmt));
        break;
    case fmt_uint16_2s:
        statusMessage("Loading uing16_t with 2s compliment DSF mask");
        mask_thread = boost::thread( boost::bind(&acquire::loadDSFMaskFromFramesU16, this, filename_s, fmt));
        break;
    case fmt_float32:
        statusMessage("Loading float32 DSF mask");
        mask_thread = boost::thread(&acquire::loadDSFMaskFloat32, this, filename_s);
        break;
    default:
        errorMessage("Unable to load DSF mask from file, format is unknown.");
        return;
        break;
    }
    statusMessage("Mask thread started.");
    //mask_thread_handler = mask_thread.native_handle();
    //pthread_setname_np(mask_thread_handler, "MASK");
}

void acquire::loadDSFMaskFloat32(std::string file_name)
{
    if(readingDSFFile)
        return;

    readingDSFFile = true;
    // Loads a file containing a single 32-bit float frame.
    float *mask_in = new float[frWidth*frHeight];
    FILE *pFile;
    unsigned long size = 0;
    pFile  = fopen(file_name.c_str(), "rb");
    if(pFile == NULL) {
        errorMessage("Could not open raw DSF mask file.");
    } else {
        fseek (pFile, 0, SEEK_END); // non-portable
        size = ftell(pFile);
        if(size != (frWidth*frHeight*sizeof(float)))
        {
            errorMessage("Mask file does not match image size.");
            fclose (pFile);
            delete [] mask_in;
            readingDSFFile = false;
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
    delete [] mask_in;
    statusMessage("Completed DSF load from float32 type.");
    readingDSFFile = false;
}
void acquire::setStdDev_N(int s)
{
    this->std_dev_filter_N = s;
}

void acquire::toggleStdDevCalculation(bool enabled)
{
    this->runStdDev = enabled;
}

void acquire::updateVertOverlayParams(int lh_start_in, int lh_end_in,
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
    std::cout << "----- In acquire::updateVertOverlayParams\n";
    std::cout << "->lh_start:   " << lh_start <<   ", lh_end:   " << lh_end << std::endl;
    std::cout << "->rh_start:   " << rh_start <<   ", rh_end:   " << rh_end << std::endl;
    std::cout << "->cent_start: " << cent_start << ", cent_end: " << cent_end << std::endl;
    std::cout << "----- end acquire::updateVertOverlayParams -----\n";
    */
}

void acquire::updateVertRange(int br, int er)
{
    meanStartRow = br;
    meanHeight = er;
#ifdef VERBOSE
    std::cout << "meanStartRow: " << meanStartRow << " meanHeight: " << meanHeight << std::endl;
#endif
}
void acquire::updateHorizRange(int bc, int ec)
{
    meanStartCol = bc;
    meanWidth = ec;
#ifdef VERBOSE
    std::cout << "meanStartCol: " << meanStartCol << " meanWidth: " << meanWidth << std::endl;
#endif
}
void acquire::updateHorizPos(int horizPos) {
    meanStartCol = horizPos;
}

void acquire::updateVertPos(int vertPos) {
    meanStartRow = vertPos;
}

void acquire::changeFFTtype(FFT_t t)
{
    whichFFT = t;
}
void acquire::startSavingRaws(std::string raw_file_name, unsigned int frames_to_save, unsigned int num_avgs_save)
{
    if(frames_to_save==0)
    {
        continuousRecording.store(true, std::memory_order_seq_cst);
    } else {
        continuousRecording.store(false, std::memory_order_seq_cst);
    }
    
    save_framenum.store(0, std::memory_order_seq_cst);
    save_count.store(0, std::memory_order_seq_cst);
#ifdef VERBOSE
    printf("ssr called\n");
#endif
    if(frameSaveBuffer.size() > 0) {
        char msgb[160] = {'\0'};
        // This happens when are "left over" frames from prior recordings which were not written out completely.
        // It should not happen, since the buffer is cleared after each recording, but we will check here anyway.
        snprintf(msgb, sizeof(msgb), "frameSaveBuffer was not empty, size: %ld.",
                frameSaveBuffer.size());
        warningMessage(msgb);
    }
    while(frameSaveBuffer.size() > 0)
    {
        statusMessage("Clearing saveFrameBuffer...");
        uint16_t* data = frameSaveBuffer.try_dequeue();
        if(data)
            delete data;
        usleep(100);
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
    saving_thread = boost::thread(&acquire::savingLoop,this,raw_file_name,num_avgs_save,frames_to_save);
}
void acquire::stopSavingRaws()
{
    statusMessage("in stopSavingRaws()");
    continuousRecording.store(false, std::memory_order_seq_cst);
    save_framenum.store(0,std::memory_order_seq_cst);
    save_count.store(0,std::memory_order_seq_cst);
    save_num_avgs=1;
    if(shmValid) {
        shm->recordingDataToFile = false;
    }

#ifdef VERBOSE
    printf("Stop Saving Raws!");
#endif
}
unsigned int acquire::getDataHeight()
{
    return dataHeight;
}
unsigned int acquire::getFrameHeight()
{
    return frHeight;
}
unsigned int acquire::getFrameWidth()
{
    return frWidth;
}
bool acquire::std_dev_ready()
{
    return sdvf->outputReady();
}
std::vector<float> * acquire::getHistogramBins()
{
    return sdvf->getHistogramBins();
}
FFT_t acquire::getFFTtype()
{
    return whichFFT;
}

// private functions

void acquire::prepareFileReading()
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

void acquire::prepareRTPCamera()
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

void acquire::prepareRTPNGCamera() {
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

void acquire::fileImageReadingLoop()
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

void acquire::markFrameForChecking(uint16_t *frame)
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

bool acquire::checkFrame(uint16_t* Frame)
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

void acquire::clearAllRingBuffer()
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
        memcpy(curFrame->raw_data_ptr,zeroFrame,frWidth*dataHeight*2);
    }
    statusMessage("Done zero-setting memory in frame_ring_buffer");
}

void acquire::fileImageCopyLoop()
{
    // This thread copies data from the XIO Camera's buffer
    // and into curFrane of acquire. It is the "consumer"
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
                                           meanStartRow,meanHeight,frWidth,useDSF, useWR,\
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
        int ngFrameCount __attribute__((unused)) = 0;
        bool wasPaused = false;
        bool wasTestPattern = false;
        bool wasDone = false;

        while(fileReadingLoopRun && (!closing))
        {
            begintp = std::chrono::steady_clock::now();

            grabbing = true;
            curFrame = & frame_ring_buffer[count % CPU_FRAME_BUFFER_SIZE];
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


            if(twoscomp)
            {
                apply_2sComp_translate_filter(curFrame->raw_data_ptr);
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
            // Subtract the new frame from the dark mask,
            // updates the curFrame->dark_subtracted_data
            dsf->update(curFrame->raw_data_ptr,curFrame->dark_subtracted_data);
            mf->update(curFrame,count,meanStartCol,meanWidth,\
                       meanStartRow,meanHeight,frWidth,useDSF, useWR,\
                       whichFFT, lh_start, lh_end,\
                                               cent_start, cent_end,\
                                               rh_start, rh_end);

            mf->start_mean();

            if((save_framenum > 0) || continuousRecording.load(std::memory_order_seq_cst))
            {
                uint16_t * raw_copy = new uint16_t[frWidth*dataHeight];
                memcpy(raw_copy,curFrame->raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
                frameSaveBuffer.enqueue_overwrite(raw_copy);
                //saving_list.push_front(raw_copy);
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

int acquire::getMicroSecondsPerFrame()
{
    // Called by the frame_worker at regular intervals
    int nElements = (meanDeltaArrayPos < meanDeltaSize)?meanDeltaArrayPos:meanDeltaSize;
    // If max is 0, we have not actually taken a reading yet.
    if(nElements == 0)
        return 0;

    int sum = 0;
    for(int i=0; i < nElements; i++)
    {
        sum += meanDeltaArray[i];
    }
    this->fpsObserved = float(1E6)/(sum/nElements);
    //printf("FPS: %f\n", fpsObserved);

    return sum / nElements;
}

void acquire::setReadDirectory(const char *directory)
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

camControlType* acquire::getCamControl()
{
    return &cameraController;
}

void acquire::rtpStreamLoop()
{
    LOG << "Entering streamLoop";
    Camera->streamLoop();
}

void acquire::rtpNGStreamLoop() {
    LOG << "Entering streamLoop";
    Camera->streamLoop();
}

void acquire::rtpConsumeFrames()
{
    // This thread copies frames from the RTP Stream Loop
    // guarenteed buffer into the acquire object.
    // The frames are copied using Camera->getFrameWait
    // which waits for new frames.

    // Initializers just in case:
    save_framenum = 0;
    continuousRecording = false;
    fhFirstFrame = true;

    mean_filter * mf = new mean_filter(curFrame,count,meanStartCol,meanWidth,\
                                       meanStartRow,meanHeight,frWidth,useDSF, useWR,\
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


    // Performance profiling variables
    std::chrono::steady_clock::time_point t_op;
    long total_getframe_us = 0, total_memcpy1_us = 0, total_twoscomp_us = 0;
    long total_invert_us = 0, total_shm_us = 0, total_stddev_us = 0;
    long total_dark_us = 0, total_white_us = 0, total_mean_us = 0;
    int profile_count = 0;

    while(rtpConsumerRun)
    {
        begintp = std::chrono::steady_clock::now();
        grabbing = true;
        curFrame = &frame_ring_buffer[count % CPU_FRAME_BUFFER_SIZE];
        curFrame->reset();
        
        // TIME: getFrameWait
        t_op = std::chrono::steady_clock::now();
        temp_frame = Camera->getFrameWait(lastFrameNumber, &this->camStatus);
        total_getframe_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();
        
        // TIME: memcpy #1 (RTP buffer to frame buffer)
        t_op = std::chrono::steady_clock::now();
        memcpy(curFrame->raw_data_ptr,temp_frame,frWidth*dataHeight*2);
        total_memcpy1_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();

        // Frame header health checks on embedded metadata.
        // Performed on the raw frame *before* the 2's complement filter, since the
        // ancillary data lives in the frame's zero row and must be read before any
        // pixel transformation can alter it. All offsets below are byte offsets
        // (pixels are 16-bit).
        //   Check 1: 16-bit magic at offset 0x50 == 0xDEAD, 0xBABE, or 0x5EAD or 0x3ABE.
        //   Check 2: 32-bit frame count at offset 0x04 increments by 1 (or laps).
        //   Check 3: 32-bit PPS count at offset 0x00 ticks up ~once per second, never backwards.
        // Each warning is edge-triggered: it fires only when a check's status
        // changes, so a long run of identical failures yields a single message.
        if(frameHealth != NULL) {
            const uint8_t *rb = reinterpret_cast<const uint8_t*>(curFrame->raw_data_ptr);
            uint16_t magic;
            uint32_t frameCount32, ppsCount;
            memcpy(&magic,        rb + 0x50, sizeof(uint16_t));
            memcpy(&frameCount32, rb + 0x04, sizeof(uint32_t));
            memcpy(&ppsCount,     rb + 0x1C, sizeof(uint32_t));

            const std::chrono::steady_clock::time_point fhNow = std::chrono::steady_clock::now();

            // --- Check 1: magic word at offset 0x50 ---
            bool magicOk = (magic == 0xDEAD || magic == 0xBABE || magic == 0x5EAD || magic == 0x3ABE);
            if(magicOk != frameHealth->fh_magicOk) { // status changed since last frame
                frameHealth->fh_magicOk = magicOk;
                if(!magicOk) {
                    frameHealth->fh_magicOkSticky = false;
                    std::ostringstream m;
                    m << "Magic number check failed. Value at 0x50: 0x"
                      << std::hex << std::uppercase << magic;
                    warningMessage(m);
#ifdef FV_DEBUG_BUILD
                    printFrameHex(rb, 160);
#endif
                }
            }

            // --- Check 2: frame count increments by exactly 1 (or laps) ---
            const uint32_t frameCountStep = (uint32_t)(frameCount32 - fhLastFrameCount);
            bool frameCountOk = fhFirstFrame || (frameCountStep == 1u);
            if(frameCountOk != frameHealth->fh_frameCountOk) { // status changed since last frame
                frameHealth->fh_frameCountOk = frameCountOk;
                if(!frameCountOk) {
                    frameHealth->fh_frameCountOkSticky = false;
                    // frameCountStep==0 means the count didn't move at all (stalled/repeated
                    // frame); any larger step means (step-1) frames were never seen.
                    const uint64_t missingFrames = (frameCountStep == 0) ? 0 : (uint64_t)frameCountStep - 1;
                    std::ostringstream m;
                    m << "Frame count check failed. Frame count: " << frameCount32
                      << " Last frame count: " << fhLastFrameCount
                      << " Missing frames: " << missingFrames;
                    warningMessage(m);
#ifdef FV_DEBUG_BUILD
                    printFrameHex(rb, 160);
#endif
                }
            }
            fhLastFrameCount = frameCount32;

            // --- Check 3: PPS count (1 Hz GPS pulse counter) ---
            // The counter advances by one roughly once per second, so it stays
            // constant across most frames. We fail immediately if it steps backwards
            // (a genuine 32-bit rollover excepted), or if it stalls for longer than
            // one second plus a 25% margin (frame arrival timing is not guaranteed).
            bool ppsOk = true;
            uint64_t missingPpsTicks = 0;
            // Captured before fhLastPpsCount is possibly reassigned below, so the warning
            // message (built further down) always reports the count as of the *previous*
            // frame rather than the value that just overwrote it.
            const uint32_t fhPrevPpsCount = fhLastPpsCount;
            if(fhFirstFrame) {
                fhLastPpsCount = ppsCount;
                fhLastPpsChangeTime = fhNow;
            } else if(ppsCount > fhLastPpsCount) {
                // Advanced as expected.
                fhLastPpsCount = ppsCount;
                fhLastPpsChangeTime = fhNow;
            } else if(ppsCount < fhLastPpsCount) {
                // Stepped backwards: permit only a true 32-bit rollover (top wrap to near zero).
                bool rollover = (fhLastPpsCount > 0xF0000000u) && (ppsCount < 0x10000000u);
                if(!rollover) {
                    ppsOk = false; // flag this frame as a backward step
                    missingPpsTicks = (uint64_t)fhPrevPpsCount - (uint64_t)ppsCount;
                }
                // Resynchronize the baseline either way, so a single backward step
                // (e.g. a looping source file) does not latch the check into a
                // permanent failure -- the next frame is judged against this value.
                fhLastPpsCount = ppsCount;
                fhLastPpsChangeTime = fhNow;
            } else {
                // Unchanged: the counter must tick at least once per second (+25% margin).
                double secsSinceChange = std::chrono::duration_cast<std::chrono::duration<double>>(
                            fhNow - fhLastPpsChangeTime).count();
                if(secsSinceChange > 1.25) {
                    ppsOk = false; // PPS counter has stalled
                    missingPpsTicks = (uint64_t)secsSinceChange; // ~1 tick/sec expected: elapsed seconds approximates ticks missed
                }
            }
            if(ppsOk != frameHealth->fh_ppsCountOk) { // status changed since last frame
                frameHealth->fh_ppsCountOk = ppsOk;
                if(!ppsOk) {
                    frameHealth->fh_ppsCountOkSticky = false;
                    std::ostringstream m;
                    m << "PPS count check failed. PPS count: " << ppsCount
                      << " (0x" << std::hex << std::uppercase << ppsCount << std::dec
                      << ") Last PPS count: " << fhPrevPpsCount
                      << " (0x" << std::hex << std::uppercase << fhPrevPpsCount << std::dec << ")"
                      << " Missing PPS ticks (approx): " << missingPpsTicks;
                    warningMessage(m);
                }
            }

            fhFirstFrame = false;
        } else {
            warningMessage("Frame health pointer was NULL!");
        }

        // TIME: 2's complement
        if(twoscomp)
        {
            t_op = std::chrono::steady_clock::now();
            apply_2sComp_translate_filter(curFrame->raw_data_ptr);
            total_twoscomp_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();
            //curFrame->image_data_ptr = curFrame->raw_data_ptr;
        }

        curFrame->image_data_ptr = curFrame->raw_data_ptr;
        
        // TIME: inversion
        if(inverted)
        { // record the data from high to low. Store the pixel buffer in INVERTED order from the camera link
            t_op = std::chrono::steady_clock::now();
            for(uint i = 0; i < frHeight*frWidth; i++ )
                curFrame->image_data_ptr[i] = invFactor - curFrame->image_data_ptr[i];
            total_invert_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();
        }

        if(setDarkStatusInFrame) {
            curFrame->image_data_ptr[obcStatusPixel] = darkStatusPixelVal;
        }

        // TIME: shared memory copy
        shmBufferPosition = (shmBufferPositionPrior + 1)%shmFrameBufferSize;
        if(shmValid) {
            t_op = std::chrono::steady_clock::now();
            uint16_t* shm_frame_ptr = SHM_GET_FRAME_POINTER(shm, shmBufferPosition);
            shm->writingFrameNum = shmBufferPosition;
            //memcpy(shm->frameBuffer[shmBufferPosition],curFrame->raw_data_ptr, frHeight*frWidth*2);
            memcpy(shm_frame_ptr, curFrame->raw_data_ptr, shm->frameWidth * shm->frameHeight * sizeof(uint16_t));
            total_shm_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();
        }

        // TIME: Standard deviation filter (GPU/CPU-based, only if available)
        // Always upload frames to GPU to keep pipeline active, but skip computation when frameskipping
        if(sdvf != nullptr && !options.noGPU && runStdDev) {
            t_op = std::chrono::steady_clock::now();
            bool skip_stddev_compute = options.frameSkipSet && (count % options.frameSkip != 0);
            sdvf->update_GPU_buffer(curFrame, std_dev_filter_N, skip_stddev_compute);
            total_stddev_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();
        }

        // Calculating the other filters for this frame
        if( (!options.frameSkipSet) || ( options.frameSkipSet && (count%options.frameSkip ==0)) ) {
            
            // TIME: Dark subtraction, white reference, and mean filters (always CPU-based)
            // Update the available dark-subtracted frame
            // and, if we are recording a mask, update the recorded mask
            t_op = std::chrono::steady_clock::now();
            dsf->update(curFrame->raw_data_ptr,curFrame->dark_subtracted_data);
            total_dark_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();

            // TIME: Update the available white-reference frame
            // and, if we are recording a white reference, update the recorded mask
            //wrf->update(in, out);
            t_op = std::chrono::steady_clock::now();
            if(takingWR) {
                wrf->updateTaking(curFrame->dark_subtracted_data, curFrame->white_referenced_data);
            } else {
                wrf->updateFrame(curFrame->dark_subtracted_data, curFrame->white_referenced_data);
            }
            total_white_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();
            
            // TIME: Mean filter
            t_op = std::chrono::steady_clock::now();
            mf->update(curFrame,count,meanStartCol,meanWidth,\
                       meanStartRow,meanHeight,frWidth,useDSF, useWR,\
                       whichFFT, lh_start, lh_end,\
                       cent_start, cent_end,\
                       rh_start, rh_end);

            mf->start_mean();
            total_mean_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_op).count();
            
            profile_count++;
        }
        
        // Print profiling report every 400 processed frames (and reset)
        if(options.showStats && profile_count > 0 && profile_count % 400 == 0) {
            LOG << "=== Frame Processing Performance (avg over last " << profile_count << " frames) ===";
            LOG << "  getFrameWait:  " << (total_getframe_us / profile_count) << " µs";
            LOG << "  memcpy (RTP):  " << (total_memcpy1_us / profile_count) << " µs";
            if(twoscomp) LOG << "  2s complement: " << (total_twoscomp_us / profile_count) << " µs";
            if(inverted) LOG << "  inversion:     " << (total_invert_us / profile_count) << " µs";
            if(shmValid) LOG << "  shm copy:      " << (total_shm_us / profile_count) << " µs";
            if(sdvf != nullptr && !options.noGPU && runStdDev) 
                LOG << "  stddev filter: " << (total_stddev_us / profile_count) << " µs";
            LOG << "  dark subtract: " << (total_dark_us / profile_count) << " µs";
            LOG << "  white ref:     " << (total_white_us / profile_count) << " µs";
            LOG << "  mean filter:   " << (total_mean_us / profile_count) << " µs";
            long total_avg = (total_getframe_us + total_memcpy1_us + total_twoscomp_us + total_invert_us + 
                             total_shm_us + total_stddev_us + total_dark_us + total_white_us + total_mean_us) / profile_count;
            LOG << "  TOTAL:         " << total_avg << " µs (" << (total_avg/1000.0) << " ms)";
            
            // Reset counters for next 400 frames
            total_getframe_us = 0;
            total_memcpy1_us = 0;
            total_twoscomp_us = 0;
            total_invert_us = 0;
            total_shm_us = 0;
            total_stddev_us = 0;
            total_dark_us = 0;
            total_white_us = 0;
            total_mean_us = 0;
            profile_count = 0;
        }

        if((save_framenum.load(std::memory_order_seq_cst) > 0) || continuousRecording.load(std::memory_order_seq_cst))
        {
            uint16_t * raw_copy = new uint16_t[frWidth*dataHeight];
            memcpy(raw_copy,curFrame->raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
            // saving_list.push_front(raw_copy);
            this->frameSaveBuffer.enqueue_overwrite(raw_copy); // we always want overwrite just in case we actually need it (which we do not in practice)
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
            shm->fps = this->fpsObserved;

            shm->frameTime[shmBufferPosition] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            // This is not based on the system boot as an epoch, unfortunately.
            //shm->frameTime[shmBufferPosition] = finaltp.time_since_epoch() / std::chrono::milliseconds(1);
            shm->counter = count;
        }
        shmBufferPositionPrior = shmBufferPosition;


        last_framecount = framecount;
        count++;
        grabbing = false;
    }
    statusMessage("RTP Consumer Loop is done providing frames");
    if(mf)
        delete mf;
}

#ifdef CAMERALINK
void acquire::pdv_loop() //Producer Thread (pdv_thread)
{
	count = 0;

    uint16_t framecount = 1;
    uint16_t last_framecount = 0;
    unsigned char* wait_ptr = NULL;

    pcv_t pointerConverter;

    mean_filter * mf = new mean_filter(curFrame,count,meanStartCol,meanWidth,\
                                       meanStartRow,meanHeight,frWidth,useDSF, useWR,\
                                       whichFFT, lh_start, lh_end,\
                                       cent_start, cent_end,\
                                       rh_start, rh_end);

    std::chrono::steady_clock::time_point finaltp;
    std::chrono::steady_clock::time_point begintp;

    if(shmValid) {
        shm->statusByte = SHM_STATUS_READY;
        shmBufferPosition = 0;
        shmBufferPositionPrior = 0;
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

        if(options.rotate) {
            pointerConverter.uc = wait_ptr;
            // Note that the height and width are reversed in this call on purpose.
            this->rotate(pointerConverter.u16, curFrame->raw_data_ptr, frWidth, dataHeight);
        }

        if(options.remapPixels) {
            pointerConverter.uc = wait_ptr;
            // As it stands right now, you cannot rotate and remap,
            // this is because these steps include memcpy functionality,
            // and thus the rotated data are already in the raw_data_ptr, meaning,
            // the only avaliable rotated data are in the same location as the
            // destination.
            // The only way to solve this is to either:
            //   1. Use a temporary memory space for in between data, not ideal, causes extra memcpy
            //   2. Combine the translation filter with the rotation filter, so that both are done at the same time.
            apply_teledyne_translation_filter(pointerConverter.u16,curFrame->raw_data_ptr);
        }

        if( (!options.remapPixels) && (!options.rotate) ) {
            memcpy(curFrame->raw_data_ptr,wait_ptr,frWidth*dataHeight*sizeof(uint16_t));
        }

        if(twoscomp)
            apply_2sComp_translate_filter(curFrame->raw_data_ptr);

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
            uint16_t* shm_frame_ptr = SHM_GET_FRAME_POINTER(shm, shmBufferPosition);
            shm->writingFrameNum = shmBufferPosition;
            memcpy(shm_frame_ptr, curFrame->raw_data_ptr, shm->frameWidth * shm->frameHeight * sizeof(uint16_t));
            //memcpy(shm->frameBuffer[shmBufferPosition],curFrame->raw_data_ptr, frHeight*frWidth*2);
        }

        // Calculating the filters for this frame
        // Standard deviation filter (GPU/CPU-based, only if available)
        if(sdvf != nullptr && !options.noGPU && runStdDev) {
            sdvf->update_GPU_buffer(curFrame,std_dev_filter_N);
        }
        
        // Dark subtraction, white reference, and mean filters (always CPU-based)
        dsf->update(curFrame->raw_data_ptr,curFrame->dark_subtracted_data);
        if(takingWR) {
            wrf->updateTaking(curFrame->dark_subtracted_data, curFrame->white_referenced_data);
        } else {
            wrf->updateFrame(curFrame->dark_subtracted_data, curFrame->white_referenced_data);
        }
        mf->update(curFrame,count,meanStartCol,meanWidth,\
                   meanStartRow,meanHeight,frWidth,useDSF, useWR,\
                   whichFFT, lh_start, lh_end,\
                   cent_start, cent_end,\
                   rh_start, rh_end);

        mf->start_mean();

        if((save_framenum > 0) || continuousRecording.load(std::memory_order_seq_cst))
        {
            uint16_t * raw_copy = new uint16_t[frWidth*dataHeight];
            memcpy(raw_copy,curFrame->raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
            frameSaveBuffer.enqueue_overwrite(raw_copy);
            //saving_list.push_front(raw_copy);
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
            shm->fps = this->fpsObserved;
            shm->frameTime[shmBufferPosition] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            //shm->frameTime[shmBufferPosition] = finaltp.time_since_epoch() / std::chrono::milliseconds(1);
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
    if(mf)
        delete mf;
}
#endif


void acquire::rotate(uint16_t *input, uint16_t *output, int origHeight, int origWidth) {
    // Rotate the input into the output.

    // NOTE: output pointer is a static cuda memory allocation and must meet the rotated size!
    // See constants.h for the max size, which is allocated in frame_c.hpp

    // This is also effectivly a memcpy
    int p=0;
    int outPos = 0;
    int c = 0;
    // height and width reference the original (input) matrix dims

#pragma omp parallel for num_threads(8)
    for(p = 0; p < origWidth; p++) {
        outPos = origHeight*p;
        for(c=0; c < origHeight*origWidth; c=c+origWidth) {
            output[outPos+(c/origWidth)] = input[c+p];
        }
    }

    // Single thread method, which may be slightly faster for single thread only:
    //    for(p = 0; p < origWidth; p++) {
    //        for(c=0; c < origHeight*origWidth; c=c+origWidth) {
    //            output[outPos] = input[c+p]; outPos++;
    //        }
    //    }

}


void acquire::savingLoop(std::string filename_in, unsigned int num_avgs_in, unsigned int num_frames)
{
    // Frame Save Thread (saving_thread)

    // The main loop (pdvLoop, etc) of acquire will place frames into save_list,
    // and this thread will remove frames in save_list. While the data are being taken,
    // this thread will not empty the list.

    // This thread ends when the file finished being written to and the buffer is empty.

    std::ostringstream ss;
    ss << "Starting saveLoop. Thread ID: " << boost::this_thread::get_id();

    statusMessage(ss);
    if(savingData)
    {
        errorMessage("Saving loop hit but already saving data! Enforcing a delay. This is unsafe, it is better to wait between acquisitions.");
        std::string loopHitMessage = "Filename: " + filename_in;
        errorMessage(loopHitMessage);
        while(savingData) {
            usleep(1*1E6);
            statusMessage("...");
        }
        statusMessage("Continuing with acquisition request.");
        // This is dangerous, but it will probably be better than dropping the request.
        // return;
    }

    unsigned int num_avgs = num_avgs_in;
    std::string fname = filename_in;

    savingData = true;

    bool averagingEnabled;
    if( (num_avgs==1) || (num_avgs==0) ) {
        averagingEnabled = false;
    } else {
        averagingEnabled = true;
    }

    if(options.debug) {
        if(num_avgs > 1) {
            statusMessage("Saving mode: averaging (float)");
        } else {
            statusMessage("Saving mode: uint16");
        }
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
    int waitCount = 0;
    char messageFrames[128];

    while(  (save_framenum != 0) || continuousRecording.load(std::memory_order_seq_cst))
    {
        if(!averagingEnabled)
        {
            // This is our not-averaging save, where most saves go:
            uint16_t * data = frameSaveBuffer.try_dequeue();
            if(data) {
                fwrite(data,sizeof(uint16_t),frWidth*dataHeight,file_target); //It is ok if this blocks
                delete[] data;
                sv_count++;
                if(sv_count == 1) {
                    save_count.store(1, std::memory_order_seq_cst);
                } else {
                    save_count++;
                }
            } else {
                // null pointer returned, wait for more frames.
                // The while loop will break once continuous recording is false.
                waitCount++; // track for debugging
                usleep(1000);
            }
        } else {
            // Averaging enabled, buckle up...
            // Since we are on the consuming side, we can do the averaging here
            // and it will not stop up the production of frames,
            // assuming we are 'on average' quicker than the frames are produced.
            // If we are slower, then we can get in a situation where we fall behind
            // This would be indicated by the overwrite count being non-zero.
            float * data = new float[frWidth*dataHeight];
            unsigned int bufferAttemptCounter __attribute__((unused)) = 0; // for debugging
//            sprintf(messageFrames, "Top of save while loop: Frames left to save: %ld, buffered frames: %ld, frames to average: %u",
//                    save_framenum.load(), frameSaveBuffer.size(), num_avgs);
//            statusMessage(messageFrames);

            if( (save_framenum.load() + frameSaveBuffer.size()) < num_avgs ) {
                // trouble. We're not gonna get enough frames to
                // window-average another set. And then we get stuck.
                // The frames left to be captured, combined with the size of the frames in the buffer,
                // are not enough to write a num_avgs average frame.

                snprintf(messageFrames, sizeof(messageFrames), "Situation: Frames left to save: %u, buffered frames: %zu, frames to average: %u",
                        (unsigned int)save_framenum.load(), frameSaveBuffer.size(), num_avgs);
                statusMessage("Could not average last set of frames. Total collection length should be an integer multiple of the averaging window size.");
                statusMessage(messageFrames);
                break; // break out of while loop, do not hit the for loop below.
            }
            for(unsigned int i2 = 0; i2 < num_avgs; i2++)
            {
                uint16_t *data2 = NULL;
                while(data2 == NULL) {
                        data2 = frameSaveBuffer.try_dequeue();
                        bufferAttemptCounter++;
                        usleep(100);
               }

//                if(hitDoneCondition)
//                    break;

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
            } else {
                save_count++;
            }
        }
    }

    while(  (save_framenum != 0) && !continuousRecording.load(std::memory_order_seq_cst)) {
        // If we were in averaging mode and we tapped out due to the multiple,
        // then we must wait and make sure there are no more frames beign pushed into the buffer.
        // Otherwise, our clearing of the buffer is premature.
        usleep(100);
    }


    // Almost done, let's take care of anything left in the buffer.
    statusMessage("Finished primary saving loop.");
    char message[128];
    snprintf(message, sizeof(message), "Size of buffer after real-time saving: %ld", frameSaveBuffer.size());
    statusMessage(message); memset(message, 0, sizeof(message));
    snprintf(message, sizeof(message), "Number of overwrite conditions: %d", frameSaveBuffer.getOverrideCount());
    statusMessage(message); memset(message, 0, sizeof(message));
    snprintf(message, sizeof(message), "Number of empty read attempts: %d", frameSaveBuffer.getEmptyRequestCount());
    statusMessage(message); memset(message, 0, sizeof(message));
    snprintf(message, sizeof(message), "Number of waits: %d", waitCount);
    statusMessage(message); memset(message, 0, sizeof(message));
    int finishingCounter = 0;
    int emptyFinishingCounter = 0;

    if(!averagingEnabled) {
        statusMessage("Finishing write:");
        while(frameSaveBuffer.size() > 0) {
            statusMessage("Writing additional frame");
//            sprintf(message, "Write Position: %ld, read position: %ld, size: %ld",
//                    frameSaveBuffer.getWritePos(), frameSaveBuffer.getReadPos(), frameSaveBuffer.size());
//            statusMessage(message); memset(message, 0, sizeof(message));
            uint16_t * data = frameSaveBuffer.try_dequeue();
            if(data) {
                fwrite(data,sizeof(uint16_t),frWidth*dataHeight,file_target);
                sv_count++;
                delete[] data;
            } else {
                emptyFinishingCounter++;
            }
            finishingCounter++;
            if(finishingCounter > 1000) {
                // Now we have a problem. We will dump some debug out and break.
                snprintf(message, sizeof(message), "Write Position: %ld", frameSaveBuffer.getWritePos());
                statusMessage(message); memset(message, 0, sizeof(message));
                snprintf(message, sizeof(message), "Read Position: %ld", frameSaveBuffer.getReadPos());
                statusMessage(message); memset(message, 0, sizeof(message));
                snprintf(message, sizeof(message), "Size: %ld", frameSaveBuffer.size());
                statusMessage(message); memset(message, 0, sizeof(message));
                snprintf(message, sizeof(message), "emptyFinishingCounter: %d", emptyFinishingCounter);
                statusMessage(message); memset(message, 0, sizeof(message));
                break;
            }
        }
        statusMessage("Done with write.");
    } else {
        if(frameSaveBuffer.size() > 0)
            statusMessage("Dropping additional frame(s) at end that did not meet the averaging interval.");
        while(frameSaveBuffer.size() > 0) {
            statusMessage("Clearing buffer...");
            uint16_t * data = frameSaveBuffer.try_dequeue();
            if(data) {
                delete[] data;
            }
        }
        statusMessage("Done with write.");
    }

    fclose(file_target);
    frameSaveBuffer.clearStats();

    std::string hdr_text;
    if(averagingEnabled)
    {
        hdr_text = "ENVI\ndescription = {FlightView raw export file, " + std::to_string(num_avgs) + " frames mean per line}\n";
    } else {
        hdr_text = "ENVI\ndescription = {FlightView raw export file}\n";
    }

    hdr_text = hdr_text + "samples = " + std::to_string(frWidth) +"\n";
    hdr_text = hdr_text + "lines   = " + std::to_string(sv_count) +"\n"; // save count, ie, number of frames in the file
    hdr_text = hdr_text + "bands   = " + std::to_string(dataHeight) +"\n";

    if(haveGPSDataPointer && (basicGPSData!= NULL)) {
        if(basicGPSData->usingGPS) {
            // Note: GPS data may be as much as one minute behind the moment of the end of the recording.
            hdr_text = hdr_text + "latitude = " + std::to_string(basicGPSData->chk_latiitude) +"\n";
            hdr_text = hdr_text + "longitude = " + std::to_string(basicGPSData->chk_longitude) +"\n";
            hdr_text = hdr_text + "altitude = " + std::to_string(basicGPSData->chk_altitude) +"\n";
            hdr_text = hdr_text + "groundspeed = " + std::to_string(basicGPSData->chk_gndspeed) +"\n";
            hdr_text = hdr_text + "course = " + std::to_string(basicGPSData->chk_course) +"\n";
            hdr_text = hdr_text + "heading = " + std::to_string(basicGPSData->chk_heading) +"\n";
            hdr_text = hdr_text + "FPS = " + std::to_string(basicGPSData->fps) + "\n";
            hdr_text = hdr_text + "CollectionID = " + std::to_string(basicGPSData->collectionID) + "\n";
        }
    }

    hdr_text = hdr_text + "NDFilter = " + std::to_string(this->useND) + "\n";
    hdr_text+= "header offset = 0\n";
    hdr_text+= "file type = ENVI Standard\n";
    if(averagingEnabled)
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
    save_count.store(0, std::memory_order_seq_cst);
    snprintf(message, sizeof(message), "Saving Complete. Saved %d frames.", sv_count);
    statusMessage(message); memset(message, 0, sizeof(message));
    savingMutex.unlock();
    savingData = false;
}

void acquire::errorMessage(const char *message)
{
    if((!options.rtpCam) || (options.rtpNextGen))
    {
        std::cerr << "acquire: ERROR: " << message << std::endl;
    } else {
        g_critical("acquire: ERROR: %s", message);
    }
    pushMessage(message);
}

void acquire::warningMessage(const char *message)
{
    if((!options.rtpCam) || (options.rtpNextGen))
    {
        std::cout << "acquire: WARNING: " << message << std::endl;
    } else {
        g_message("acquire: WARNING: %s", message);
    }
    pushMessage(message);
}

void acquire::statusMessage(const char *message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "acquire: STATUS: " << message << std::endl;
    } else {
        g_message("take_object: STATUS: %s", message);
    }
    pushMessage(message);
}

void acquire::errorMessage(const string message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cerr << "acquire: ERROR: " << message << std::endl;
    } else {
        g_error("acquire: ERROR: %s", message.c_str());
    }
    pushMessage(message);
}

void acquire::warningMessage(const string message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "acquire: WARNING: " << message << std::endl;
    } else {
        g_message("take_object: WARNING: %s", message.c_str());
    }
    pushMessage(message);
}

void acquire::statusMessage(const string message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "acquire: STATUS: " << message << std::endl;
    } else {
        g_message("acquire: STATUS: %s", message.c_str());
    }
    pushMessage(message);
}
void acquire::errorMessage(std::ostringstream &message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "acquire: ERROR: " << message.str() << std::endl;
    } else {
        g_message("acquire: ERROR: %s", message.str().c_str());
    }
    pushMessage(message.str());
}

void acquire::warningMessage(std::ostringstream &message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "acquire: WARNING: " << message.str() << std::endl;
    } else {
        g_message("acquire: WARNING: %s", message.str().c_str());
    }
    pushMessage(message.str());
}

void acquire::statusMessage(std::ostringstream &message)
{
    if((!options.rtpCam) || (options.rtpNextGen)) {
        std::cout << "acquire: STATUS: " << message.str() << std::endl;
    } else {
        g_message("acquire: STATUS: %s", message.str().c_str());
    }
    pushMessage(message.str());
}

void acquire::pushMessage(const std::string &text, int timeoutMs)
{
    // Called from the acquisition thread (e.g. rtpConsumeFrames()) while tryGetMessage()
    // is called from frameWorker's thread. Never block acquisition waiting on the GUI
    // side to catch up -- if the lock isn't free almost immediately, drop this message
    // and move on. A dropped status/warning line is far cheaper than a stalled capture
    // loop, or the previous bug where an unsynchronized shared buffer produced log lines
    // with garbage trailing bytes from whatever message used to occupy messagePasser.
    std::unique_lock<std::timed_mutex> lock(messageMutex, std::chrono::milliseconds(timeoutMs));
    if(!lock.owns_lock())
        return;
    strncpy(this->messagePasser, text.c_str(), takeMessageSize-1);
    haveMessage = true;
}

bool acquire::tryGetMessage(std::string &out, int timeoutMs)
{
    std::unique_lock<std::timed_mutex> lock(messageMutex, std::chrono::milliseconds(timeoutMs));
    if(!lock.owns_lock() || !haveMessage)
        return false;
    out = messagePasser;
    haveMessage = false;
    return true;
}

void acquire::printFrameHex(const uint8_t *data, int numBytes)
{
    // Debug helper: dump the first numBytes of a frame as hex (e.g. "00 AA FF 22"),
    // 16 bytes per line, for inspecting the embedded ancillary data layout.
    // Intended to be called only from debug builds (see FV_DEBUG_BUILD).
    if(data == NULL)
        return;
    std::ostringstream hexDump;
    hexDump << "First " << numBytes << " bytes of frame:" << std::endl;
    hexDump << std::hex << std::uppercase << std::setfill('0');
    for(int i = 0; i < numBytes; i++) {
        hexDump << std::setw(2) << (unsigned int)data[i];
        hexDump << ((i % 16 == 15) ? "\n" : " ");
    }
    statusMessage(hexDump);
}
