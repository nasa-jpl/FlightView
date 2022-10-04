#include "xiocamera.h"

XIOCamera::XIOCamera(int frWidth,
        int frHeight, int dataHeight

) : nFrames(32), framesize(0),
    //headsize(frWidth * int(sizeof(uint16_t))), image_no(0),
	headsize(1280), image_no(0),
    tmoutPeriod(100) // milliseconds
{
    frameVecLocked = true;

    LOG << ": Starting xio camera class, ID: " << this;
    source_type = XIO;
    camera_type = SSD_XIO;
    frame_width = frWidth;
    frame_height = frHeight;
    data_height = dataHeight;
    is_reading = false;
    LOG << ": rsv - headsize: " << headsize << " frWidth: " << frWidth << ", data_height: " << data_height << ", size of pixel: " << int(sizeof(uint16_t));
    xioHeadsize = headsize;
    nFramesXio = nFrames;
    header.resize(size_t(headsize));
    std::fill(header.begin(), header.end(), 0);

    dummy.resize(size_t(frame_width * data_height));
    // dummyPtr is the size of two frames, this is just to make things safer.
    dummyPtr = (uint16_t*)malloc(frame_width * data_height * sizeof(uint16_t) * 2);
    if(dummyPtr == NULL)
        abort();

    // Test image gradients:
    // Values are zero to 65534, data type is uint16_t, may be substituted
    // in for the frame to verify data.
    for (size_t n=0; n < dummy.size(); n++)
    {
        dummy.at(n) = ((float)n/((float)frame_width * data_height)) * 65535;
    }

    for(int n=0; n < frame_width * data_height; n++)
    {
        dummyPtr[n] = ((float)n/((float)frame_width * data_height)) * 65535;
    }

    zero_vec = std::vector<uint16_t>(size_t(frame_width * data_height) - (size_t(framesize) / sizeof(uint16_t)));
    std::fill(zero_vec.begin(), zero_vec.end(), 10000); // fill with value 10k so that I can spot it during debug.

    //std::fill(dummy.begin(), dummy.end(), 0);
    for (int n = 0; n < nFrames; n++) {
        frame_buf.emplace_back(std::vector<uint16_t>(size_t(frame_width * data_height), n*1000));
    }

    // Frame buffer to hold guaranteed safe data. This buffer never chagnes size:
    for(int f = 0; f < guaranteedBufferFramesCount; f++)
    {
        guaranteedBufferFrames[f] = (uint16_t*)calloc(frame_width*data_height, sizeof(uint16_t));
        if(guaranteedBufferFrames[f] == NULL)
            abort();
    }

    LOG << ": Initial size of frame_buf: " << frame_buf.size() ;
    LOG << ": finished XIO Camera constructor for ID: " << this;
    frameVecLocked = false;
    fileListVecLocked = false;

    // LOG TEST:
    LOG << "Testing the log.";
    LOG << "Current log level: " << logginglevel;
    // Change in cudalog.h
    for(int n=0; n < 10; n++)
    {
        LL(n) << "Testing log level " << n;
    }
    LOG << "Done testing log.";
}

XIOCamera::~XIOCamera()
{
    LOG << " Running XIO camera destructor for ID:   " << this;
    running.store(false);
    //emit timeout();
    is_reading = false;
    while(frameVecLocked)
    {
        usleep(1);
    }
    frameVecLocked = true;
    while(fileListVecLocked)
    {
        usleep(1);
    }
    fileListVecLocked = true;

    free(dummyPtr);

    LOG << " Completed XIO camera destructor for ID: " << this;
}

void XIOCamera::setDir(const char *dirname)
{
    LOG << ": Starting setDIR function for dirname " << dirname;
    is_reading = false;
    LOG << ": Clearing frame_buf. Initial size: " << frame_buf.size();
    {
        std::lock_guard<std::mutex> lock(frame_buf_lock);
        while (!frame_buf.empty()) {
            frame_buf.pop_back();
        }
    }

    data_dir = dirname;
    if (data_dir.empty()) {
        if (running.load()) {
            running.store(false);
            LOG << ": emit timeout(), dir_data empty and running.load true";
            //emit timeout();
        }
        LOG << ": dir_data empty.";

        return;
    }

    while(fileListVecLocked)
    {
        // tap tap tap
        usleep(1);
    }

    fileListVecLocked = true;
    xio_files.clear();
    dev_p.clear();
    dev_p.close();
    image_no = 0;
    std::vector<std::string> fname_list;
    os::listdir(fname_list, data_dir);
    LOG << ": os::listdir found this many files: " << fname_list.size() << " while looking in " << data_dir;

    // Sort the frames in the product directory by filename, as mtime is unreliable.
    std::sort(fname_list.begin(), fname_list.end(), doj::alphanum_less<std::string>());

    for (auto &f : fname_list) {
        std::string ext = os::getext(f);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (f.empty())
            continue;
        if(  (std::strcmp(ext.data(), "xio")    != 0) and
             (std::strcmp(ext.data(), "decomp") != 0) and
             (std::strcmp(ext.data(), "raw")    != 0))
        {
            LOG << "Rejecting file " << f;
        } else {
            LOG << "Accepting file " << f;
            xio_files.push_back(f);
        }
    }

    LOG << "Finished for loop in dir file search (part of setDir)";
    LOG << "Found this many files: " <<  xio_files.size();

    running.store(true);
    //emit started(); // seems to work without this

    is_reading = true;
    fileListVecLocked = false;
    LOG << "Finished XIO setDir.";
}

void XIOCamera::setCamControlPtr(camControlType *p)
{
    this->camcontrol = p;
}

camControlType* XIOCamera::getCamControlPtr()
{
    return camcontrol;
}

std::string XIOCamera::getFname()
{
    LOG << " Starting getFname() in XIO camera";
    std::string fname; // will return empty string if no unread files are found.
    std::vector<std::string> fname_list;
    bool has_file = false;
    if (data_dir.empty()) {
        // This usually happens if the directory hasn't been set yet on startup.
        LOG << "Empty data_dir. Leaving early.";
        return fname;
    }

    while(fileListVecLocked)
    {
        usleep(1);
    }
    fileListVecLocked = true;

    if (image_no < xio_files.size()) {
        fname = xio_files[image_no++];
    } else {

        os::listdir(fname_list, data_dir);

        if (fname_list.size() < 1) {
            LOG << " Finished XIO getFname() early, found filename: " << fname;
            fileListVecLocked = false;
            return fname;
        }
        /* if necessary, there may need to be code to sort the "frames" in the data directory
        * by product name, as mtime is unreliable.
        */
        // Each time we ask for a new filename, we sort the entire list.
        // We also sort the list when the directory is set.

        std::sort(fname_list.begin(), fname_list.end(), doj::alphanum_less<std::string>());
        for (auto f = fname_list.end() - 1; f != fname_list.begin(); --f) {
            has_file = std::find(xio_files.begin(), xio_files.end(), *f) != xio_files.end();
            std::string ext = os::getext(*f);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if ((*f).empty() or (std::strcmp(ext.data(), "xio") != 0 and
                                 std::strcmp(ext.data(), "decomp") != 0 and
                                 std::strcmp(ext.data(), "raw") != 0  )) {
                continue;
            } else if (has_file) {
                break;
            } else {
                xio_files.emplace_back(*f);
            }

        }



        if (image_no < xio_files.size()) {
            if(sizeof(xio_files.at(image_no)) != 0 )
            {
                fname = xio_files.at(image_no++);
            }
        }
    }
    fileListVecLocked = false;

    if(fname.length() > 0)
        LOG << " Finished XIO getFname(), found filename: " << fname;
    else
        LOG << " Finished XIO getFname(), zero-length string.";
    return fname;
}

void XIOCamera::readFile()
{
    // Reads a SINGLE file, processing all available frames within the file.
    LOG << " Starting readfile";
    bool isRawFile = false;
    bool validFile = false;
    while(!validFile) {
        ifname = getFname();
        if (ifname.empty()) {
            if (dev_p.is_open()) {
                dev_p.close();
            }

            if (running.load()) {
                running.store(false);
                //emit timeout();
            }
            LOG << ": All out of files, give up. ifname from getFname() was an empty string.";
            this->is_reading = false; // otherwise we get stuck reading and not reading.
            return; //If we're out of files, give up
        }
        // otherwise check if data is valid
        dev_p.open(ifname, std::ios::in | std::ios::binary);
        if (!dev_p.is_open()) {
            LOG << ": Could not open file" << ifname.data() << ". Does it exist?";
            dev_p.clear();
            return;
        }

        LOG << ": Successfully opened " << ifname.data();

        if(ifname.size() > 3 && ifname.compare(ifname.size()-3, 3, "raw") == 0)
        {
            isRawFile = true;
        } else {
            isRawFile = false;
        }

        dev_p.unsetf(std::ios::skipws);

        dev_p.read(reinterpret_cast<char*>(header.data()), headsize);

        std::streampos filesize(0);
        std::string ext = os::getext(ifname);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (!std::strcmp(ext.data(), "decomp")) {
            headsize = xioHeadsize;
            nFrames = nFramesXio;
            dev_p.seekg(0, std::ios::end);
            filesize = dev_p.tellg();
            dev_p.seekg(headsize, std::ios::beg);
            //filesize = filesize - headsize; // will this help?
            LL(3) << ": decomp read, filesize read from actual file stream is: " << filesize;
        } else if (isRawFile){
            dev_p.seekg(0, std::ios::end);
            filesize = dev_p.tellg();
            nFrames = filesize / (frame_width * frame_height * sizeof(uint16_t));
            headsize = 0;
            dev_p.seekg(std::ios::beg);
        } else {
            headsize = xioHeadsize;
            nFrames = nFramesXio;
            // convert the raw hex string to decimal, one digit at a time.
            filesize = int(header[7]) * 16777216 + int(header[6]) * 65536 + int(header[5]) * 256 + int(header[4]);
            LL(9) << ": Not-decomp, filesize read from header is: " << filesize;
        }

        framesize = static_cast<size_t>(filesize / nFrames);
        if (framesize == 0) { //If header reports a 0 filesize (invalid data), then skip this file.
            dev_p.close();
            LL(8) << ": Skipped file \"" << ifname.data() << "\" due to invalid data. size: "
                               << framesize << ", nFrames: " << nFrames;
        } else { //otherwise we load it
            validFile = true;
            LOG <<__PRETTY_FUNCTION__<< ": rsv - dev_p.seekg(): " << headsize;
            dev_p.seekg(headsize, std::ios::beg);

            LOG << ": File size is " << filesize << " bytes, which corresponds to a framesize of " << framesize << " bytes.";
            LOG << ": nFrames: " << nFrames;
/*
            if( (int)(framesize / sizeof(uint16_t)) > (int)(frame_width * data_height))
                abort();
            std::vector<uint16_t> zero_vec(size_t(frame_width * data_height) - (size_t(framesize) / sizeof(uint16_t)));
            std::fill(zero_vec.begin(), zero_vec.end(), 10000); // fill with value 10k so that I can spot it during debug.
*/


            // std::vector<uint16_t> copy_vec(size_t(framesize), 5000); // fill with value 5k so that I can spot it during debug.
            std::vector<uint16_t> copy_vec(size_t(frame_height * frame_width * sizeof(uint16_t)), 5000); // fill with value 5k so that I can spot it during debug.

            int read_size = 0;

            {
                std::lock_guard<std::mutex> lock(frame_buf_lock); // wait until we have a lock
                for (volatile int n = 0; n < nFrames; ++n) {
                    //copy_vec.clear();
                    //copy_vec.resize(size_t(frame_height * frame_width * sizeof(uint16_t)));
                    dev_p.read(reinterpret_cast<char*>(copy_vec.data()), std::streamsize(framesize));
                    read_size = dev_p.gcount();
                    LL(3)  << ": Read " << read_size << " bytes from frame " << n << ", copy_vec size is: " << copy_vec.size();
                    LL(3)  << "Rows: " << read_size / frame_width;
                    LL(2) << ": Size of frame_buf pre-push: " << frame_buf.size();

                    if (framesize / sizeof(uint16_t) < size_t(frame_width * data_height)) {
                        std::copy(zero_vec.begin(), zero_vec.begin() + ( ((frame_width * data_height)) - (framesize / sizeof(uint16_t)) ), copy_vec.data() + framesize / sizeof(uint16_t));
                    }
                    frameVecLocked = true;
                    frame_buf.push_front(copy_vec);  // double-ended queue of a vector of uint_16.
                    // each frame_buf unit is a frame vector.
                    LL(2) << ": Size of frame_buf post-push: " << frame_buf.size();
                    frameVecLocked = false;
                }
            } // end lock

            running.store(true);
            LOG << ": About done, emitting started signal.";
            //emit started(); // doesn't seem to be needed?
            dev_p.close();
        }
    }
    LOG << ": is done.";
}

void XIOCamera::readLoop()
{
    LOG << ": Entering readLoop()";
    int waits = 1;
    bool sizeSmall = false;

    do {
        {
            std::lock_guard<std::mutex> lock(frame_buf_lock);
            sizeSmall = (frame_buf.size() <= 96);
        }
        // if( we have fewer than 97 frames)
        if (sizeSmall) {
            waits = 1; //reset wait counter
            readFile(); // read in the next files, runs getFname() over and over

        } else {
            waits++;
            //LOG << ": Waiting: Wait step: " << waits++ << ", frame_buf.size(): " << frame_buf.size(); // hapens 8 times in between files.
            //LOG << ": Waiting: Wait step: " << waits++; // hapens 8 times in between files.
        }
    } while (is_reading);
    LOG << ": finished readLoop(). is_reading must be false now: " << is_reading;
}

uint16_t* XIOCamera::getFrame(CameraModel::camStatusEnum *stat)
{
    // This seems to run constantly.
    uint16_t *frameVecPtr = NULL;

    if(camcontrol->pause && *stat != camDone)
    {
        // TODO: replace good with an enumeration for status
        *stat = CameraModel::camPaused;
        LL(4) << "Camera paused.";
        return doneFramePtr;
    }

    bool showOutput = ((getFrameCounter % 100) == 0);
    {
        std::lock_guard<std::mutex> lock(frame_buf_lock); // gone once out of scope.
        if(showOutput)
        {
            LOG << ": Getting frame: " << getFrameCounter << ", empty status: " << frame_buf.empty() << ", is_reading: " << is_reading << ", locked: " << frameVecLocked;
        }
        getFrameCounter++;

        if ( (!frame_buf.empty()) ) {
            LL(4) << "Returning good data.";
            frameVecLocked = true;
            //temp_frame = frame_buf.back();
            frameVecPtr = frame_buf.back().data();
            if(frameVecPtr == NULL)
                abort();
            // This memory was allocated with the class constructor; it is always valid and
            // does not change size. Data are copied from the locked vector into
            // this guarenteed space so that the calling function will never come up
            // with invalid memory.
            memcpy(guaranteedBufferFrames[gbPos%guaranteedBufferFramesCount], frameVecPtr, frame_height*frame_width*sizeof(uint16_t));

            frame_buf.pop_back();
            frameVecLocked = false;
            doneFramePtr = guaranteedBufferFrames[gbPos%guaranteedBufferFramesCount];
            gbPos++;
            *stat = camPlaying;
            return doneFramePtr;
        } else {
            //if(showOutput) cout << __PRETTY_FUNCTION__ << ": Returning dummy data. locked: " << frameVecLocked << ", is_reading: " << is_reading << "empty status: " << frame_buf.empty() << endl;
            usleep(1000 * 1000); // 1 FPS, "timeout" style framerate, like the PDV driver.
            *stat = camDone;
            return dummyPtr;
        }
    }
}

// Helper functions:
void XIOCamera::debugMessage(const char *msg)
{
    // std::cout << "XIO Camera, " << __PRETTY_FUNCTION__ << ", debug message: " << msg << std::endl;
    LOG << "DEBUG MESSAGE: " << msg;
}

void XIOCamera::debugMessage(const std::string msg)
{
    // std::cout << "XIO Camera, " << __PRETTY_FUNCTION__ << ", debug message: " << msg << std::endl;
    LOG << "DEBUG MESSAGE: " << msg;
}
