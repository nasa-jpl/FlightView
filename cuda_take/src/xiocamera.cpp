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

    header.resize(size_t(headsize));
    std::fill(header.begin(), header.end(), 0);

    dummy.resize(size_t(frame_width * data_height));

    // Test image gradient:
    for (size_t n=0; n < dummy.size(); n++)
    {
        dummy.at(n) = ((float)n/((float)frame_width * data_height)) * 65535;
    }

    //std::fill(dummy.begin(), dummy.end(), 0);
    for (int n = 0; n < nFrames; n++) {
        frame_buf.emplace_back(std::vector<uint16_t>(size_t(frame_width * data_height), n*1000));
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
    //readLoopFuture.waitForFinished();
    LOG << " Completed XIO camera destructor for ID: " << this;
}

void XIOCamera::setDir(const char *dirname)
{
    LOG << ": Starting setDIR function for dirname " << dirname;
    is_reading = false;
    LOG << ": Clearing frame_buf. Initial size: " << frame_buf.size();
    while (!frame_buf.empty()) {
        frame_buf.pop_back();
    }

    /*
    if (readLoopFuture.isRunning()) {
        readLoopFuture.waitForFinished();
    }
    */

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
        if( (std::strcmp(ext.data(), "xio") != 0) and (std::strcmp(ext.data(), "decomp") != 0))
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
    //readLoopFuture = QtConcurrent::run(this, &XIOCamera::readLoop);
    fileListVecLocked = false;
    LOG << "Finished XIO setDir.";
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
                                 std::strcmp(ext.data(), "decomp") != 0)) {
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
    LOG << " Starting readfile";
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
            //readFile(); // circular?
            return;
        }

        LOG << ": Successfully opened " << ifname.data();
        dev_p.unsetf(std::ios::skipws);

        dev_p.read(reinterpret_cast<char*>(header.data()), headsize);

        std::streampos filesize(0);
        std::string ext = os::getext(ifname);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (!std::strcmp(ext.data(), "decomp")) {
            dev_p.seekg(0, std::ios::end);
            filesize = dev_p.tellg();
            dev_p.seekg(headsize, std::ios::beg);
            LOG << ": decomp read, filesize read from actual file stream is: " << filesize;
        } else {
            // convert the raw hex string to decimal, one digit at a time.
            filesize = int(header[7]) * 16777216 + int(header[6]) * 65536 + int(header[5]) * 256 + int(header[4]);
            LOG << ": Not-decomp, filesize read from header is: " << filesize;
        }

        framesize = static_cast<size_t>(filesize / nFrames);
        if (framesize == 0) { //If header reports a 0 filesize (invalid data), then skip this file.
            dev_p.close();
            LOG << ": Skipped file \"" << ifname.data() << "\" due to invalid data. size: "
                               << framesize << ", nFrames: " << nFrames;
        } else { //otherwise we load it
            validFile = true;
            LOG <<__PRETTY_FUNCTION__<< ": rsv - dev_p.seekg(): " << headsize;
            dev_p.seekg(headsize, std::ios::beg);

            LOG << ": File size is " << filesize << " bytes, which corresponds to a framesize of " << framesize << " bytes.";
            LOG << ": nFrames: " << nFrames;
            std::vector<uint16_t> zero_vec(size_t(frame_width * data_height) - (size_t(framesize) / sizeof(uint16_t)));
            std::fill(zero_vec.begin(), zero_vec.end(), 10000); // fill with value 10k so that I can spot it during debug.

            std::vector<uint16_t> copy_vec(size_t(framesize), 5000); // fill with value 5k so that I can spot it during debug.

            int read_size = 0;

            for (volatile int n = 0; n < nFrames; ++n) {
                dev_p.read(reinterpret_cast<char*>(copy_vec.data()), std::streamsize(framesize));
                read_size = dev_p.gcount();
                LL(3)  << ": Read " << read_size << " bytes from frame " << n << ", copy_vec size is: " << copy_vec.size();
                LL(3)  << "Rows: " << read_size / frame_width;
                //frame_buf.emplace_front(copy_vec);  // double-ended queue of a vector of uint_16.
                LL(2) << ": Size of frame_buf pre-push: " << frame_buf.size();

                while(frameVecLocked)
                    usleep(1);
                frameVecLocked = true;
                frame_buf.push_front(copy_vec);  // double-ended queue of a vector of uint_16.
                                                 // each frame_buf unit is a frame vector.
                LL(2) << ": Size of frame_buf post-push: " << frame_buf.size();

                // If the frame data is smaller than a frame, fill the rest of the frame with zeros:
                if (framesize / sizeof(uint16_t) < size_t(frame_width * data_height)) {
                    // We can't copy to space inside the frame_buf[n] that has not been allocated yet!!
                    LL(2) << ": size of frame_buf: " << frame_buf.size();
                    //qDebug() << __PRETTY_FUNCTION__ << "size of frame_buf[0]:" << frame_buf[0].size();
                    LL(2) << ": framesize: " << framesize << ",framesize in uint16 size:     " << framesize/sizeof(uint16_t);
                    LL(2) << ": frame_width: " << frame_width << ", data_height: " << data_height << ", product: " << frame_width*data_height;

                    if(frame_buf[size_t(n)].size() != 0)
                    {
                        //std::copy(zero_vec.begin(), zero_vec.end(), frame_buf[size_t(n)].begin() + framesize / sizeof(uint16_t));
                    } else {
                        LOG << ": ERROR, frame_buf at [" << n << "].size() is zero.";
                    }
                }
                frameVecLocked = false;
            }

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
    do {
        // Yeah yeah whatever it's a magic buffer size recommendation
        // if( we have fewer than 97 frames)
        if (frame_buf.size() <= 10) {
            waits = 1; //reset wait counter
            readFile(); // read in the next files, runs getFname() over and over

        } else {
            LOG << ": Waiting: Wait step: " << waits++ << ", frame_buf.size(): " << frame_buf.size(); // hapens 8 times in between files.
            usleep(10*tmoutPeriod);
        }
    } while (is_reading);
    LOG << ": finished readLoop(). is_reading must be false now: " << is_reading;
}

uint16_t* XIOCamera::getFrame()
{
    // This seems to run constantly.

    bool showOutput = ((getFrameCounter % 100) == 0);

    if(showOutput)
    {
        LOG << ": Getting frame: " << getFrameCounter << ", empty status: " << frame_buf.empty() << ", is_reading: " << is_reading << ", locked: " << frameVecLocked;
    }
    getFrameCounter++;

    // This is not ideal because it could become locked after,
    // but the idea is to not feed dummy frames unless we really have to,
    // so this gives us a chance to use the new data.


    // TODO: Strong mutex with timeout

    while(frameVecLocked)
        usleep(1);

    if ( (!frameVecLocked) && (!frame_buf.empty()) ) {
        frameVecLocked = true;
        temp_frame = frame_buf.back();
        // prev_frame = &temp_frame;
        frame_buf.pop_back(); // I have seen it crash here. We really need to assure exclusive access or use another method of storage.
        frameVecLocked = false;
        dummyrepeats = 0;
        LL(4) << "Returning good data.";
        usleep(1000 * 10);
        return temp_frame.data();
    } else {
        //if(showOutput) cout << __PRETTY_FUNCTION__ << ": Returning dummy data. locked: " << frameVecLocked << ", is_reading: " << is_reading << "empty status: " << frame_buf.empty() << endl;
        usleep(1000 * 1000); // 1 FPS, "timeout" style framerate, like the PDV driver.
        return NULL;
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
