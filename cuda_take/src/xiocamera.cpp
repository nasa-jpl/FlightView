#include "xiocamera.h"

XIOCamera::XIOCamera(int frWidth,
        int frHeight, int dataHeight

) : nFrames(32), framesize(0),
    //headsize(frWidth * int(sizeof(uint16_t))), image_no(0),
	headsize(1280), image_no(0),
    tmoutPeriod(100) // milliseconds
{
    frameVecLocked = true;

    cout <<  __PRETTY_FUNCTION__ << ": Starting xio camera class, ID: " << this << endl;
    source_type = XIO;
    camera_type = SSD_XIO;
    frame_width = frWidth;
    frame_height = frHeight;
    data_height = dataHeight;
    is_reading = false;
    cout <<  __PRETTY_FUNCTION__ << ": rsv - headsize: " << headsize << " frWidth: " << frWidth << ", data_height: " << data_height << ", size of pixel: " << int(sizeof(uint16_t))  << endl;

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
    cout << __PRETTY_FUNCTION__ << ": Initial size of frame_buf: " << frame_buf.size() << endl;
    cout << __PRETTY_FUNCTION__ << ": finished XIO Camera constructor for ID: " << this << endl;
    frameVecLocked = false;
    fileListVecLocked = false;
}

XIOCamera::~XIOCamera()
{
    cout << __PRETTY_FUNCTION__ << " Running XIO camera destructor for ID:   " << this << endl;
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
    cout << __PRETTY_FUNCTION__ << " Completed XIO camera destructor for ID: " << this << endl;
}

void XIOCamera::setDir(const char *dirname)
{
    cout << __PRETTY_FUNCTION__ << ": Starting setDIR function for dirname " << dirname << endl;
    is_reading = false;
    cout << __PRETTY_FUNCTION__ << ": Clearing frame_buf. Initial size: " << frame_buf.size() << endl;
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
            cout << __PRETTY_FUNCTION__ << ": emit timeout(), dir_data empty and running.load true" << endl;
            //emit timeout();
        }
        cout << __PRETTY_FUNCTION__ << ": dir_data empty." << endl;

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
    cout << __PRETTY_FUNCTION__ << ": os::listdir found this many files: " << fname_list.size() << " while looking in " << data_dir << endl;

    // Sort the frames in the product directory by filename, as mtime is unreliable.
    std::sort(fname_list.begin(), fname_list.end(), doj::alphanum_less<std::string>());

    for (auto &f : fname_list) {
        std::string ext = os::getext(f);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (f.empty())
            continue;
        if( (std::strcmp(ext.data(), "xio") != 0) and (std::strcmp(ext.data(), "decomp") != 0))
        {
            cout << "Rejecting file " << f << endl;
        } else {
            cout << "Accepting file " << f << endl;
            xio_files.push_back(f);
        }
    }

    cout << "Finished for loop in dir file search (part of setDir)" << endl;
    cout << "Found this many files: " <<  xio_files.size() << endl;

    running.store(true);
    //emit started(); // seems to work without this

    is_reading = true;
    //readLoopFuture = QtConcurrent::run(this, &XIOCamera::readLoop);
    fileListVecLocked = false;
    cout << "Finished XIO setDir." << endl;
}

std::string XIOCamera::getFname()
{
    cout << __PRETTY_FUNCTION__ << " Starting getFname() in XIO camera" << endl;
    std::string fname; // will return empty string if no unread files are found.
    std::vector<std::string> fname_list;
    bool has_file = false;
    if (data_dir.empty()) {
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
            cout << __PRETTY_FUNCTION__ << " Finished XIO getFname() early, found filename: " << fname << endl;
            fileListVecLocked = false;
            return fname;
        }
        /* if necessary, there may need to be code to sort the "frames" in the data directory
        * by product name, as mtime is unreliable.
        */
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
        cout << __PRETTY_FUNCTION__ << " Finished XIO getFname(), found filename: " << fname << endl;
    else
        cout << __PRETTY_FUNCTION__ << " Finished XIO getFname(), zero-length string." << endl;
    return fname;
}

void XIOCamera::readFile()
{
    cout << __PRETTY_FUNCTION__ << " Starting readfile" << endl;
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
            cout << __PRETTY_FUNCTION__ << ": All out of files, give up." << endl;
            this->is_reading = false; // otherwise we get stuck reading and not reading.
            return; //If we're out of files, give up
        }
        // otherwise check if data is valid
        dev_p.open(ifname, std::ios::in | std::ios::binary);
        if (!dev_p.is_open()) {
            cout << __PRETTY_FUNCTION__ << ": Could not open file" << ifname.data() << ". Does it exist?" << endl;
            dev_p.clear();
            //readFile(); // circular?
            return;
        }

        cout << __PRETTY_FUNCTION__ << ": Successfully opened " << ifname.data() << endl;
        dev_p.unsetf(std::ios::skipws);

        dev_p.read(reinterpret_cast<char*>(header.data()), headsize);

        std::streampos filesize(0);
        std::string ext = os::getext(ifname);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (!std::strcmp(ext.data(), "decomp")) {
            dev_p.seekg(0, std::ios::end);
            filesize = dev_p.tellg();
            dev_p.seekg(headsize, std::ios::beg);
            cout << __PRETTY_FUNCTION__ << ": decomp read, filesize read from actual file stream is: " << filesize<< endl;
        } else {
            // convert the raw hex string to decimal, one digit at a time.
            filesize = int(header[7]) * 16777216 + int(header[6]) * 65536 + int(header[5]) * 256 + int(header[4]);
            cout << __PRETTY_FUNCTION__ << ": Not-decomp, filesize read from header is: " << filesize<< endl;
        }

        framesize = static_cast<size_t>(filesize / nFrames);
        if (framesize == 0) { //If header reports a 0 filesize (invalid data), then skip this file.
            dev_p.close();
            cout  << __PRETTY_FUNCTION__ << ": Skipped file \"" << ifname.data() << "\" due to invalid data. size: "
                               << framesize << ", nFrames: " << nFrames<< endl;
        } else { //otherwise we load it
            validFile = true;
            cout <<__PRETTY_FUNCTION__<< ": rsv - dev_p.seekg(): " << headsize  << endl;
            dev_p.seekg(headsize, std::ios::beg);

            cout <<__PRETTY_FUNCTION__<< ": File size is " << filesize << " bytes, which corresponds to a framesize of " << framesize << " bytes."<< endl;
            cout <<__PRETTY_FUNCTION__<< ": nFrames: " << nFrames << endl;
            std::vector<uint16_t> zero_vec(size_t(frame_width * data_height) - (size_t(framesize) / sizeof(uint16_t)));
            std::fill(zero_vec.begin(), zero_vec.end(), 10000); // fill with value 10k so that I can spot it during debug.

            std::vector<uint16_t> copy_vec(size_t(framesize), 5000); // fill with value 5k so that I can spot it during debug.

            int read_size = 0;

            for (volatile int n = 0; n < nFrames; ++n) {
                dev_p.read(reinterpret_cast<char*>(copy_vec.data()), std::streamsize(framesize));
                read_size = dev_p.gcount();
                cout << __PRETTY_FUNCTION__ << ": Read " << read_size << " bytes from frame " << n << ", copy_vec size is: " << copy_vec.size() << endl;
                //frame_buf.emplace_front(copy_vec);  // double-ended queue of a vector of uint_16.
                cout << __PRETTY_FUNCTION__ << ": Size of frame_buf pre-push: " << frame_buf.size() << endl;

                while(frameVecLocked)
                    usleep(1);
                frameVecLocked = true;
                frame_buf.push_front(copy_vec);  // double-ended queue of a vector of uint_16.
                                                 // each frame_buf unit is a frame vector.
                frameVecLocked = false;
                cout << __PRETTY_FUNCTION__ << ": Size of frame_buf post-push: " << frame_buf.size() << endl;

                // If the frame data is smaller than a frame, fill the rest of the frame with zeros:
                if (framesize / sizeof(uint16_t) < size_t(frame_width * data_height)) {
                    // We can't copy to space inside the frame_buf[n] that has not been allocated yet!!
                    cout << __PRETTY_FUNCTION__ << ": size of frame_buf: " << frame_buf.size()<< endl;
                    //qDebug() << __PRETTY_FUNCTION__ << "size of frame_buf[0]:" << frame_buf[0].size();
                    cout << __PRETTY_FUNCTION__ << ": framesize: " << framesize << ",framesize in uint16 size:     " << framesize/sizeof(uint16_t)<< endl;
                    cout << __PRETTY_FUNCTION__ << ": frame_width: " << frame_width << ", data_height: " << data_height << ", product: " << frame_width*data_height<< endl;

                    if(frame_buf[size_t(n)].size() != 0)
                    {
                        while(frameVecLocked)
                            usleep(1);
                        frameVecLocked = true;
                        std::copy(zero_vec.begin(), zero_vec.end(), frame_buf[size_t(n)].begin() + framesize / sizeof(uint16_t));
                        frameVecLocked = false;
                    } else {
                        cout << __PRETTY_FUNCTION__ << ": ERROR, frame_buf at [" << n << "].size() is zero." << endl;
                    }
                }
            }

            running.store(true);
            cout << __PRETTY_FUNCTION__ << ": About done, emitting started signal."<< endl;
            //emit started(); // doesn't seem to be needed?
            dev_p.close();
        }
    }
    cout << __PRETTY_FUNCTION__ << ": is done."<< endl;
}

void XIOCamera::readLoop()
{
    cout << __PRETTY_FUNCTION__ << ": Entering readLoop()"<< endl;
    int waits = 1;
    do {
        // Yeah yeah whatever it's a magic buffer size recommendation
        if (frame_buf.size() <= 96) {
            waits = 1;
            readFile();

        } else {
            cout << __PRETTY_FUNCTION__ << ": Waiting: Wait step: " << waits++<< endl; // hapens 8 times in between files.
            usleep(1000*tmoutPeriod);
        }
    } while (is_reading);
    cout << __PRETTY_FUNCTION__ << ": finished readLoop()" << endl;
}

uint16_t* XIOCamera::getFrame()
{
    // This seems to run constantly.

    bool showOutput = ((getFrameCounter % 100) == 0);

    if(showOutput)
    {
        cout << __PRETTY_FUNCTION__ << ": Getting frame: " << getFrameCounter << ", empty status: " << frame_buf.empty() << ", is_reading: " << is_reading << ", locked: " << frameVecLocked << endl;
    }
    getFrameCounter++;

    // This is not ideal because it could become locked after,
    // but the idea is to not feed dummy frames unless we really have to,
    // so this gives us a chance to use the new data.

    while(frameVecLocked)
        usleep(10);

    if ( (!frameVecLocked) && (!frame_buf.empty()) ) {
        frameVecLocked = true;
        temp_frame = frame_buf.back();
        // prev_frame = &temp_frame;
        frame_buf.pop_back();
        frameVecLocked = false;
        return temp_frame.data();
    } else {
        //if(showOutput) cout << __PRETTY_FUNCTION__ << ": Returning dummy data. locked: " << frameVecLocked << ", is_reading: " << is_reading << "empty status: " << frame_buf.empty() << endl;
        return dummy.data();
    }
}

// Helper functions:
void XIOCamera::debugMessage(const char *msg)
{
    std::cout << "XIO Camera, " << __PRETTY_FUNCTION__ << ", debug message: " << msg << std::endl;
}

void XIOCamera::debugMessage(const std::string msg)
{
    std::cout << "XIO Camera, " << __PRETTY_FUNCTION__ << ", debug message: " << msg << std::endl;
}
