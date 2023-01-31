#include "rtpcamera.hpp"

// Remove this from the final
// but keep in while diagnosing the build system:
#warning "Compiling rtpcamera.cpp"

RTPCamera::RTPCamera(int frWidth, int frHeight, int port, char *interface)
{
    LOG << "Starting RTP camera with width: " << frWidth << ", height: " << frHeight
        << ", port: " << port <<", network interface: " << interface;
    haveInitialized = false;

    this->port = port;
    this->frHeight = frHeight;
    this->frame_height = frHeight;
    this->data_height = frHeight;
    this->frWidth = frWidth;
    this->frame_width = frWidth;

    this->interface = interface;

    if (initialize())
    {
        LOG << "initialize successful";
    } else {
        LOG << "initialize fail";
    }
}

RTPCamera::~RTPCamera()
{
    LOG << "Running RTP camera destructor.";
}

bool RTPCamera::initialize()
{
    if(haveInitialized)
    {
        LOG << "Warning, running initializing function after initialization...";
        // continue for now.
    }


    data = g_new0(ProgramData, 1);

    data->loop = g_main_loop_new (NULL, FALSE);

    // Init for gstreamer:
    int argc = 0;
    char** argv = NULL;
    gst_init (&argc, &argv);

    // Arguments are first the module's special name, and second, a user-defined label
    source = gst_element_factory_make("udpsrc", "source");
    rtp = gst_element_factory_make("rtpvrawdepay", "rtp");
    appSink = gst_element_factory_make("appsink", "appsink");

    // Create pipe:
    sourcePipe = gst_pipeline_new ("sourcepipe");

    if (!sourcePipe || !source || !rtp || !appSink || !appSink) {
        g_printerr ("Not all gstreamer elements could be created.\n");
        LOG << "ERROR, gstreamer RTP camera source failed to be created.";
        return false;
    }

    gst_bin_add_many (GST_BIN (sourcePipe), source,  rtp,  appSink, NULL);
    gst_element_link_many(source, rtp, appSink, NULL);

    // TODO: Use parameters, int types, etc.
    g_object_set (source, "multicast-group", "ff02::1", NULL);
    g_object_set (source, "port", 5004, NULL);
    GstCaps *sourceCaps = gst_caps_new_simple( "application/x-rtp",
                                               "media", G_TYPE_STRING, "video",
                                               "clock-rate", G_TYPE_INT, 90000,
                                               "encoding-name", G_TYPE_STRING, "RAW",
                                               "sampling", G_TYPE_STRING, "RGB",
                                               "depth", G_TYPE_STRING, "8",
                                               "width", G_TYPE_STRING, "640",
                                               "height", G_TYPE_STRING, "481",
                                               "payload", G_TYPE_INT, 96, NULL);
    g_object_set (source, "caps", sourceCaps, NULL);

    // "data" is our i/o to the land of static functions and c functions.
    data->sourcePipe = sourcePipe;
    currentFrameNumber = &data->currentFrameNumber;
    doneFrameNumber = &data->doneFrameNumber;
    frameCounter = &data->frameCounter;

    g_object_set(appSink, "emit-signals", TRUE, "sync", FALSE, NULL);
    g_signal_connect(appSink, "new-sample", G_CALLBACK (on_new_sample_from_sink), data);

    timeoutFrame = (uint16_t*)calloc(frame_width*data_height, sizeof(uint16_t));

    for(int f = 0; f < guaranteedBufferFramesCount; f++)
    {
        guaranteedBufferFrames[f] = (uint16_t*)calloc(frame_width*data_height, sizeof(uint16_t));
        if(guaranteedBufferFrames[f] == NULL)
            abort();
    }

    data->buffer = guaranteedBufferFrames;
    busSourcePipe = gst_element_get_bus(data->sourcePipe);

    gst_bus_add_watch(busSourcePipe, (GstBusFunc) on_source_message, data);

    haveInitialized = true;
    return true;
}

void RTPCamera::streamLoop()
{
    // This should be a thread which can run without expectation of returning
    // Here, the stream is started and ran from.

    ret = gst_element_set_state (sourcePipe, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        //g_printerr ("Unable to set the sourcePipe to the playing state.\n");
        LOG << "Unable to set the sourcePipe to the playing state.";
        gst_object_unref (sourcePipe);
        abort();
    }

    //g_print ("Starting main gstreamer RTP loop.\n");
    LOG << "Starting main gstreamer RTP loop.";
    g_main_loop_run (data->loop); // this will run until told to stop
    //g_print ("Main gstreamer RTP loop ended.\n");
    LOG << "Main gstreamer RTP loop ended.";
    msg = gst_bus_timed_pop_filtered (busSourcePipe, GST_CLOCK_TIME_NONE,
                                      (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    on_source_message(busSourcePipe, msg, data);

    return;
}

static GstFlowReturn on_new_sample_from_sink(GstElement * elt, ProgramData * data)
{
    // This is the entry point from which we obtain the stream's data.
    // This function is called whenever there is a new frame in the source pipe appSink.

    g_print("new sample from sink\n");
    GstSample *sample;
    GstBuffer *app_buffer, *buffer;
    //GstElement *source;
    GstFlowReturn ret = GST_FLOW_OK;
    GstMapInfo map;
    //    guint8 *rdata;
    //    int dataLength;
    //    int i;
    //g_print ("%s\n", __func__);

    // Obtain sample
    sample = gst_app_sink_pull_sample (GST_APP_SINK (elt));
    buffer = gst_sample_get_buffer (sample);

    // Make a copy:
    app_buffer = gst_buffer_copy_deep (buffer);

    gst_buffer_map (app_buffer, &map, GST_MAP_WRITE);

    // Copy the data into liveview:
    siphonData (&map, data);

    gst_sample_unref (sample);
    gst_buffer_unmap (app_buffer, &map);

    (void)data;
    return ret;
}

static void siphonData (GstMapInfo* map, ProgramData *data)
{
    size_t dataLengthBytes;
    guint8 *rdata;

    dataLengthBytes = map->size;
    rdata = map->data;
    g_print ("%s dataLen = %zu\n", __func__, dataLengthBytes);

    // The dataLength indicates the entire length of the frame
    // in bytes. RGB format is 3 bytes per pixel,
    // GRAY format is 2 bytes per pixel.

    // The data are inside rdata, and are three bytes per pixel.
    // The intended RGB format is just 24-bit grayscale

    // TODO: How are the bytes packed?
    // 24 bits per pixel
    // R  G  B
    // But... R G is sufficient for 16-bit.

    // TODO unpack into guarenteedBufferFrames[];

#ifdef GST_HAS_GRAY
    // if this is defined, then the data are already 16-bit

    memcpy(guaranteedBufferFrames[data->currentFrame], rdata, dataLengthBytes);

#else
    // We need just the lower two

    //uint16_t* singleFrame = data->buffer[data->currentFrame];

    data->doneFrameNumber = ( data->currentFrameNumber - 1)% (guaranteedBufferFramesCount);

    uint16_t* singleFrame = data->buffer[data->currentFrameNumber];

    size_t pixelNum = 0;
    uint16_t pixel;
    for(size_t pbyte=0; pbyte < dataLengthBytes; pbyte+=3)
    {
        pixel = (rdata[pbyte]&0x00ff) | ((rdata[pbyte+1] << 8)&0xff00);
        // not used: rdata[pbyte+3];
        pixelNum = pbyte/3;
        singleFrame[pixelNum] = pixel;
    }

#endif
    data->currentFrameNumber = (data->currentFrameNumber+1) % (guaranteedBufferFramesCount);
    data->frameCounter++;

    //size_t bytesWritten = 0;
    // Write one frame to the fole, appending:
    //bytesWritten = fwrite(rdata, 1, dataLength, fp);
    //g_print("Wrote %lu bytes to binary file. Frame was %lu bytes.\n", bytesWritten, dataLength);
    //g_print("---start frame---\n");
    //  for (size_t i=0; i <= dataLength/2; i++) {
    //      //g_print("[%d]=%d, ", i, rdata[i]);
    //      rdata[i] = 0xff;
    //  }
    //g_print("---end frame---\n");

    // Frame timing metric
#if 0
    gettimeofday(&tval_before, NULL);
    timersub(&tval_before, &tval_after, &tval_result);

    double deltaTsec = tval_result.tv_sec + (double)tval_result.tv_usec/(double)1E6;
    if(deltaTsec != 0)
        printf("FPS: %f\n", 1.0/deltaTsec);
    tval_after = tval_before;
#endif

    return;
}

static gboolean on_source_message (GstBus * bus, GstMessage * message, ProgramData * data)
{
    // Called whenever there is a message from the source pipe.
    return true;
}

uint16_t* RTPCamera::getFrameWait(int lastFrameNumber, CameraModel::camStatusEnum *stat)
{
    // This function pauses until a new frame is received,
    // and then returns a pointer to the start of the new frame.
    int tap = 0;
    int pos = 0;
    if(camcontrol->pause)
    {
        *stat = CameraModel::camPaused;
        LL(4) << "Camera paused";
        return timeoutFrame;
    }
    while(*currentFrameNumber == lastFrameNumber)
    {
        *stat = camWaiting;
        usleep(FRAME_WAIT_MIN_DELAY_US);
        if(tap++ > MAX_FRAME_WAIT_TAPS)
        {
            LOG << "RTP Camera timeout waiting for frames. Total frame count: " << *frameCounter;
            *stat = camTimeout;
            return timeoutFrame;
        }
    }
    // TODO, check on this idea...
    *stat = camPlaying;
    pos = *doneFrameNumber; // doneFrameNumber is a number that is the most recent frame finished.
    return guaranteedBufferFrames[pos];
}

uint16_t* RTPCamera::getFrame(CameraModel::camStatusEnum *stat)
{
    uint16_t *ptr = NULL;

    // return latest-1 frame
    (void)stat;
    return ptr;
}

camControlType* RTPCamera::getCamControlPtr()
{
    return this->camcontrol;
}

void RTPCamera::setCamControlPtr(camControlType* p)
{
    this->camcontrol = p;
}















