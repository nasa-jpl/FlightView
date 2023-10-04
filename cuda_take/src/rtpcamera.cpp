#include "rtpcamera.hpp"

static struct timeval tval_before, tval_after, tval_result;
pthread_mutex_t rtpStreamLock;

//RTPCamera::RTPCamera(int frWidth, int frHeight, int port, const char *interface)


RTPCamera::RTPCamera(takeOptionsType opts)
{
    this->options = opts;
    RLOG << "Starting RTP camera with width: " << options.rtpWidth << ", height: " << options.rtpHeight
        << ", port: " << options.rtpPort <<", network interface: " << options.rtpInterface << ", multicast-group: " << options.rtpAddress
        << ", rgb mode: " << opts.rtprgb;
    haveInitialized = false;

    this->port = options.rtpPort;
    this->rtprgb = options.rtprgb;
    this->frHeight = options.rtpHeight;
    this->frame_height = frHeight;
    this->data_height = frHeight;
    this->frWidth = options.rtpWidth;
    this->frame_width = options.rtpWidth;

    this->interface = options.rtpInterface;

    if (initialize())
    {
        RLOG << "initialize successful";
    } else {
        RLOG << "initialize fail";
    }
}

RTPCamera::~RTPCamera()
{
    destructorRunning = true;
    RLOG << "Running RTP camera destructor.";
    g_main_loop_quit (data->loop);
//    GstMessage *lmsg = gst_bus_timed_pop_filtered (busSourcePipe, GST_CLOCK_TIME_NONE,
//                                        (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

//    if (lmsg != NULL)
//    {
//        on_source_message(busSourcePipe, lmsg, data);
//        gst_message_unref (lmsg);
//    }

    while(loopRunning)
    {
        usleep(10000);
    }
    RLOG << "Ran main_loop_quit.";

    //LOG << "unreferencing bus source pipe";
    //gst_object_unref (busSourcePipe);
    //LOG << "unreferenced.";

    RLOG << "setting state to GST_STATE_NULL";
    gst_element_set_state (sourcePipe, GST_STATE_NULL);
    RLOG << "State set.";

    if(busSourcePipe != NULL)
    {
        RLOG << "unreferencing bus source pipe";
        gst_object_unref (busSourcePipe);
        RLOG << "unreferenced.";
    }

    if(sourcePipe != NULL)
    {
        RLOG << "Unreferencing sourcePipe";
        gst_object_unref (sourcePipe);
        RLOG << "Unreferenced.";
    }


    RLOG << "Freeing buffer:";
    for(int b =0; b < guaranteedBufferFramesCount_rtp; b++)
    {
        if(guaranteedBufferFrames[b] != NULL)
        {
            free(guaranteedBufferFrames[b]);
        }
    }
    RLOG << "Done freeing buffer";

}

bool RTPCamera::initialize()
{
    if(haveInitialized)
    {
        RLOG << "Warning, running initializing function after initialization...";
        // continue for now.
    }


    data = g_new0(ProgramData, 1);

    data->loop = g_main_loop_new (NULL, FALSE);

    // Init for gstreamer:
    int argc = 0;
    char** argv = NULL;
//    int argc = 2;
//    const char* argv[4] = {"liveview\0", "-v\0", NULL, NULL};

    gst_init (&argc, &argv);

    // Arguments are first the module's special name, and second, a user-defined label
    source = gst_element_factory_make("udpsrc", "source");
    rtp = gst_element_factory_make("rtpvrawdepay", "rtp");
    appSink = gst_element_factory_make("appsink", "appsink");
    queue = gst_element_factory_make("queue", "queue");

    // Create pipe:
    sourcePipe = gst_pipeline_new ("sourcepipe");

    if (!sourcePipe || !source || !rtp || !appSink || !appSink || !queue) {
        g_printerr ("Not all gstreamer elements could be created.\n");
        RLOG << "ERROR, gstreamer RTP camera source failed to be created.";
        return false;
    }

    gst_bin_add_many (GST_BIN (sourcePipe), source,  rtp,  queue, appSink, NULL);
    gst_element_link_many(source, rtp, queue, appSink, NULL);

    // TODO: Use parameters, int types, etc.

    //g_object_set(queue, "min-threshold-bytes",  10E6, NULL); // had to remove this for newer gst
    //g_object_set(queue, "min-threshold-time",  1E6, NULL); // ns

    g_object_set(queue, "min-threshold-buffers", 4, NULL);

    //g_object_set (source, "multicast-group", "::1", NULL);
    //g_object_set (source, "port", 5004, NULL);
    if(options.rtpInterface != NULL)
        g_object_set (source, "multicast-iface", options.rtpInterface, NULL);

    if(options.rtpAddress != NULL)
        g_object_set (source, "multicast-group", options.rtpAddress, NULL);

    g_object_set (source, "port", options.rtpPort, NULL);

    RLOG << "RTP Caps: Height: " << options.rtpHeight;
    RLOG << "RTP Caps: Width: " << options.rtpWidth;

    // For reasons that I do not understand, height and width do not work as G_TYPE_INT
    // Therefore, I have converted them to strings:
    char widthStr[6] = {'\0'};
    char heightStr[6] = {'\0'};
    sprintf(widthStr, "%d", options.rtpWidth);
    sprintf(heightStr, "%d", options.rtpHeight);
    GstCaps *sourceCaps = NULL;

    if(rtprgb) {
        // RGB
        RLOG << "Setting up RTP camera for 24-bit RGB to 16-bit Grayscale frame-conversion transfer.";
        sourceCaps = gst_caps_new_simple( "application/x-rtp",
                                           "media", G_TYPE_STRING, "video",
                                           "clock-rate", G_TYPE_INT, 90000,
                                           "encoding-name", G_TYPE_STRING, "RAW",
                                           "sampling", G_TYPE_STRING, "RGB",
                                           "depth", G_TYPE_STRING, "8",
                                           "width", G_TYPE_STRING, widthStr,
                                           "height", G_TYPE_STRING, heightStr,
                                           "payload", G_TYPE_INT, 96, NULL);
    } else {
        // GRAY 16
        RLOG << "Setting up RTP camera for native 16-bit Grayscale frame transfer.";
        sourceCaps = gst_caps_new_simple( "application/x-rtp",
                                           "media", G_TYPE_STRING, "video",
                                           "clock-rate", G_TYPE_INT, 90000,
                                           "encoding-name", G_TYPE_STRING, "RAW",
                                           "sampling", G_TYPE_STRING, "GRAY",
                                           "depth", G_TYPE_STRING, "16",
                                           "width", G_TYPE_STRING, widthStr,
                                           "height", G_TYPE_STRING, heightStr,
                                           "payload", G_TYPE_INT, 96, NULL);

    }

    g_object_set (source, "caps", sourceCaps, NULL);

    // "data" is our i/o to the land of static functions and c functions.
    data->sourcePipe = sourcePipe;
    currentFrameNumber = &data->currentFrameNumber;
    doneFrameNumber = &data->doneFrameNumber;
    frameCounter = &data->frameCounter;

    g_object_set(appSink, "emit-signals", TRUE, "sync", FALSE, NULL);
    g_signal_connect(appSink, "new-sample", G_CALLBACK (on_new_sample_from_sink), data);

    timeoutFrame = (uint16_t*)calloc(frame_width*data_height, sizeof(uint16_t));

    for(int f = 0; f < guaranteedBufferFramesCount_rtp; f++)
    {
        guaranteedBufferFrames[f] = (uint16_t*)calloc(frame_width*data_height, sizeof(uint16_t));
        if(guaranteedBufferFrames[f] == NULL) {
            RLOG << "ERROR, cannot allocate memory for frame buffer.";
            RLOG << "Calling abort()";
            abort();
        }
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
    loopRunning = true;
    RLOG << "Starting rtp streamLoop(), setting status to PLAYING";

    ret = gst_element_set_state (sourcePipe, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr ("Unable to set the sourcePipe to the playing state.\n");
        RLOG << "Unable to set the sourcePipe to the playing state.";
        RLOG << "State is: GST_STATE_CHANGE_FAILURE";
        gst_object_unref (sourcePipe);
        std::cerr  << "Check the ethernet interface for valid IP configuration\n" << std::flush;
        std::cerr  << "and also verify the liveview program arguments.\n" << std::flush;
        if(!rtprgb) {
            std::cerr  << "Make sure that gstreamer was compiled with 16-bit grayscale support.\n" << std::flush;
        }
        std::cerr  << "Calling abort()\n" << std::flush;
        abort();
    }

    RLOG << "Starting main gstreamer RTP loop.";
    g_main_loop_run (data->loop); // this will run until told to stop
    RLOG << "Main gstreamer RTP loop ended.";

    loopRunning = false;
    return;
}

static GstFlowReturn on_new_sample_from_sink(GstElement * elt, ProgramData * data)
{
    // This is the entry point from which we obtain the stream's data.
    // This function is called whenever there is a new frame in the source pipe appSink.
    //LOG << "New RTP Sample";
    //g_print("new sample from sink\n");
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
    siphonDataRGB (&map, data);

    gst_sample_unref (sample);
    gst_buffer_unmap (app_buffer, &map);
    gst_buffer_unref(app_buffer);

    (void)data;
    return ret;
}

static void siphonDataRGB (GstMapInfo* map, ProgramData *data)
{
    pthread_mutex_lock(&rtpStreamLock);

    size_t dataLengthBytes;
    guint8 *rdata;

    dataLengthBytes = map->size;
    rdata = map->data;
    //g_print ("%s dataLen = %zu\n", __func__, dataLengthBytes);

    // The dataLength indicates the entire length of the frame
    // in bytes. RGB format is 3 bytes per pixel,
    // GRAY format is 2 bytes per pixel.

    // The data are inside rdata, and are three bytes per pixel.
    // The RGB data are 24 bits per pixel.

    int ftab[10] = {4, 5, 6, 7, 8, 9, 0, 1, 2, 3};

    data->doneFrameNumber = ftab[data->currentFrameNumber];
    uint16_t* singleFrame = data->buffer[data->currentFrameNumber];

    // Here, the first 16 bits of each 24-bit pixel are copied
    size_t pixelNum = 0;
    uint16_t pixel;
    for(size_t pbyte=0; pbyte < dataLengthBytes; pbyte+=3)
    {
        pixel = (rdata[pbyte]&0x00ff) | ((rdata[pbyte+1] << 8)&0xff00);
        // not used: rdata[pbyte+3];
        pixelNum = pbyte/3;
        singleFrame[pixelNum] = pixel;
        //LOG << "byte: " << pbyte;
    }

    data->currentFrameNumber = (data->currentFrameNumber+1) % (guaranteedBufferFramesCount_rtp);
    data->frameCounter++;

    // Frame timing metric
#ifdef FPS_MEAS_ACQ
    gettimeofday(&tval_before, NULL);
    timersub(&tval_before, &tval_after, &tval_result);

    double deltaTsec = tval_result.tv_sec + (double)tval_result.tv_usec/(double)1E6;
    if(deltaTsec != 0)
        printf("FPS: %f\n", 1.0/deltaTsec);
    tval_after = tval_before;
#endif
    pthread_mutex_unlock(&rtpStreamLock);
    return;
}

static void siphonDataGray (GstMapInfo* map, ProgramData *data)
{
    pthread_mutex_lock(&rtpStreamLock);

    size_t dataLengthBytes;
    guint8 *rdata;

    dataLengthBytes = map->size;
    rdata = map->data;
    //g_print ("%s dataLen = %zu\n", __func__, dataLengthBytes);

    // The dataLength indicates the entire length of the frame
    // in bytes. RGB format is 3 bytes per pixel,
    // GRAY format is 2 bytes per pixel.

    // The data are inside rdata, and are three bytes per pixel.
    // The intended RGB format is just 24-bit grayscale

    int ftab[10] = {4, 5, 6, 7, 8, 9, 0, 1, 2, 3};

    data->doneFrameNumber = ftab[data->currentFrameNumber];
    uint16_t* singleFrame = data->buffer[data->currentFrameNumber];
    // We copy the entire frame, without any manipulation
    memcpy((char*)singleFrame, rdata, dataLengthBytes);


    data->currentFrameNumber = (data->currentFrameNumber+1) % (guaranteedBufferFramesCount_rtp);
    data->frameCounter++;

    // Frame timing metric
#ifdef FPS_MEAS_ACQ
    gettimeofday(&tval_before, NULL);
    timersub(&tval_before, &tval_after, &tval_result);

    double deltaTsec = tval_result.tv_sec + (double)tval_result.tv_usec/(double)1E6;
    if(deltaTsec != 0)
        printf("FPS: %f\n", 1.0/deltaTsec);
    tval_after = tval_before;
#endif
    pthread_mutex_unlock(&rtpStreamLock);
    return;
}

static gboolean on_source_message (GstBus * bus, GstMessage * message, ProgramData * data)
{
    // Called whenever there is a message from the source pipe.
    //RLOG << "RTP Message: ";
    GstElement *source;

    GError *err;
    gchar *name, *debug, *full_message, *msgText;

    //g_print ("%s\n", __func__);

    g_print("SOURCE pipe (generator and appsink) message: ");

    switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
        g_print ("The source got dry (EOS = end of stream)\n");
        source = gst_bin_get_by_name (GST_BIN (data->sinkPipe), "appsrc");
        gst_app_src_end_of_stream (GST_APP_SRC (source));
        gst_object_unref (source);
        break;
    case GST_MESSAGE_ERROR:
        g_print ("Received error\n");
        goto showerror;
        g_main_loop_quit (data->loop);
        break;
    case GST_MESSAGE_STATE_CHANGED:
    {
        g_print("State changed.\n");

        GstState old_state, new_state, pending_state;
        gst_message_parse_state_changed (message, &old_state, &new_state, &pending_state);
        g_print ("Pipeline state changed from %s to %s:\n",
                 gst_element_state_get_name (old_state), gst_element_state_get_name (new_state));

        break;
    }
    case GST_MESSAGE_NEW_CLOCK:
        g_print("New clock\n");
        break;
    case GST_MESSAGE_STREAM_STATUS:
        g_print("Stream status.\n");
        break;
    case GST_MESSAGE_ASYNC_DONE:
        g_print("ASYNC done.\n");
        break;
    case GST_MESSAGE_STREAM_START:
        g_print("STREAM START\n");
        break;
    default:
        g_print("Other type: %d\n", message->type);
        break;
    }
    (void)bus;

    return true;


showerror:
    //g_print("Error breakdown: \n");
    gst_message_parse_error (message, &err, &debug);
    name = gst_object_get_path_string (message->src);
    msgText = gst_error_get_message (err->domain, err->code);

    if (debug)
        full_message =
                g_strdup_printf ("Error from element %s: %s\n%s\n%s", name, msgText,
                                 err->message, debug);
    else
        full_message =
                g_strdup_printf ("Error from element %s: %s\n%s", name, msgText,
                                 err->message);
    g_printf("Error message: %s\n", full_message);
    //    GST_ERROR_OBJECT (self, "ERROR: from element %s: %s\n", name, err->message);
    //    if (debug != NULL)
    //        GST_ERROR_OBJECT (self, "Additional debug info:\n%s\n", debug);
    g_main_loop_quit (data->loop);

    return true;
}

uint16_t* RTPCamera::getFrameWait(unsigned int lastFrameNumber, CameraModel::camStatusEnum *stat)
{
    // This function pauses until a new frame is received,
    // and then returns a pointer to the start of the new frame.
    volatile uint64_t tap = 0;
    volatile int lastFrameNumber_local_debug = lastFrameNumber;
    int pos = 0;
    pos = *doneFrameNumber; // doneFrameNumber is a number that is the most recent frame finished.

    if(camcontrol->pause)
    {
        *stat = CameraModel::camPaused;
        RLL(4) << "Camera paused";
        return timeoutFrame;
    }
    if(camcontrol->exit)
    {
        *stat = CameraModel::camDone;
        RLOG << "Closing down RTP stream";
        g_main_loop_quit (data->loop);
        return timeoutFrame;
    }
    // TODO: There are states where these numbers do not update
    // and that too should be a timeout.
    while(lastFrameDelivered==(unsigned int)pos)
    {
        *stat = camWaiting;
        usleep(FRAME_WAIT_MIN_DELAY_US);
        if(tap++ > MAX_FRAME_WAIT_TAPS)
        {
            *stat = camTimeout;
            RLOG << "RTP Camera timeout waiting for frames. Total frame count: " << *frameCounter << ", lastFrameDelivered: " << lastFrameDelivered << ", pos: " << pos;
            RLOG << "Timeout frame pixel zero: " << timeoutFrame[0]; // debug info
            return timeoutFrame;
        }
        pos = *doneFrameNumber;
    }
    // TODO, check on this idea...
    *stat = camPlaying;
    lastFrameDelivered = pos; // keep a copy around
    //LOG << "waitFrame: "// Remove this from the final
    // but keep in while diagnosing the build system:
    return guaranteedBufferFrames[pos];
    (void)lastFrameNumber_local_debug;
}

uint16_t* RTPCamera::getFrame(CameraModel::camStatusEnum *stat)
{
    // DO NOT USE
    (void)stat;
    RLOG << "ERROR, incorrect getFrame function called for RTP stream.";
    return timeoutFrame;
}

camControlType* RTPCamera::getCamControlPtr()
{
    return this->camcontrol;
}

void RTPCamera::setCamControlPtr(camControlType* p)
{
    this->camcontrol = p;
}















