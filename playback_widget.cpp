#include <QFileDialog>
#include <QStyle>
#include <QUrl>

#include "playback_widget.h"

/* ========================================================================= */
// BUFFER HANDLER

buffer_handler::buffer_handler(int height, int width, QObject *parent) :
    QObject(parent)
{
    this->fr_height = height;
    this->fr_width = width;
    this->fr_size = fr_height * fr_width;
    this->fp = NULL;
    this->running = true;
    this->current_frame = 1;
}
buffer_handler::~buffer_handler()
{
    if (fp) {
        fclose(fp);
        free(frame);
    }
}

// public function(s)
bool buffer_handler::hasFP()
{
    /*! \brief Returns whether or not there is a loaded file pointer. */
    return fp != NULL ? true : false;
}

// public slots
void buffer_handler::loadFile(QString file_name)
{
    /*! \brief Prepares a file for read in the backend.
     * \param file_name The filename as received by the GUI in playback_widget or from the drag and drop.
     */
    /*! Step 1: Open the file specified in the parameter */
    fp = fopen(file_name.toStdString().c_str(), "rb");
    if (!fp) {
        emit loaded(NO_LOAD);
        return;
    }

    /* Step 2: Find the size of the raw file */
    fseek(fp, 0, SEEK_END);
    unsigned int filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    num_frames = filesize / (fr_size * pixel_size);
    if (!filesize) {
        emit loaded(NO_DATA);
        return;
    }

    /* Step 3: Allocate the memory for the frames */
    frame = (uint16_t*) malloc(pixel_size * fr_size);
    dark_data = (float*) calloc(fr_size, sizeof(float));

    /* Step 4: Populate data for the first frame */
    current_frame = 1;
    fseek(fp, (current_frame - 1) * fr_size * pixel_size, SEEK_SET);
    fread(frame, pixel_size, fr_size, fp);

    /* Step 5: Pass back the handling of the data to the playback_widget */
    emit loaded(SUCCESS);
}
void buffer_handler::loadDSF(QString file_name, unsigned int elements_to_read, long offset)
{
    /*! \brief Generates a dark mask the size of one frame from a range within a file.
     * \param file_name The file name as specified by the GUI in ControlsBox.
     * \param elements_to_read The number of words to read in the file.
     * \param offset The number of words to offset before beginning to read the file. */
    unsigned int ndx = 0;
    float frames_in_mask = 0;

    /* Step 0: Check that there is a file loaded first... */
    if (!frame) {
        emit loaded(NO_FILE);
        return;
    }

    /* Step 1: Allocate some memory and attempt to open the file */
    float *mask_in = (float*) calloc(fr_size, sizeof(float));
    FILE *mask_file;
    mask_file = fopen(file_name.toStdString().c_str(), "rb");

    /* Step 1.5: Check that the file is valid */
    if (mask_file) {
        /* Step 2: Offset the file pointer to the requested frame */
        fseek (mask_file, offset, SEEK_SET);

        /* Step 3: Read the data into a temporary buffer, then copy it into an unsigned int buffer which we can use to
         * process the whole file at the same time */
        uint16_t *temp_buffer = (uint16_t*) malloc(elements_to_read * sizeof(*temp_buffer));
        unsigned int *pic_buffer = (unsigned int*) malloc(elements_to_read * sizeof(*pic_buffer));
        fread(temp_buffer, sizeof(uint16_t), elements_to_read, mask_file);
        for ( ; ndx < elements_to_read; ndx++)
            pic_buffer[ndx] = (unsigned int)temp_buffer[ndx];
        free(temp_buffer);

        /* Step 4: For each frame, begin collecting the value at each position. */
        for (unsigned int *pic_in = pic_buffer; pic_in <= (pic_buffer + elements_to_read - fr_size); pic_in += fr_size) {
            for (ndx = 0; ndx <= fr_size; ndx++)
                mask_in[ndx] += (float)pic_in[ndx];
            frames_in_mask++;
        }

        /* Step 5: Average the collected values and close the file */
        for (ndx = 0; ndx < fr_size; ndx++)
            mask_in[ndx] /= frames_in_mask;
        free(pic_buffer);
    } else {
        emit loaded(NO_MASK);
        return;
    }

    /* Step 6: Load the mask, then free the memory we allocated */
    emit loadMask(mask_in);
    fclose (mask_file);

    /* Step 7: Call playback_widget to finish loading the file */
    emit loaded(SUCCESS);
}
void buffer_handler::getFrame()
{
    /*! \brief The main event loop of the buffer_handler class.
     * \paragraph
     *
     * The function runs in an infinite loop until a stop is requested by setting running to false. Every microsecond,
     * the values of the current_frame and old_frame are compared. If they are not equal, the current_frame is read from
     * the file. This operation is memory locked to avoid simultaneous read and write operations on the frame array.
     */
    while (running) {
        if (current_frame != old_frame && fp) {
            buf_access.lock();
                fseek(fp, (current_frame - 1) * fr_size * pixel_size, SEEK_SET);
                fread(frame, pixel_size, fr_size, fp);
            buf_access.unlock();

            old_frame = current_frame;
        } else {
            usleep(1);
        }
    }
    emit finished();
}
uint16_t* buffer_handler::tapPixelRemap()
{
    uint16_t pic_buffer[fr_size];
    unsigned int div;
    unsigned int mod;
    for(unsigned int i = 0; i < fr_size; i++)
        pic_buffer[i] = frame[i];
    for (int row = 0; row < fr_height; row++) {
        for (int col = 0; col < fr_width; col++) {
            div = col / num_taps;
            mod = col % 4;
            frame[div + TAP_WIDTH * mod + row * fr_width] = pic_buffer[col + row * fr_width];
        }
    }
    return frame;
}
void buffer_handler::stop()
{
    /*! \brief Ends the event loop. */
    running = false;
}
void buffer_handler::debug()
{
    std::cout << "Hello, World!" << std::endl;
}

/* ========================================================================= */
// PLAYBACK WIDGET

playback_widget::playback_widget(frameWorker *fw, QWidget *parent) :
    QWidget(parent)
{
    this->fw = fw;
    this->frHeight = fw->getDataHeight();
    this->frWidth = fw->getFrameWidth();

    frame_size = frHeight * frWidth;
    bh = new buffer_handler(frHeight, frWidth);
    buffer_thread = new QThread();
    bh->moveToThread(buffer_thread);

    setAcceptDrops(true);

    // making the controls
    openFileButton = new QPushButton(this);
    openFileButton->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
    forwardButton = new QPushButton(this);
    forwardButton->setIcon(style()->standardIcon(QStyle::SP_MediaSkipForward));
    forwardButton->setEnabled(false);
    backwardButton = new QPushButton(this);
    backwardButton->setIcon(style()->standardIcon(QStyle::SP_MediaSkipBackward));
    backwardButton->setEnabled(false);
    playPauseButton = new QPushButton(this);
    playIcon = style()->standardIcon(QStyle::SP_MediaPlay);
    pauseIcon = style()->standardIcon(QStyle::SP_MediaPause);
    playPauseButton->setIcon(playIcon);
    playPauseButton->setEnabled(false);
    progressBar = new QSlider(this);
    progressBar->setOrientation(Qt::Horizontal);
    progressBar->setEnabled(false);
    statusLabel = new QLabel(this);
    statusLabel->setText("Error: No file selected. Please open a .raw file or drop one in the window.");
    statusLabel->setStyleSheet("QLabel { color: red; }");

    // making the viewing window
    qcp = new QCustomPlot();
    qcp->setNotAntialiasedElement(QCP::aeAll);
    QSizePolicy qsp(QSizePolicy::Preferred, QSizePolicy::Preferred);
    qsp.setHeightForWidth(true);
    qcp->setSizePolicy(qsp);
    qcp->heightForWidth(200);
    qcp->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    qcp->axisRect()->setupFullAxesBox(true);
    qcp->xAxis->setLabel("x");
    qcp->yAxis->setLabel("y");

    colorMap = new QCPColorMap(qcp->xAxis, qcp->yAxis);
    colorMapData = NULL;
    qcp->addPlottable(colorMap);

    colorScale = new QCPColorScale(qcp);
    qcp->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight);

    colorMap->setColorScale(colorScale);
    colorMap->data()->setValueRange(QCPRange(frHeight, 0));
    colorMap->data()->setKeyRange(QCPRange(0, frWidth));
    colorMap->setDataRange(QCPRange(floor, ceiling));
    colorMap->setGradient(QCPColorGradient::gpJet);
    colorMap->setInterpolate(false);
    colorMap->setAntialiased(false);

    QCPMarginGroup *marginGroup = new QCPMarginGroup(qcp);
    qcp->axisRect()->setMarginGroup(QCP::msBottom | QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom | QCP::msTop, marginGroup);

    colorMapData = new QCPColorMapData(frWidth, frHeight, QCPRange(0, frWidth), QCPRange(0, frHeight));
    colorMap->setData(colorMapData);

    updateFloor(0);
    updateCeiling(fw->base_ceiling);
    qcp->rescaleAxes();
    qcp->axisRect()->setBackgroundScaled(false);

    dark = new dark_subtraction_filter(frWidth,frHeight);

    // connecting the buttons to slots
    connect(openFileButton, SIGNAL(clicked()), this, SLOT(loadFile()));
    connect(playPauseButton, SIGNAL(clicked()), this, SLOT(playPause()));
    connect(forwardButton, SIGNAL(clicked()), this, SLOT(moveForward()));
    connect(backwardButton, SIGNAL(clicked()), this, SLOT(moveBackward()));
    connect(progressBar, SIGNAL(valueChanged(int)), this, SLOT(handleFrame(int)));
    connect(qcp->yAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(colorMapScrolledY(QCPRange)));
    connect(qcp->xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(colorMapScrolledX(QCPRange)));
    connect(buffer_thread, SIGNAL(started()), bh, SLOT(getFrame()));
    connect(bh, SIGNAL(finished()), buffer_thread, SLOT(deleteLater()));
    connect(bh, SIGNAL(loaded(err_code)), this, SLOT(finishLoading(err_code)));
    connect(bh, SIGNAL(loadMask(float*)), this, SLOT(loadMaskIn(float*)));
    connect(this, SIGNAL(frameDone(int)), progressBar, SLOT(setValue(int)));
    connect(this, SIGNAL(frameDone(int)), this, SLOT(updateStatus(int)));

    // setting the layout
    qgl.addWidget(qcp, 0, 0, 7, 8);
    qgl.addWidget(progressBar, 7, 0, 1, 8);
    qgl.addWidget(openFileButton, 8, 0, 1, 1);
    qgl.addWidget(backwardButton, 8, 2, 1, 1);
    qgl.addWidget(playPauseButton, 8, 3, 1, 1);
    qgl.addWidget(forwardButton, 8, 4, 1, 1);
    qgl.addWidget(statusLabel, 8, 5, 1, 3);
    this->setLayout(&qgl);

    buffer_thread->start();
}
playback_widget::~playback_widget()
{
    bh->stop();
    usleep(1000);
    delete bh;
    delete dark;
    delete qcp;
}

// public functions
bool playback_widget::isPlaying()
{
    /*! \brief Returns the status of whether or not the video is playing. */
    return play;
}
double playback_widget::getCeiling()
{
    /*! \brief Return the value of the ceiling for this widget as a double */
    return ceiling;
}
double playback_widget::getFloor()
{
    /*! \brief Return the value of the floor for this widget as a double */
    return floor;
}

//public slots
void playback_widget::toggleUseDSF(bool t)
{
    /*! \brief Toggles the use of the dark mask for the playback widget only */
    useDSF = t;
    bh->old_frame = -1;
    handleFrame(bh->current_frame);
}
void playback_widget::loadDSF(QString f, unsigned int e, long o)
{
    /*! \brief A function that shoots through information from ControlsBox to buffer_handler */
    bh->loadDSF(f, e, o);
}
void playback_widget::stop()
{
    /*! \brief Stops playback and returnsto the first frame. */
    // ...taking advantage of our clunky corner case catching code

    bh->current_frame = 1;
    if(bh->hasFP())
    {
        play = false;
        playBackward = true;
        playPause();
    }
}
void playback_widget::colorMapScrolledX(const QCPRange &newRange)
{
    /*! \brief Controls the behavior of zooming the plot.
     * \param newRange Mouse wheel scrolled range.
     * Color Maps must not allow the user to zoom past the dimensions of the frame.
     */
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = frWidth;
    if (boundedRange.size() > upperRangeBound - lowerRangeBound) {
        boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    } else {
        double oldSize = boundedRange.size();
        if (boundedRange.lower < lowerRangeBound) {
            boundedRange.lower = lowerRangeBound;
            boundedRange.upper = lowerRangeBound + oldSize;
        }
        if (boundedRange.upper > upperRangeBound) {
            boundedRange.lower = upperRangeBound - oldSize;
            boundedRange.upper = upperRangeBound;
        }
    }
    qcp->xAxis->setRange(boundedRange);
}
void playback_widget::colorMapScrolledY(const QCPRange &newRange)
{
    /*! \brief Controls the behavior of zooming the plot.
     * \param newRange Mouse wheel scrolled range.
     * Color Maps must not allow the user to zoom past the dimensions of the frame.
     */
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = frHeight;
    if (boundedRange.size() > upperRangeBound - lowerRangeBound) {
        boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    } else {
        double oldSize = boundedRange.size();
        if (boundedRange.lower < lowerRangeBound) {
            boundedRange.lower = lowerRangeBound;
            boundedRange.upper = lowerRangeBound + oldSize;
        }
        if (boundedRange.upper > upperRangeBound) {
            boundedRange.lower = upperRangeBound - oldSize;
            boundedRange.upper = upperRangeBound;
        }
    }
    qcp->yAxis->setRange(boundedRange);
}
void playback_widget::updateCeiling(int c)
{
    /*! \brief Change the value of the ceiling for this widget to the input parameter and replot the color scale. */
    ceiling = (double)c;
    rescaleRange();
}
void playback_widget::updateFloor(int f)
{
    /*! \brief Change the value of the floor for this widget to the input parameter and replot the color scale. */
    floor = (double)f;
    rescaleRange();
}
inline void playback_widget::rescaleRange()
{
    /*! \brief Set the color scale of the display to the last used values for this widget */
    colorScale->setDataRange(QCPRange(floor, ceiling));
    qcp->replot(); // We need to replot in this version because there are discrete frames, so the floor and ceiling are not updated automatically.
}
void playback_widget::playPause()
{
    /*! \brief Toggles the playback of a video.
     * \paragraph
     *
     * Allows playback to proceed in the backwards or forward direction depending on the commands entered by the user. Stops
     * playback when pause is activated.
     */
    play = !play;
    if (play && playBackward) {
        // clunkcy corner-case catching code
        play = false;
        playBackward = false;
    }
    if (play || playBackward) {
        // We need to disconnect each button from any possible connected states before we can make new connections
        disconnect(forwardButton,  SIGNAL(clicked()), this, SLOT(moveForward()));
        disconnect(backwardButton, SIGNAL(clicked()), this, SLOT(moveBackward()));
        disconnect(forwardButton,  SIGNAL(clicked()), this, SLOT(fastForward()));
        disconnect(backwardButton, SIGNAL(clicked()), this, SLOT(fastRewind()));
        connect(forwardButton,  SIGNAL(clicked()), this, SLOT(fastForward()));
        connect(backwardButton, SIGNAL(clicked()), this, SLOT(fastRewind()));

        playPauseButton->setIcon(pauseIcon);

        if (play) {
            disconnect(&render_timer, SIGNAL(timeout()), this, SLOT(moveBackward()));
            connect(   &render_timer, SIGNAL(timeout()), this, SLOT(moveForward()));
        } else if (playBackward) {
            disconnect(&render_timer, SIGNAL(timeout()), this, SLOT(moveForward()));
            connect(   &render_timer, SIGNAL(timeout()), this, SLOT(moveBackward()));
        }
        render_timer.start(75);
    } else {
        render_timer.stop();
        interval = 1;
        bh->old_frame = -1;
        handleFrame(bh->current_frame);

        disconnect(forwardButton,  SIGNAL(clicked()), this, SLOT(fastForward()));
        disconnect(backwardButton, SIGNAL(clicked()), this, SLOT(fastRewind()));
        connect(forwardButton,  SIGNAL(clicked()), this, SLOT(moveForward()));
        connect(backwardButton, SIGNAL(clicked()), this, SLOT(moveBackward()));

        playPauseButton->setIcon(playIcon);
    }
}
void playback_widget::moveForward()
{
    /*! \brief Sequences the frames in ascending order. */
    bh->old_frame = -1;
    bh->current_frame += interval;
    if (bh->current_frame > bh->num_frames)
        bh->current_frame = 1 + (bh->current_frame - bh->num_frames - 1);
    handleFrame(bh->current_frame);
}
void playback_widget::moveBackward()
{
    /*! \brief Sequences the frames in descending order. */
    bh->old_frame = -1;
    bh->current_frame -= interval;
    if (bh->current_frame < 1)
        bh->current_frame = bh->num_frames + bh->current_frame;
    handleFrame(bh->current_frame);
}
void playback_widget::fastForward()
{
    /*! \brief Increases the frameskip by a factor of two for each call. Will reverse the play direction if necessary. */
    if (interval == 1)
        interval++;
    else if (interval <= 32 && play)
        interval *= 2;
    if (!play) {
        interval = 1;
        playBackward = false;
        playPause();
    }
}
void playback_widget::fastRewind()
{
    /*! Increases the frameskip in reverse by a factor of two for each call. Will reverse the play direction if
     * necessary. */
    if (interval == 1)
        interval++;
    else if (interval <= 32 && playBackward)
        interval *= 2;
    if (!playBackward) {
        if (!play) {
            play = true;
            playBackward = true;
            playPause();
        } else {
            interval = 1;
            playBackward = true;
            playPause();
        }
    }
}

// protected
void playback_widget::dragEnterEvent(QDragEnterEvent *event)
{
    if (event->mimeData()->hasUrls())
        event->acceptProposedAction();
}
void playback_widget::dropEvent(QDropEvent *event)
{
    foreach (const QUrl &url, event->mimeData()->urls()) {
        const QString &fileName = url.toLocalFile();
#ifdef VERBOSE
        qDebug() << "Dropped file:" << fileName;
#endif
            bh->loadFile(fileName);
    }
}

//private slots
void playback_widget::loadFile()
{
    /*! \brief Begins the process of loading in data.
     * \paragraph
     *
     * Opens a dialog to select a file, then passes it to the backend. The only check is to see if the file name is empty, which would
     * correspond to the cancel button being pressed or a similar case. We also want to make sure the video is not playing while this
     * dialog is open. */
    if (play || playBackward) {
        play = true; // ew
        playBackward = false;
        playPause();
    }
    statusLabel->setText("Error: No file selected. Please open a .raw file or drop one in the window.");
    QString fname = QFileDialog::getOpenFileName(this, tr("Please Select a Raw File"), tr("/"), tr("Raw (*.raw *.bin *.hsi *.img)"));
    if (fname.isEmpty()) {
        updateStatus(bh->current_frame);
        return;
    }
    bh->loadFile(fname);
}
void playback_widget::finishLoading(err_code e)
{
    /*! Checks the result of the error_t enum and returns an appropriate error message. Otherwise, queues the controls and renders the
     * first frame. */
    switch (e) {
    case NO_LOAD:
        statusLabel->setText("Error: Selected file could not be opened.");
        break;
    case NO_DATA:
        statusLabel->setText("Error: The selected file contains no data.");
        break;
    case NO_FILE:
        statusLabel->setText("Error: Please load a data file before selecting dark frames.");
        break;
    case READ_FAIL:
        statusLabel->setText("Error: Mask file read failed.");
        break;
    case NO_MASK:
        statusLabel->setText("Error: Could not load Dark Mask");
        break;
    default: /* SUCCESS */
        // Queue up the controls
        playPauseButton->setEnabled(true);
        forwardButton->setEnabled(true);
        backwardButton->setEnabled(true);
        progressBar->setEnabled(true);
        progressBar->setMinimum(1);
        progressBar->setMaximum(bh->num_frames);
        // Process the newly loaded frame
        bh->old_frame = -1; // shhhhhh... don't tell anyone how janky this is ;)
        handleFrame(bh->current_frame);
        break;
    }
}
void playback_widget::loadMaskIn(float *mask_arr)
{
    /*! \brief Accepts a mask array and loads it into the dark subtraction filter. */
    dark->load_mask(mask_arr);
    free(mask_arr);
}
void playback_widget::updateStatus(int frameNum)
{
    /*! \brief Updates the statusLabel to the current frame number, and displays the total number frames. */
    if (bh->frame) {
        if ( interval == 1 )
            statusLabel->setText(tr("Frame: %1 / %2").arg(frameNum).arg(bh->num_frames));
        else if( interval > 1 )
            statusLabel->setText(tr("Frame: %1 / %2  (x%3)").arg(frameNum).arg(bh->num_frames).arg(interval));
    } else {
        statusLabel->setText("Error: No file selected. Please open a .raw file or drop one in the window.");
    }
}

void playback_widget::handleFrame(int frameNum)
{
    /*! \brief Renders a frame at a position within a file.
     * \param frameNum Requested frame to render.
     * This function updates the value of the current_frame, which will trigger the buffer_handler getFrame() process.
     * The process then waits 50 microseconds to ensure that the mutex lock at the backend has triggered. The rendering of
     * the data will then be serialized to occur after the frame has been read into the array. The y-axis is reversed. */
    bh->current_frame = frameNum;
    usleep(50);
    if(bh->current_frame == bh->old_frame)
        return;

    bh->buf_access.lock();
        dark->update_dark_subtraction(bh->frame, bh->dark_data);

        for (int col = 0; col < frWidth; col++)
        {
            for(int row = 0; row < frHeight; row++) {
                if (useDSF)
                    colorMap->data()->setCell(col, row, \
                                            bh->dark_data[(frHeight-row-1) * frWidth + col]);
                else
                    colorMap->data()->setCell(col,row, \
                                            bh->frame[(frHeight - row - 1) * frWidth + col]);
            }
        }
    bh->buf_access.unlock();

    qcp->replot();
    emit frameDone(bh->current_frame);
}
