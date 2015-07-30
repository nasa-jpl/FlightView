#include <QFileDialog>
#include <QStyle>
#include <QUrl>

#include "playback_widget.h"

/* ========================================================================= */
// BUFFER HANDLER

buffer_handler::buffer_handler(int height, int width, QObject* parent) :
    QObject(parent)
{
    this->fr_height = height;
    this->fr_width = width;
    this->fr_size = fr_height*fr_width;
    this->fp = NULL;
    this->running = true;
}
buffer_handler::~buffer_handler()
{
    if(fp)
    {
        fclose(fp);
        free(frame);
    }
}
void buffer_handler::loadFile(QString file_name)
{
    // Step 1: Open the file specified in the parameter
    fp = fopen(file_name.toStdString().c_str(), "rb");
    if(!fp)
    {
        emit loaded(NO_LOAD);
        return;
    }

    //Step 2: Find the size of the raw file
    fseek(fp, 0, SEEK_END);
    unsigned int filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    num_frames = filesize / (fr_size*pixel_size);
    if(!filesize)
    {
        emit loaded(NO_DATA);
        return;
    }

    frame = (uint16_t*) malloc(pixel_size*fr_size);
    dark_data = (float*) calloc(fr_size,sizeof(float));
    current_frame = 1;
    fseek(fp,(current_frame-1)*fr_size*pixel_size,SEEK_SET);
    fread(frame,pixel_size,fr_size,fp);

    emit loaded(SUCCESS);
}
void buffer_handler::loadDSF(QString file_name, unsigned int elements_to_read, long offset)
{
    unsigned int ndx = 0;
    float frames_in_mask = 0;

    // Step 0: Check that there is a file loaded first...
    if(!frame)
    {
        emit loaded(NO_FILE);
        return;
    }

    // Step 1: Allocate some memory and attempt to open the file
    float* mask_in = (float*) calloc(fr_size, sizeof(float));
    FILE* mask_file;
    mask_file = fopen(file_name.toStdString().c_str(), "rb");

    // Step 1.5: Check that the file is valid
    if(mask_file)
    {
        // Step 2: Offset the file pointer to the requested frame
        if(fseek(mask_file, offset, SEEK_SET) >= 0)
        {
            // Step 3: Read the data into a temporary buffer, then copy it into an unsigned int buffer which we can use to
            // process the whole file at the same time
            uint16_t* temp_buffer = (uint16_t*) malloc(elements_to_read*sizeof(uint16_t));
            unsigned int* pic_buffer = (unsigned int*) malloc(elements_to_read*sizeof(unsigned int));
            fread(temp_buffer, sizeof(uint16_t), elements_to_read, mask_file);
            for( ; ndx < elements_to_read; ndx++) { pic_buffer[ndx] = (unsigned int)temp_buffer[ndx]; }
            free(temp_buffer);

            // Step 4: For each frame, begin collecting the value at each position.
            for(unsigned int* pic_in = pic_buffer; pic_in <= (pic_buffer + elements_to_read - fr_size); pic_in += fr_size)
            {
                for(ndx = 0; ndx <= fr_size; ndx++)
                    mask_in[ndx] += (float)pic_in[ndx];
                frames_in_mask++;
            }

            // Step 5: Average the collected values and close the file
            for(ndx = 0; ndx < fr_size; ndx++) { mask_in[ndx] /= frames_in_mask; }
            free(pic_buffer);
        }
        else
        {
            emit loaded(READ_FAIL);
            return;
        }
    }
    else
    {
        emit loaded(NO_MASK);
        return;
    }

    // Step 6: Load the mask, then free the memory we allocated
    emit loadMask(mask_in);
    fclose (mask_file);

    emit loaded(SUCCESS);
}
void buffer_handler::getFrame()
{
    current_frame = 1;
    while(running)
    {
        if(current_frame != old_frame)
        {
            fseek(fp,(current_frame-1)*fr_size*pixel_size,SEEK_SET);

            buf_access.lock();
                fread(frame,pixel_size,fr_size,fp);
            buf_access.unlock();

            old_frame = current_frame;
        }
        else
        {
            usleep(5);
        }
    }
    emit finished();
}
void buffer_handler::stop()
{
    running = false;
}
void buffer_handler::debug()
{
    std::cout << "Hello, World!" << std::endl;
}

/* ========================================================================= */
// PLAYBACK WIDGET

playback_widget::playback_widget(frameWorker* fw, QWidget* parent) :
    QWidget(parent)
{
    this->fw = fw;
    this->frHeight = fw->getDataHeight();
    this->frWidth = fw->getFrameWidth();

    frame_size = frHeight*frWidth;
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
    QSizePolicy qsp(QSizePolicy::Preferred,QSizePolicy::Preferred);
    qsp.setHeightForWidth(true);
    qcp->setSizePolicy(qsp);
    qcp->heightForWidth(200);
    qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);
    qcp->axisRect()->setupFullAxesBox(true);
    qcp->xAxis->setLabel("x");
    qcp->yAxis->setLabel("y");

    colorMap = new QCPColorMap(qcp->xAxis,qcp->yAxis);
    colorMapData = NULL;
    qcp->addPlottable(colorMap);

    colorScale = new QCPColorScale(qcp);
    qcp->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight);

    colorMap->setColorScale(colorScale);
    colorMap->data()->setValueRange(QCPRange(frHeight,0));
    colorMap->data()->setKeyRange(QCPRange(0,frWidth));
    colorMap->setDataRange(QCPRange(floor,ceiling));
    colorMap->setGradient(QCPColorGradient::gpJet);
    colorMap->setInterpolate(false);
    colorMap->setAntialiased(false);

    QCPMarginGroup *marginGroup = new QCPMarginGroup(qcp);
    qcp->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);

    colorMapData = new QCPColorMapData(frWidth,frHeight,QCPRange(0,frWidth),QCPRange(0,frHeight));
    colorMap->setData(colorMapData);

    updateFloor(0);
    updateCeiling(fw->base_ceiling);
    qcp->rescaleAxes();
    qcp->axisRect()->setBackgroundScaled(false);

    dark = new dark_subtraction_filter(frWidth,frHeight);

    // connecting the buttons to slots
    connect(openFileButton,SIGNAL(clicked()),this,SLOT(loadFile()));
    connect(playPauseButton,SIGNAL(clicked()),this,SLOT(playPause()));
    connect(forwardButton,SIGNAL(clicked()),this,SLOT(moveForward()));
    connect(backwardButton,SIGNAL(clicked()),this,SLOT(moveBackward()));
    connect(progressBar,SIGNAL(valueChanged(int)),this,SLOT(handleFrame(int)));
    connect(qcp->yAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledY(QCPRange)));
    connect(qcp->xAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledX(QCPRange)));
    connect(buffer_thread,SIGNAL(started()),bh,SLOT(getFrame()));
    connect(bh,SIGNAL(finished()),buffer_thread,SLOT(deleteLater()));
    connect(bh,SIGNAL(loaded(err_code)),this,SLOT(finishLoading(err_code)));
    connect(bh,SIGNAL(loadMask(float*)),this,SLOT(loadMaskIn(float*)));
    connect(this,SIGNAL(frameDone(int)),progressBar,SLOT(setValue(int)));
    connect(this,SIGNAL(frameDone(int)),this,SLOT(updateStatus(int)));

    // setting the layout
    qgl.addWidget(qcp,0,0,7,8);
    qgl.addWidget(progressBar,7,0,1,8);
    qgl.addWidget(openFileButton,8,0,1,1);
    qgl.addWidget(backwardButton,8,2,1,1);
    qgl.addWidget(playPauseButton,8,3,1,1);
    qgl.addWidget(forwardButton,8,4,1,1);
    qgl.addWidget(statusLabel,8,5,1,3);
    this->setLayout(&qgl);

    buffer_thread->start();
}
playback_widget::~playback_widget()
{
    bh->stop();
    delete bh;
    buffer_thread->exit(0);
    buffer_thread->wait();
    delete buffer_thread;
    delete dark;
    delete colorScale;
    delete colorMap;
    delete qcp;
}

// public functions
bool playback_widget::isPlaying()
{
    return play;
}
double playback_widget::getCeiling()
{
    return ceiling;
}
double playback_widget::getFloor()
{
    return floor;
}

//public slots
void playback_widget::toggleUseDSF(bool t)
{
    useDSF = t;
    bh->old_frame = -1;
    handleFrame(bh->current_frame);
}
void playback_widget::loadDSF(QString f, unsigned int e, long o)
{
    bh->loadDSF(f, e, o);
}
void playback_widget::stop()
{
    // taking advantage of our clunky corner case catching code
    play = false;
    playBackward = true;
    bh->current_frame = 1;
    playPause();
}
void playback_widget::colorMapScrolledX(const QCPRange &newRange)
{
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = frWidth;
    if (boundedRange.size() > upperRangeBound-lowerRangeBound)
    {
        boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    }
    else
    {
        double oldSize = boundedRange.size();
        if (boundedRange.lower < lowerRangeBound)
        {
            boundedRange.lower = lowerRangeBound;
            boundedRange.upper = lowerRangeBound+oldSize;
        }
        if (boundedRange.upper > upperRangeBound)
        {
            boundedRange.lower = upperRangeBound-oldSize;
            boundedRange.upper = upperRangeBound;
        }
    }
    qcp->xAxis->setRange(boundedRange);
}
void playback_widget::colorMapScrolledY(const QCPRange &newRange)
{
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = frHeight;
    if (boundedRange.size() > upperRangeBound-lowerRangeBound)
    {
        boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    }
    else
    {
        double oldSize = boundedRange.size();
        if (boundedRange.lower < lowerRangeBound)
        {
            boundedRange.lower = lowerRangeBound;
            boundedRange.upper = lowerRangeBound+oldSize;
        }
        if (boundedRange.upper > upperRangeBound)
        {
            boundedRange.lower = upperRangeBound-oldSize;
            boundedRange.upper = upperRangeBound;
        }
    }
    qcp->yAxis->setRange(boundedRange);
}
void playback_widget::updateCeiling(int c)
{
    ceiling = (double)c;
    colorScale->setDataRange(QCPRange(floor,ceiling));
    qcp->replot();
}
void playback_widget::updateFloor(int f)
{
    floor = (double)f;
    colorScale->setDataRange(QCPRange(floor,ceiling));
    qcp->replot();
}
void playback_widget::rescaleRange()
{
    colorScale->setDataRange(QCPRange(floor,ceiling));
    qcp->replot();
}
void playback_widget::playPause()
{
    play = !play;
    if(play && playBackward) // clunky corner case catching code
    {
        play = false;
        playBackward = false;
    }
    if(play || playBackward)
    {
        // We need to disconnect each button from any possible connected states before we can make new connections
        disconnect(forwardButton,SIGNAL(clicked()),this,SLOT(moveForward()));
        disconnect(backwardButton,SIGNAL(clicked()),this,SLOT(moveBackward()));
        disconnect(forwardButton,SIGNAL(clicked()),this,SLOT(fastForward()));
        disconnect(backwardButton,SIGNAL(clicked()),this,SLOT(fastRewind()));
        connect(forwardButton,SIGNAL(clicked()),this,SLOT(fastForward()));
        connect(backwardButton,SIGNAL(clicked()),this,SLOT(fastRewind()));

        playPauseButton->setIcon(pauseIcon);

        if(play)
        {
            disconnect(&render_timer,SIGNAL(timeout()),this,SLOT(moveBackward()));
            connect(&render_timer,SIGNAL(timeout()),this,SLOT(moveForward()));
        }
        else if(playBackward)
        {
            disconnect(&render_timer,SIGNAL(timeout()),this,SLOT(moveForward()));
            connect(&render_timer,SIGNAL(timeout()),this,SLOT(moveBackward()));
        }
        render_timer.start(50);
    }
    else
    {
        render_timer.stop();
        interval = 1;
        bh->old_frame = -1;
        handleFrame(bh->current_frame);

        disconnect(forwardButton,SIGNAL(clicked()),this,SLOT(fastForward()));
        disconnect(backwardButton,SIGNAL(clicked()),this,SLOT(fastRewind()));
        connect(forwardButton,SIGNAL(clicked()),this,SLOT(moveForward()));
        connect(backwardButton,SIGNAL(clicked()),this,SLOT(moveBackward()));

        playPauseButton->setIcon(playIcon);
    }
}
void playback_widget::moveForward()
{
    bh->current_frame += interval;
    if(bh->current_frame > bh->num_frames)
    {
        bh->current_frame = 1 + (bh->current_frame-bh->num_frames-1);
    }
    handleFrame(bh->current_frame);
}
void playback_widget::moveBackward()
{
    bh->current_frame -= interval;
    if(bh->current_frame < 1)
    {
        bh->current_frame = bh->num_frames + bh->current_frame;
    }
    handleFrame(bh->current_frame);
}
void playback_widget::fastForward()
{
    if( interval == 1 )
        interval++;
    else if(interval <= 32 && play)
        interval *= 2;
    if(!play)
    {
        interval = 1;
        playBackward = false;
        playPause();
    }
}
void playback_widget::fastRewind()
{
    if(interval == 1)
        interval++;
    else if(interval <= 32 && playBackward)
        interval *= 2;
    if(!playBackward)
    {
        if(!play)
        {
            play = true;
            playBackward = true;
            playPause();
        }
        else
        {
            interval = 1;
            playBackward = true;
            playPause();
        }
    }
}

// protected
void playback_widget::dragEnterEvent(QDragEnterEvent* event)
{
    if(event->mimeData()->hasUrls())
        event->acceptProposedAction();
}
void playback_widget::dropEvent(QDropEvent* event)
{
    foreach (const QUrl &url, event->mimeData()->urls())
    {
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
    if(this->isPlaying() || playBackward == true)
    {
        play = true;
        playBackward = false;
        playPause();
    }
    statusLabel->setText("Error: No file selected. Please open a .raw file or drop one in the window.");

    QString fname = QFileDialog::getOpenFileName(this, "Please Select a Raw File", "/home/jryan/NGIS_DATA/jryan/",tr("Raw (*.raw *.bin *.hsi *.img)"));
    if(fname.isEmpty()) // if the cancel button is pressed
    {
        updateStatus(bh->current_frame);
        return;
    }
    bh->loadFile(fname);
}
void playback_widget::finishLoading(err_code e)
{
    switch(e)
    {
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
void playback_widget::loadMaskIn(float* mask_arr)
{
    dark->load_mask(mask_arr);
    free(mask_arr);
}
void playback_widget::updateStatus(int frameNum)
{
    if(bh->frame)
    {
        if( interval == 1 )
            statusLabel->setText(tr("Frame: %1 / %2").arg(frameNum).arg(bh->num_frames));
        else if( interval > 1 )
            statusLabel->setText(tr("Frame: %1 / %2  (x%3)").arg(frameNum).arg(bh->num_frames).arg(interval));
    }
    else
    {
        statusLabel->setText("Error: No file selected. Please open a .raw file or drop one in the window.");
    }
}
void playback_widget::handleFrame(int frameNum)
{
    bh->current_frame = frameNum;
    if(bh->current_frame == bh->old_frame)
        return;
    usleep(5);
    bh->buf_access.lock();
    dark->update_dark_subtraction(bh->frame, bh->dark_data);
    for(int col = 0; col < frWidth; col++)
    {
        for(int row = 0; row < frHeight; row++)
        {
            if(useDSF)
            {
                colorMap->data()->setCell(col,row, \
                                          bh->dark_data[(frHeight-row-1)*frWidth + col]);
            }
            else
            {
                colorMap->data()->setCell(col,row, \
                                          bh->frame[(frHeight-row-1)*frWidth + col]);
            }
        }
    }
    bh->buf_access.unlock();
    qcp->replot();
    emit frameDone(bh->current_frame);
}
