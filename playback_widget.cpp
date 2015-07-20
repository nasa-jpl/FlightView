#include <QFileDialog>
#include <QStyle>

#include "playback_widget.h"

playback_widget::playback_widget(frameWorker* fw, QWidget* parent) :
    QWidget(parent)
{
    this->fw = fw;
    current_frame = 0;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    frame_size = frHeight*frWidth;

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

    dsf = new dark_subtraction_filter(frWidth,frHeight);

    // connecting the buttons to slots
    connect(openFileButton,SIGNAL(clicked()),this,SLOT(loadFile()));
    connect(playPauseButton,SIGNAL(clicked()),this,SLOT(playPause()));
    connect(forwardButton,SIGNAL(clicked()),this,SLOT(moveForward()));
    connect(backwardButton,SIGNAL(clicked()),this,SLOT(moveBackward()));
    //connect(progressBar,SIGNAL(valueChanged(int)),this,SLOT(handleFrame(int)));
    connect(qcp->yAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledY(QCPRange)));
    connect(qcp->xAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledX(QCPRange)));
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
}
playback_widget::~playback_widget()
{
    delete dsf;
    free(input_array);
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
}
void playback_widget::loadDSF(QString file_name, unsigned int bytes_to_read)
{
    float* mask_in = new float[frWidth*frHeight];
    FILE* mask_file;
    unsigned long mask_size = 0;
    mask_file = fopen(file_name.toStdString().c_str(), "rb");
    if(mask_file)
    {
        fseek (mask_file, 0, SEEK_END); // non-portable
        mask_size = ftell(mask_file);
        if(mask_size != (frWidth*frHeight*sizeof(float)))
        {
            std::cerr << "Error: Dark Mask File does not match image size" << std::endl;
            fclose (mask_file);
            return;
        }
        rewind(mask_file); // go back to beginning
        fread(mask_in, sizeof(float), frWidth*frHeight, mask_file);
        fclose (mask_file);
    }
    else
    {
        std::cerr << "Error: could not Dark Mask" << std::endl;
        return;
    }
    dsf->load_mask(mask_in);
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

// protected
void playback_widget::keyPressEvent(QKeyEvent* c)
{
    if(!c->modifiers())
    {
        if(c->key() == Qt::Key_Space)
        {
            this->playPause();
            c->accept();
            return;
        }
        if(c->key() == Qt::Key_A)
        {
            this->moveBackward();
            c->accept();
            return;
        }
        if(c->key() == Qt::Key_D)
        {
            this->moveForward();
            c->accept();
            return;
        }
        // More key mappings can be provided here
    }
}

//private slots
void playback_widget::loadFile()
{
    // remove reference to file in current memory, if there is one...
    if(input_array)
        free(input_array);

    QString fname = QFileDialog::getOpenFileName(this, "Please Select a Raw File", "/home/jryan/NGIS_DATA/jryan/",tr("Raw (*.raw *.bin *.hsi *.img)"));
    if(fname == NULL) // if the cancel button is pressed
        return;
    FILE* fp = fopen(fname.toStdString().c_str(), "rb");

    // find the size of the raw file
    fseek(fp, 0, SEEK_END);
    unsigned int filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    nFrames = filesize / (frame_size*pixel_size);

    // allocate memory for the frames
    frames = (uint16_t*) malloc(filesize);
    input_array = (unsigned int*) malloc(sizeof(double)*nFrames*frame_size);

    // load the bytes from the file into the frames array
    fread(frames,pixel_size,filesize/pixel_size,fp);
    fclose(fp);

    for(unsigned int ndx = 0; ndx < frame_size*nFrames; ndx++)
    {
        input_array[ndx] = (unsigned int)frames[ndx];
    }
    free(frames);

    playPauseButton->setEnabled(true);
    forwardButton->setEnabled(true);
    backwardButton->setEnabled(true);
    progressBar->setEnabled(true);
    progressBar->setMinimum(++current_frame);
    progressBar->setMaximum(nFrames);

    handleFrame(current_frame);
}
void playback_widget::updateStatus(int frameNum)
{
    if( interval == 1 )
    {
        statusLabel->setText(tr("Frame: %1 / %2").arg(frameNum).arg(nFrames));
    }
    else if( interval > 1 )
    {
        statusLabel->setText(tr("Frame: %1 / %2  (x%3)").arg(frameNum).arg(nFrames).arg(interval));
    }
}
void playback_widget::handleFrame(int frameNum)
{
    current_frame = frameNum;
    for(int col = 0; col < frWidth; col++)
    {
        for(int row = 0; row < frHeight; row++)
        {
            colorMap->data()->setCell(col,row,\
                             input_array[(current_frame-1)*frame_size + (frHeight-row-1)*frWidth + col]);
        }
    }
    qcp->replot();
    emit frameDone(current_frame);
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
        disconnect(forwardButton,SIGNAL(clicked()),this,SLOT(moveForward()));
        disconnect(backwardButton,SIGNAL(clicked()),this,SLOT(moveBackward()));
        disconnect(forwardButton,SIGNAL(clicked()),this,SLOT(fastForward()));
        disconnect(backwardButton,SIGNAL(clicked()),this,SLOT(fastRewind()));
        connect(forwardButton,SIGNAL(clicked()),this,SLOT(fastForward()));
        connect(backwardButton,SIGNAL(clicked()),this,SLOT(fastRewind()));

        playPauseButton->setIcon(pauseIcon);

        if(play)
        {
            disconnect(&renderTimer,SIGNAL(timeout()),this,SLOT(moveBackward()));
            connect(&renderTimer,SIGNAL(timeout()),this,SLOT(moveForward()));
        }
        else if(playBackward)
        {
            disconnect(&renderTimer,SIGNAL(timeout()),this,SLOT(moveForward()));
            connect(&renderTimer,SIGNAL(timeout()),this,SLOT(moveBackward()));
        }
        renderTimer.start(75);
    }
    else
    {
        renderTimer.stop();
        interval = 1;
        updateStatus(current_frame);

        disconnect(forwardButton,SIGNAL(clicked()),this,SLOT(fastForward()));
        disconnect(backwardButton,SIGNAL(clicked()),this,SLOT(fastRewind()));
        connect(forwardButton,SIGNAL(clicked()),this,SLOT(moveForward()));
        connect(backwardButton,SIGNAL(clicked()),this,SLOT(moveBackward()));

        playPauseButton->setIcon(playIcon);
    }
}
void playback_widget::moveForward()
{
    current_frame += interval;
    if(current_frame > nFrames)
    {
        current_frame = 1;
    }
    handleFrame(current_frame);
}
void playback_widget::moveBackward()
{
    current_frame -= interval;
    if(current_frame <= 1)
    {
        current_frame = nFrames - 1;
    }
    handleFrame(current_frame);
}
void playback_widget::fastForward()
{
    if( interval == 1 )
        interval++;
    else if(interval <= 32 && play)
        interval *= 2;
    if(!play)
    {
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
        playBackward = true;
        playPause();
    }
}
