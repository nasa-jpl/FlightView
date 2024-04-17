#include "waterfall.h"

// This is the RGB waterfall widget used in the flight screen.
// The "waterfall" tab is handled by a special instance of the Frameview Widget.

waterfall::waterfall(QWidget *parent) : QWidget(parent) {
    // Widget constructor for inclusion in main window via Qt Designer.
    // basically don't do much yet.

}

void waterfall::setup(frameWorker *fw, int vSize, int hSize, bool isSecondary, startupOptionsType options) {
    this->fw = fw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    this->options = options;
    this->isSecondary = isSecondary;
    if(isSecondary) {
        recordToJPG = false;
    } else {
        recordToJPG = options.wfPreviewContinuousMode;
    }
    // if not continuous mode, then if previewEnabled,
    // the flight widget will call the waterfall
    // to enable previews when recording.

    //rgbLineStruct blank = allocateLine();
    maxWFlength = 1024;
    wflength = maxWFlength;
    //wf.resize(maxWFlength, blank);
    allocateBlankWF();

    ceiling = 16000;
    floor = 0;

    r_row = 200;
    g_row = 250;
    b_row = 300;

    // Drawing:
    // Pixel format is AA RR GG BB, where AA = 8-bit Alpha value
    // AA = 0x00 = fully transparent
    // AA = 0xff = fully opaque

    vEdge = 0;
    hEdge = 0;
    this->vSize = vSize;
    this->hSize = hSize;
    // Override for now:
    this->vSize = maxWFlength;
    this->hSize = frWidth;
    opacity = 0xff;
    useDSF = false; // default to false since the program can't start up with a DSF mask anyway

    specImage = QImage(this->hSize, this->vSize, QImage::Format_ARGB32);
    statusMessage(QString("Created specImage with height %1 and width %2.").arg(specImage.height()).arg(specImage.width()));

    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
    rendertimer.setInterval(FRAME_DISPLAY_PERIOD_MSECS);

    if(options.headless && (!options.wfPreviewEnabled)) {
        statusMessage("Not starting waterfall display update timer for headless mode without waterfall previews.");
    } else {
        statusMessage("Starting waterfall");
        rendertimer.start();
    }
    if(!isSecondary) {
        if(options.wfPreviewEnabled || options.wfPreviewContinuousMode) {
            statusMessage("Waterfall preview ENABLED.");
            prepareWfImage();
            if(options.headless) {
                this->useDSF = true; // start with this ON since it will never get toggled
            }
        }
    }
    QSizePolicy policy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    this->setSizePolicy(policy);
    statusMessage("Finished waterfall constructor.");
}

waterfall::waterfall(frameWorker *fw, int vSize, int hSize, startupOptionsType options, QWidget *parent) : QWidget(parent)
{
    this->fw = fw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    this->options = options;
    recordToJPG = options.wfPreviewContinuousMode;
    // if not continuous mode, then if previewEnabled,
    // the flight widget will call the waterfall
    // to enable previews when recording.

    //rgbLineStruct blank = allocateLine();
    maxWFlength = 1024;
    wflength = maxWFlength;
    //wf.resize(maxWFlength, blank);
    allocateBlankWF();

    ceiling = 16000;
    floor = 0;

    r_row = 200;
    g_row = 250;
    b_row = 300;

    // Drawing:
    // Pixel format is AA RR GG BB, where AA = 8-bit Alpha value
    // AA = 0x00 = fully transparent
    // AA = 0xff = fully opaque

    vEdge = 0;
    hEdge = 0;
    this->vSize = vSize;
    this->hSize = hSize;
    // Override for now:
    this->vSize = maxWFlength;
    this->hSize = frWidth;
    opacity = 0xff;
    useDSF = false; // default to false since the program can't start up with a DSF mask anyway

    specImage = QImage(this->hSize, this->vSize, QImage::Format_ARGB32);
    statusMessage(QString("Created specImage with height %1 and width %2.").arg(specImage.height()).arg(specImage.width()));

    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
    rendertimer.setInterval(FRAME_DISPLAY_PERIOD_MSECS);

    if(options.headless && (!options.wfPreviewEnabled)) {
        statusMessage("Not starting waterfall display update timer for headless mode with preview disabled.");
    } else {
        statusMessage("Starting waterfall");
        rendertimer.start();
    }
    if(options.wfPreviewEnabled || options.wfPreviewContinuousMode) {
        statusMessage("Waterfall preview ENABLED.");
        prepareWfImage();
        if(options.headless) {
            this->useDSF = true; // start with this ON since it will never get toggled
        }
    }
    QSizePolicy policy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    this->setSizePolicy(policy);
    statusMessage("Finished waterfall constructor.");
}

void waterfall::prepareWfImage() {
    statusMessage("Preparing waterfall data storage.");
    bool borrowedFilePath = false;
    // Examine WF path
    if(!options.wfPreviewlocationset) {
        if(options.dataLocationSet) {
            options.wfPreviewLocation = options.dataLocation;
            statusMessage("Saving waterfall image to datastoragelocation");
            borrowedFilePath = true;
        } else {
            saveImageReady = false;
            statusMessage("Waterfall image location and datastoragelocation both blank. Not saving waterfall previews.");
            return;
        }
    }
    options.wfPreviewlocationset = true;


    if(!options.wfPreviewLocation.endsWith("/"))
        options.wfPreviewLocation.append("/");
    // Add "day" stamp to name
    QDateTime t = QDateTime::currentDateTime();
    QString dayStr = t.toUTC().toString("yyyyMMdd/");

    if(!borrowedFilePath) {
        statusMessage("Appending day to waterfall preview location");
        options.wfPreviewLocation.append(dayStr);
    }

    options.wfPreviewLocation.append("wf/");

    // mkdir
    QString command = "mkdir -p " + options.wfPreviewLocation;
    int sys_rtn = system(command.toLocal8Bit());
    if(sys_rtn) {
        statusMessage("Error, could not make waterfall preview location directory.");
        saveImageReady = false;
        return;
    }

    // if successful, mark ready:
    statusMessage(QString("Saving waterfall preview images to %1").arg(options.wfPreviewLocation));
    saveImageReady = true;
}

void waterfall::process()
{
    statusMessage("Thread started");
}

void waterfall::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);

    // anti-alias:
    painter.setRenderHint(QPainter::SmoothPixmapTransform);

    painter.setWindow(QRect(0, 0, hSize/4.0, 1024));

    // Target: Where the image ultimately goes
    // Source: A rectangle which defines the portion of the specImage
    //         to copy into the target.

    // The source is stretched out over the target.

    // For length trimming, adjust the wflength parameter on the source
    // for Left-Right trimming, adjust the first two parameters of source.

    QRectF target(0, 0, hSize/4.0, vSize);
    QRectF source(0.0f, 0.0f, hSize, wflength); // use source geometry to "crop" the waterfall image
    painter.drawImage(target, specImage, source);
}

void waterfall::redraw()
{
    QColor c;
    for(int y = 0; y < vSize; y++)
    {
        for(int x = 0; x < hSize; x++)
        {
            c.setAlpha(opacity);
            c.setRed(wf.at(y)->getRed()[x]);
            c.setGreen(wf.at(y)->getGreen()[x]);
            c.setBlue(wf.at(y)->getBlue()[x]);

            specImage.setPixel(x, y, c.rgba());
        }
    }
    this->repaint();
}

void waterfall::allocateBlankWF()
{
    for(int n=0; n < maxWFlength; n++)
    {
        rgbLine* line = new rgbLine(frWidth, false);
        wf.push_back(std::shared_ptr<rgbLine>(line));
    }
}

void waterfall::copyPixToLine(float *image, float *dst, int rowSelection)
{
    for(int p=0; p < frWidth; p++)
    {
        dst[p] = image[ rowSelection + p];
    }
}

void waterfall::copyPixToLine(uint16_t *image, float *dst, int rowSelection)
{
    for(int p=0; p < frWidth; p++)
    {
        dst[p] = (float)image[ rowSelection + p];
    }
}

void waterfall::addNewFrame()
{
    // MUTEX wait-lock
    addingFrame.lock();
    float *local_image_ptr;
    uint16_t* local_image_ptr_uint16;
    if(fw->curFrame == NULL)
    {
        addingFrame.unlock();
        return;
    }

    local_image_ptr = fw->curFrame->dark_subtracted_data;
    local_image_ptr_uint16 = fw->curFrame->image_data_ptr;

    rgbLine *line = new rgbLine(frWidth);

    // Copy portions of the frame into the line

    int r_row_pix = frWidth * r_row; // width times how many rows
    int g_row_pix = frWidth * g_row;
    int b_row_pix = frWidth * b_row;

    //    if(fw->dsfMaskCollected() && useDSF); // prior method
    if(useDSF) // concurrent
    {
        copyPixToLine(local_image_ptr, line->getr_raw(), r_row_pix);
        copyPixToLine(local_image_ptr, line->getg_raw(), g_row_pix);
        copyPixToLine(local_image_ptr, line->getb_raw(), b_row_pix);
    } else {
        copyPixToLine(local_image_ptr_uint16, line->getr_raw(), r_row_pix);
        copyPixToLine(local_image_ptr_uint16, line->getg_raw(), g_row_pix);
        copyPixToLine(local_image_ptr_uint16, line->getb_raw(), b_row_pix);
    }

    // process initial RGB values:
    processLineToRGB_MP(line);

    QMutexLocker lockwf(&wfInUse);

    wf.push_front(std::shared_ptr<rgbLine>(line));
    wf.resize(maxWFlength);
    addingFrame.unlock();
}

void waterfall::processLineToRGB(rgbLine* line)
{
    // go from float to RGB, with floor and ceiling scaling

    if(gammaLevel == 1.0)
    {
        for(int p=0; p < frWidth; p++)
        {
            line->getRed()[p] =   (unsigned char)MAX8(redLevel *   scaleDataPoint(line->getr_raw()[p]));
            line->getGreen()[p] = (unsigned char)MAX8(greenLevel * scaleDataPoint(line->getg_raw()[p]));
            line->getBlue()[p] =  (unsigned char)MAX8(blueLevel *  scaleDataPoint(line->getb_raw()[p]));
        }
    } else {
        for(int p=0; p < frWidth; p++)
        {
            line->getRed()[p] = (unsigned char)MAX8(redLevel * pow(scaleDataPoint(line->getr_raw()[p]), gammaLevel));
            line->getGreen()[p] = (unsigned char)MAX8(greenLevel * pow(scaleDataPoint(line->getg_raw()[p]), gammaLevel));
            line->getBlue()[p] = (unsigned char)MAX8(blueLevel * pow(scaleDataPoint(line->getb_raw()[p]), gammaLevel));
        }
    }
}

void waterfall::processLineToRGB_MP(rgbLine* line)
{
    // go from float to RGB, with floor and ceiling scaling

    if(gammaLevel == 1.0)
    {
#pragma omp parallel for num_threads(16)
        for(int p=0; p < frWidth; p++)
        {
            pthread_setname_np(pthread_self(), "GUI_WF");
            line->getRed()[p] =   (unsigned char)MAX8(redLevel *   scaleDataPoint(line->getr_raw()[p]));
            line->getGreen()[p] = (unsigned char)MAX8(greenLevel * scaleDataPoint(line->getg_raw()[p]));
            line->getBlue()[p] =  (unsigned char)MAX8(blueLevel *  scaleDataPoint(line->getb_raw()[p]));
        }
    } else {
#pragma omp parallel for num_threads(16)
        for(int p=0; p < frWidth; p++)
        {
            pthread_setname_np(pthread_self(), "GUI_WF");
            line->getRed()[p] = (unsigned char)MAX8(redLevel * pow(scaleDataPoint(line->getr_raw()[p]), gammaLevel));
            line->getGreen()[p] = (unsigned char)MAX8(greenLevel * pow(scaleDataPoint(line->getg_raw()[p]), gammaLevel));
            line->getBlue()[p] = (unsigned char)MAX8(blueLevel * pow(scaleDataPoint(line->getb_raw()[p]), gammaLevel));
        }
    }
}

unsigned char waterfall::scaleDataPoint(float dataPt)
{
    // This function must map:
    // FLOOR --  CEILING
    //  \/        \/
    //   0   --  255

    if(ceiling == floor)
        return 127;

    dataPt = dataPt - floor;

    if(dataPt > ceiling-floor)
    {
        dataPt = ceiling-floor;
    }

    // Now that the data are between ceiling and floor:
    float span = ceiling - floor;

    float factor = 255.0f / span;

    if (  (dataPt) * factor >= 0 )
    {
        return (dataPt) * factor;
    } else {
        return 0;
    }
}

void waterfall::handleNewFrame()
{
    // Called externally when a new frame is available.
    // We can add other functions that happen per-frame here.
    // But first, we will copy the frame in:
    addNewFrame();
    this->redraw();
    frameCount++;
    if(recordToJPG && (frameCount%maxWFlength == 0)) {
        saveImage();
    }
}

void waterfall::saveImage() {
    if(options.wfPreviewlocationset && saveImageReady) {
        QString filename = "AV3";
        QDateTime now = QDateTime::currentDateTime();
        QString dayStr = now.toUTC().toString("yyyyMMdd");
        QString timeStr = now.toUTC().toString("hhmmss");
        filename.append(QString("%1t%2-wf.jpg").arg(dayStr).arg(timeStr));
        statusMessage(QString("Writing waterfall image to filename [%1]")
                      .arg(options.wfPreviewLocation + filename));
        specImage.save(options.wfPreviewLocation + filename,
                       nullptr, jpgQuality);
    }
}

void waterfall::setRecordWFImage(bool recordImageOn) {
    if(isSecondary)
        return;

    recordToJPG = recordImageOn;
    if(recordToJPG && options.headless) {
        this->useDSF = true;
    }
}

void waterfall::setSecondaryWF(bool isSecondary) {
    this->isSecondary = isSecondary;
    if(isSecondary)
        recordToJPG = false;
}

void waterfall::changeRGB(int r, int g, int b)
{
    this->r_row = r;
    this->g_row = g;
    this->b_row = b;
}

void waterfall::setRGBLevels(double r, double g, double b, double gamma, bool reprocess)
{
    this->redLevel = r;
    this->greenLevel = g;
    this->blueLevel = b;
    this->gammaLevel = gamma;
    if(reprocess)
        rescaleWF();
}
void waterfall::setRGBLevelsAndReprocess(double r, double g, double b, double gamma)
{
    this->redLevel = r;
    this->greenLevel = g;
    this->blueLevel = b;
    this->gammaLevel = gamma;
    rescaleWF();
}

void waterfall::setSpecOpacity(unsigned char opacity)
{
    this->opacity = opacity;
}

void waterfall::changeWFLength(int length)
{
    // Note, this actually just changes
    // the length that is displayed.
    if(length <=0)
        return;

    if(length <= maxWFlength)
    {
        wflength = length;
    } else {
        wflength = maxWFlength;
    }
}

void waterfall::updateCeiling(int c)
{
    ceiling = c;
    rescaleWF();
}

void waterfall::updateFloor(int f)
{
    floor = f;
    rescaleWF();
}

void waterfall::setUseDSF(bool useDSF)
{
    this->useDSF = useDSF;
}

void waterfall::rescaleWF()
{
    // mutex lock
    // atomic bool spin-lock on deque data
    scalingValues.lock();
    QMutexLocker lock(&wfInUse);

#pragma omp parallel for num_threads(24)
    for(int wfrow=0; wfrow < maxWFlength; wfrow++)
    {
        pthread_setname_np(pthread_self(), "GUIRepro");
        processLineToRGB( wf[wfrow].get() );
    }
    scalingValues.unlock();

}

waterfall::wfInfo_t waterfall::getSettings() {

    waterfall::wfInfo_t info;
    info.wflength = this->wflength;
    info.ceiling = this->ceiling;
    info.floor = this->floor;
    info.useDSF = this->useDSF;
    info.r_row = this->r_row;
    info.g_row = this->g_row;
    info.b_row = this->b_row;
    info.redLevel = this->redLevel;
    info.greenLevel = this->greenLevel;
    info.blueLevel = this->blueLevel;
    info.gammaLevel = this->gammaLevel;
    info.recordToJPG = this->recordToJPG;
    info.jpgQuality = this->jpgQuality;
    return info;
}

void waterfall::debugThis()
{
    statusMessage("In debugThis function.");
    qDebug() << "isSecondary: " << isSecondary << ", r_row: " << r_row << ", g_row: " << g_row << ", b_row: " << b_row;
}

void waterfall::statusMessage(QString m)
{
    m.prepend(QString("WATERFALL: "));
    // Note: Messages made during the constructor might get emitted before
    // the console log is ready. Uncomment the next line to see them anyway:
#ifdef QT_DEBUG
    std::cout << m.toLocal8Bit().toStdString() << std::endl; fflush(stdout);
#endif
    emit statusMessageOut(m);
}
