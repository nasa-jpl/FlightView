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

    maxWFlength = 1024;
    wflength = maxWFlength;
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

    specImage = new QImage(this->hSize, this->vSize, QImage::Format_ARGB32);
    statusMessage(QString("Created specImage with height %1 and width %2.").arg(specImage->height()).arg(specImage->width()));

    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
    if(isSecondary) {
        rendertimer.setInterval(WF_DISPLAY_PERIOD_MSECS_SECONDARY);
    } else {
        rendertimer.setInterval(WF_DISPLAY_PERIOD_MSECS);
    }

    connect(&FPSTimer, SIGNAL(timeout()), this, SLOT(computeFPS()));
    FPSElapsedTimer.start();
    FPSTimer.setInterval(1000);
    FPSTimer.setSingleShot(false);


    if(options.headless && (!options.wfPreviewEnabled)) {
        statusMessage("Not starting waterfall display update timer for headless mode without waterfall previews.");
    } else {
        statusMessage("Starting waterfall");
        rendertimer.start();
        FPSTimer.start();
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
    statusMessage("Finished waterfall setup.");
}

waterfall::waterfall(frameWorker *fw, int vSize, int hSize, startupOptionsType options, QWidget *parent) : QWidget(parent)
{
    setup(fw, vSize, hSize, false, options);
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

QImage* waterfall::getImage() {
    return specImage;
}

void waterfall::setSpecImage(bool followMe, QImage *extSpecImage) {
    if(followMe && (extSpecImage != NULL)) {
        statusMessage("Switching to extSpecImage");
        statusMessage("Locking mutexes and pausing render timer");
        rendertimer.stop();
        addingFrame.lock();
        scalingValues.lock();

        QMutexLocker lockwf(&wfInUse);
        this->priorSpecImage = this->specImage;
        this->specImage = extSpecImage;
        followingExternalSpecImage = true;
        statusMessage("Disconnecting render timer from handleNewFrame");
        disconnect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
        statusMessage("Connecting render time directly to cheapRedraw function");
        connect(&rendertimer, SIGNAL(timeout()), this, SLOT(cheapRedraw()));
        statusMessage("Starting render timer.");
        addingFrame.unlock();
        scalingValues.unlock();
        rendertimer.start();
    } else {
        // Either we were told not to follow or we were given a NULL pointer,
        // in either case we go back to using our old image.
        if(priorSpecImage!=NULL) {
            statusMessage("Reverting to priorSpecImage");
            statusMessage("Locking mutexes and pausing render timer");
            rendertimer.stop();
            scalingValues.lock();
            addingFrame.lock();
            QMutexLocker lockwf(&wfInUse);
            followingExternalSpecImage = false;
            this->specImage = this->priorSpecImage;
            connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
            addingFrame.unlock();
            scalingValues.unlock();
            rendertimer.start();
        } else {
            statusMessage("Error, priorSpecImage is NULL. Not sure what to do.");
            // do nothing
        }
    }
}

void waterfall::paintEvent(QPaintEvent *event)
{
    if(specImage == NULL)
        return;

    QPainter painter(this);


    // anti-alias:
    painter.setRenderHint(QPainter::SmoothPixmapTransform); // smooth images
    //painter.setRenderHint(QPainter::HighQualityAntialiasing); // blocky
    //painter.setRenderHint(QPainter::Antialiasing); // blocky


    painter.setWindow(QRect(0, 0, hSize/4.0, 1024));

    // Target: Where the image ultimately goes
    // Source: A rectangle which defines the portion of the specImage
    //         to copy into the target.

    // The source is stretched out over the target.

    // For length trimming, adjust the wflength parameter on the source
    // for Left-Right trimming, adjust the first two parameters of source.

    QRectF target(0, 0, hSize/4.0, vSize);
    QRectF source(0.0f, 0.0f, hSize, wflength); // use source geometry to "crop" the waterfall image
        painter.drawImage(target, *specImage, source);
}

void waterfall::cheapRedraw() {
    // Redraws from a given spec image, does not compute much.
    framesDelivered++;
    this->repaint();
}

void waterfall::redraw()
{
    // Copy the waterfall data into the specImage.
    // To increase speed, we are ignoring the alpha (opacity) value

    QColor c;
    QRgb *line = NULL;
    //rgbLine *cl;
    // new method:
    unsigned char *r = NULL;
    unsigned char *g = NULL;
    unsigned char *b = NULL;

    int wfpos = currentWFLine-1;

    // Row zero of the specImage is the bottom
    for(int y=maxWFlength; y > 0; y--) {
        line = (QRgb*)specImage->scanLine(y-1);
        wfpos = (wfpos+1)%maxWFlength;
        r = wflines[wfpos]->getRed();
        g = wflines[wfpos]->getGreen();
        b = wflines[wfpos]->getBlue();

        for(int x = 0; x < hSize; x++)
        {
            c.setRgb(r[x],
                     g[x],
                     b[x]);
            line[x] = c.rgb();
        }
    }

    framesDelivered++;
    this->repaint();
}

void waterfall::allocateBlankWF()
{
    // Static waterfall allocation:
    for(int n=0; n < maxWFlength; n++)
    {
        rgbLine* line = new rgbLine(frWidth, false);
        wflines[n] = line;
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

    rgbLine *line = wflines[currentWFLine];

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
    //processLineToRGB(line); // single processor
    processLineToRGB_MP(line); // multi-processor


    //QMutexLocker lockwf(&wfInUse);

    //wf.push_front(std::shared_ptr<rgbLine>(line));
    //wf.resize(maxWFlength);
    currentWFLine = (currentWFLine + 1) % maxWFlength;
    addingFrame.unlock();
}

void waterfall::processLineToRGB(rgbLine* line)
{
    // go from float to RGB, with floor and ceiling scaling

    if(!useGamma)
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
    // Note: If concurrency is set too high, then the hit
    // taken is actually worse than single-cpu mode.
    // Recommended value is 4.

    float *r = line->getr_raw();
    float *g = line->getg_raw();
    float *b = line->getb_raw();

    unsigned char *gr = line->getRed();
    unsigned char *gg = line->getGreen();
    unsigned char *gb = line->getBlue();

    if(!useGamma)
    {
#pragma omp parallel for num_threads(4)
        for(int p=0; p < frWidth; p++)
        {
            pthread_setname_np(pthread_self(), "GUI_WF");            
            gr[p] =   (unsigned char)MAX8(redLevel *   scaleDataPoint(r[p]));
            gg[p] = (unsigned char)MAX8(greenLevel * scaleDataPoint(g[p]));
            gb[p] =  (unsigned char)MAX8(blueLevel *  scaleDataPoint(b[p]));
        }
    } else {
#pragma omp parallel for num_threads(4)
        for(int p=0; p < frWidth; p++)
        {
            pthread_setname_np(pthread_self(), "GUI_WF_G");
            gr[p] = (unsigned char)MAX8(redLevel * pow(scaleDataPoint(r[p]), gammaLevel));
            gg[p] = (unsigned char)MAX8(greenLevel * pow(scaleDataPoint(g[p]), gammaLevel));
            gb[p] = (unsigned char)MAX8(blueLevel * pow(scaleDataPoint(b[p]), gammaLevel));
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
    // Called via the renderTimer at regular intervals.
    // We can add other functions that happen per-frame here.
    // But first, we will copy the frame in:
    addNewFrame();
    this->redraw();
    frameCount++;

    if(saveImageReady) {
        // Either we are in a recording and it's time to save, or we are just starting, or we are just ending.
        if(  ((recordingStartLineNumber == (frameCount%maxWFlength)) && recordToJPG) || (justStartedRecording) || (justStoppedRecording)  ) {
            saveImage();
            // Clear the flags here
            justStartedRecording = false;
            justStoppedRecording = false;
        }
    }
}

void waterfall::immediatelySaveImage() {
    statusMessage("Immediately saving current waterfall image.");
    saveImage();
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
        specImage->save(options.wfPreviewLocation + filename,
                       nullptr, jpgQuality);
    }
}

void waterfall::setRecordWFImage(bool recordImageOn) {
    if(isSecondary)
        return;

    recordToJPG = recordImageOn;
    recordingStartLineNumber = (currentWFLine - 1)%maxWFlength;
    if(recordImageOn) {
        justStartedRecording = true;
    }
    if(!recordImageOn) {
        justStoppedRecording = true;
    }
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
    if( (gamma > 0.999) && (gamma < 1.001) ) {
        useGamma = false;
    } else {
        useGamma = true;
    }
    if(reprocess)
        rescaleWF();
}
void waterfall::setRGBLevelsAndReprocess(double r, double g, double b, double gamma)
{
    this->redLevel = r;
    this->greenLevel = g;
    this->blueLevel = b;
    this->gammaLevel = gamma;
    if( (gamma > 0.999) && (gamma < 1.001) ) {
        useGamma = false;
    } else {
        useGamma = true;
    }
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
        processLineToRGB( wflines[wfrow] );
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

void waterfall::computeFPS() {
    // Called every one second for debug reasons:
    float timeElapsed = FPSElapsedTimer.elapsed() / 1000.0;
#ifdef QT_DEBUG
#ifdef WF_DEBUG_FPS
    if(timeElapsed != 0) {
        float fps = framesDelivered/timeElapsed;
        QString s;
        s = QString("isSecondary: %1, framesDelivered: %2, timeElapsed: %3, FPS: %4")
                .arg(isSecondary).arg(framesDelivered).arg(timeElapsed).arg(fps);
        statusMessage(s);
    }
#endif
#endif
    FPSElapsedTimer.restart();
    framesDelivered = 0;
}

void waterfall::debugThis()
{
    statusMessage("In debugThis function.");
}

void waterfall::statusMessage(QString m)
{
    if(isSecondary) {
        m.prepend(QString("WATERFALL (2): "));
    } else {
        m.prepend(QString("WATERFALL: "));
    }
    // Note: Messages made during the constructor might get emitted before
    // the console log is ready. Uncomment the next line to see them anyway:
#ifdef QT_DEBUG
    std::cout << m.toLocal8Bit().toStdString() << std::endl; fflush(stdout);
#endif
    emit statusMessageOut(m);
}
