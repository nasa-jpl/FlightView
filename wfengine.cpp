#include "wfengine.h"

// This is the RGB waterfall widget used in the flight screen.
// The "waterfall" tab is handled by a special instance of the Frameview Widget.

wfengine::wfengine(QWidget *parent) : QObject(parent) {
    // Widget constructor for inclusion in main window via Qt Designer.
    // basically don't do much yet.
#ifdef QT_DEBUG
    std::cerr << "Constructing wfengine. Thread Name: " << this->thread()->objectName().toStdString() << std::endl;
    std::cerr << "wfengine This pointer: " << Qt::hex << this << std::endl;
#endif
}

wfengine::~wfengine() {
#ifdef QT_DEBUG
    std::cerr << "wfengine destructor called.\n";
    std::cerr << "wfengine Thread Name: " << this->thread()->objectName().toStdString() << std::endl;
    std::cerr << "wfengine This pointer: " << Qt::hex << this << std::endl;
#endif

    if(specImage) {
        std::cerr << "wfengine deleting specImage.\n";
        delete specImage;
    } else {
        std::cerr << "wfengine not deleting specImage.\n";
    }
}

void wfengine::setParameters(frameWorker *fw, int vSize, int hSize, startupOptionsType options) {
    this->fw = fw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    this->options = options;
    this->vSize = vSize;
    this->hSize = hSize;
}

void wfengine::setup() {
#ifdef QT_DEBUG
    std::cerr << "wfengine setup called.\n";
    std::cerr << "wfengine Running setup for wfengine. Thread Name: " << this->thread()->objectName().toStdString() << std::endl;
    std::cerr << "wfengine This pointer: " << Qt::hex << this << std::endl;
#endif

    this->fw = fw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    this->options = options;
    this->isSecondary = isSecondary;

    recordToJPG = options.wfPreviewContinuousMode;

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

    rendertimer = new QTimer();

    connect(rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));

    initialFPSSetting = TARGET_WF_FRAMERATE;
    resetFPS(TARGET_WF_FRAMERATE);
    statusMessage(QString("Setting initial target framerate to %1 FPS").arg(initialFPSSetting));

    FPSTimer = new QTimer();
    connect(FPSTimer, SIGNAL(timeout()), this, SLOT(computeFPS()));
    FPSElapsedTimer.start();
    FPSTimer->setInterval(1000);
    FPSTimer->setSingleShot(false);


    if(options.headless && (!options.wfPreviewEnabled)) {
        statusMessage("Not starting waterfall display update timer for headless mode without waterfall previews.");
    } else {
        statusMessage("Starting waterfall");
        rendertimer->start();
        FPSTimer->start();
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
    statusMessage("Finished waterfall setup.");
    emit wfReady(); // now the specImage is available.
    emit hereIsTheImage(specImage); // send the image out
}

wfengine::wfengine(frameWorker *fw, int vSize, int hSize, startupOptionsType options, QWidget *parent) : QObject(parent)
{
    statusMessage("Finished waterfall constructor.");
}

void wfengine::prepareWfImage() {
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

void wfengine::process()
{
    statusMessage("Thread started");
}

void wfengine::stop() {
#ifdef QT_DEBUG
    std::cerr << "wfengine stop, Thread Name: " << this->thread()->objectName().toStdString() << std::endl;
    std::cerr << "wfengine stop, This pointer: " << Qt::hex << this << std::endl;
#endif
    timeToStop = true;
    rendertimer->stop();
    FPSTimer->stop();
}

QImage* wfengine::getImage() {
    // This is how the pointer to the image
    // is communicated to the widget(s)
    // drawing the waterfall for display.
    if(specImage==NULL) {
        statusMessage("WARNING, waterfall image pointer being returned is NULL!");
    }
    return specImage;
}

void wfengine::redraw()
{
    // Copy the waterfall data into the specImage.
    // To increase speed, we are ignoring the alpha (opacity) value
    // Called manually by handleNewFrame

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
}

void wfengine::allocateBlankWF()
{
    // Static waterfall allocation:
    for(int n=0; n < maxWFlength; n++)
    {
        rgbLine* line = new rgbLine(frWidth, false);
        wflines[n] = line;
    }
}

void wfengine::copyPixToLine(float *image, float *dst, int rowSelection)
{
    for(int p=0; p < frWidth; p++)
    {
        if(dst != NULL)
            dst[p] = image[ rowSelection + p];
    }
}

void wfengine::copyPixToLine(uint16_t *image, float *dst, int rowSelection)
{
    for(int p=0; p < frWidth; p++)
    {
        if(dst != NULL)
            dst[p] = (float)image[ rowSelection + p];
    }
}

void wfengine::addNewFrame()
{
    if(timeToStop)
        return;
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

void wfengine::processLineToRGB(rgbLine* line)
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

void wfengine::processLineToRGB_MP(rgbLine* line)
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

unsigned char wfengine::scaleDataPoint(float dataPt)
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

void wfengine::requestImage() {
    statusMessage("Returning image via signal/slot method");
    emit hereIsTheImage(specImage);
}

void wfengine::handleNewFrame()
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

void wfengine::setGPSStart(gpsMessage m) {
    this->startGPSMessage = m;
}

void wfengine::setGPSEnd(gpsMessage m) {
    this->destGPSMessage = m;
}

void wfengine::immediatelySaveImage() {
    statusMessage("Immediately saving current waterfall image.");
    saveImage();
}

void wfengine::saveImage() {
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
#ifdef WF_GPS_TAGGING
        QString fileLocationFull = options.wfPreviewLocation + filename;
        bool success = imageTagger(fileLocationFull.toLocal8Bit(),
                                   startGPSMessage, destGPSMessage, fps,
                                   r_row, g_row, b_row,
                                   gammaLevel, floor, ceiling);
        if(!success) {
            statusMessage(QString("Warning, could not modify metadata for waterfall image."));
        }
#endif
    }
}

void wfengine::setRecordWFImage(bool recordImageOn) {
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

void wfengine::setSecondaryWF(bool isSecondary) {
    this->isSecondary = isSecondary;
    if(isSecondary)
        recordToJPG = false;
}

void wfengine::changeRGB(int r, int g, int b)
{
    this->r_row = r;
    this->g_row = g;
    this->b_row = b;
}

void wfengine::setRGBLevels(double r, double g, double b, double gamma, bool reprocess)
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
void wfengine::setRGBLevelsAndReprocess(double r, double g, double b, double gamma)
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

void wfengine::setSpecOpacity(unsigned char opacity)
{
    this->opacity = opacity;
}

void wfengine::changeWFLength(int length)
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

void wfengine::updateCeiling(int c)
{
    ceiling = c;
    rescaleWF();
}

void wfengine::updateFloor(int f)
{
    floor = f;
    rescaleWF();
}

void wfengine::setUseDSF(bool useDSF)
{
    this->useDSF = useDSF;
}

void wfengine::rescaleWF()
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

wfengine::wfInfo_t wfengine::getSettings() {

    wfengine::wfInfo_t info;
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

void wfengine::computeFPS() {
    // Called every one second for debug reasons:
    float timeElapsed = FPSElapsedTimer.elapsed() / 1000.0;

    if(timeElapsed != 0) {
        fps = framesDelivered/timeElapsed;
#ifdef QT_DEBUG
#ifdef WF_DEBUG_FPS
        QString s;
        s = QString("framesDelivered: %2, timeElapsed: %3, FPS: %4")
                .arg(framesDelivered).arg(timeElapsed).arg(fps);
        statusMessage(s);
#endif
#endif
#ifdef dynamicFPS
        if(fps < 2) {
            statusMessage("FPS Warning. Checking Render Timer now.");
            if(rendertimer->isActive()) {
                statusMessage(QString("Render timer is active. Interval: %1 ms").arg(rendertimer->interval()));
            } else {
                statusMessage("Render timer is NOT active.");
                if(this->timeToStop)
                    return;
                // rendertimer->start();
            }
            return; // no point or measurement is in error.
        }

        if(isSecondary) {

        } else {
            if(fps < (TARGET_WF_FRAMERATE*0.90)) {
                fpsUnderEvents++;
                debugMessage(QString("Not meeting FPS. Expected > %1, got %2, under events %3")
                             .arg(TARGET_WF_FRAMERATE*0.90).arg(fps).arg(fpsUnderEvents));
                if(fpsUnderEvents > fpsUEThreshold) {
                    unsigned int newFR = (TARGET_WF_FRAMERATE + fps) / 2;
                    if(newFR > minimumFPS) {
                        TARGET_WF_FRAMERATE = newFR;
                        WF_DISPLAY_PERIOD_MSECS = 1000 / TARGET_WF_FRAMERATE;
                        rendertimer->setInterval(WF_DISPLAY_PERIOD_MSECS);
                        debugMessage(QString("Adjusting FPS down to %1 FPS. Minimum allowed is %2, observed is %3.")
                                     .arg(TARGET_WF_FRAMERATE)
                                     .arg(minimumFPS).arg(fps));
                        justMovedDownFPS = true;
                        if(justMovedUpFPS) {
                            flipFlopFPSCounter++;
                            justMovedUpFPS = false;
                        } else {
                            flipFlopFPSCounter = 0;
                        }
                    }
                    fpsUnderEvents = 0;
                    metFPS = 0; // we did not meet the FPS, zero the counter.
                }
            } else {
                fpsUnderEvents = 0; //reset, only care about FPS under in a row.
                if(fps > (((float)TARGET_WF_FRAMERATE)*0.95)) {
                    // We only count this to haev been met if we were at or better than 90% of the requested frame rate.
                    metFPS = metFPS>1024?1024:metFPS+1; // clamp at 1024
                    debugMessage(QString("Meeting 95% FPS. Thresh: %1, got %2, metFPS count: %3")
                                 .arg(TARGET_WF_FRAMERATE*0.95).arg(fps).arg(metFPS));
                } else {
                    metFPS = 0;
                }
            }
        }

        // TODO: Secondary display
        // Now, if we did not just decrease the FPS and if we have met the 95% FPS 15 times in a row,
        // Then we can consider increasing the FPS.
        if(metFPS > 15) {
            // we can, potentially, bump up the FPS.
            if(flipFlopFPSCounter > 10) {
                // We have flip-flopped ten times, time to lock it in.
                justMovedUpFPS = false;
                debugMessage(QString("Detected FPS up/down flipflop, not raising FPS"));
            } else {
                // bump it up
                if(justMovedDownFPS) {
                    flipFlopFPSCounter++;
                    justMovedDownFPS = false;
                }
                if(((unsigned int)(fps+1.5)) < maximumFPS) {
                    debugMessage(QString("Increasing FPS to %1, observed framerate is %2")
                                 .arg((int)(fps+1.5))
                                 .arg(fps));
                    resetFPS((int)(fps+1.5));
                }
                justMovedUpFPS = true;
                metFPS = 0;
            }
        }
#endif
    }

    FPSElapsedTimer.restart();
    framesDelivered = 0;
}

void wfengine::resetFPS(int desiredFPS) {
    if(desiredFPS >= (int)minimumFPS/5) {

            TARGET_WF_FRAMERATE = desiredFPS;
            WF_DISPLAY_PERIOD_MSECS = 1000 / TARGET_WF_FRAMERATE;
            rendertimer->setInterval(WF_DISPLAY_PERIOD_MSECS);
            statusMessage(QString("Adjusting FPS setpoint to %1 FPS.").arg(TARGET_WF_FRAMERATE));

        fpsUnderEvents = 0;
        //metFPS = 0;
    }
}

void wfengine::debugThis()
{
    statusMessage("In debugThis function.");
}

void wfengine::debugMessage(QString m) {
#ifndef QT_DEBUG
    return;
#endif
    m.prepend(QString("DBG WF ENGINE: "));

    std::cout << m.toLocal8Bit().toStdString() << std::endl; fflush(stdout);
    emit statusMessageOut(m);
}

void wfengine::statusMessage(QString m)
{
    m.prepend(QString("WF ENGINE: "));

    // Note: Messages made during the constructor might get emitted before
    // the console log is ready. Uncomment the next line to see them anyway:
#ifdef QT_DEBUG
    std::cout << m.toLocal8Bit().toStdString() << std::endl; fflush(stdout);
#endif
    emit statusMessageOut(m);
}
