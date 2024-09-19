#include "waterfall.h"

// This is the RGB waterfall widget used in the flight screen.
// The "waterfall" tab is handled by a special instance of the Frameview Widget.

waterfall::waterfall(QWidget *parent) : QWidget(parent) {
    // Widget constructor for inclusion in main window via Qt Designer.
    // basically don't do much yet.

}

waterfall::~waterfall() {
    if(isSecondary) {
        // do not delete image
        specImage = priorSpecImage;
        // and maybe delete it now?
    } else {
        // The specImage is deleted via the destructor
        // of the wf engine.
        // delete specImage;
    }
}

void waterfall::setup(frameWorker *fw, int vSize, int hSize, bool isSecondary, startupOptionsType options) {
    this->fw = fw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    this->options = options;
    this->isSecondary = isSecondary;

    // if not continuous mode, then if previewEnabled,
    // the flight widget will call the waterfall
    // to enable previews when recording.

    maxWFlength = 1024;
    wflength = maxWFlength;

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

    //specImage = new QImage(this->hSize, this->vSize, QImage::Format_ARGB32);
    //statusMessage(QString("Created specImage with height %1 and width %2.").arg(specImage->height()).arg(specImage->width()));

    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(cheapRedraw()));

    initialFPSSetting = TARGET_WF_FRAMERATE;
    resetFPS(TARGET_WF_FRAMERATE);
    statusMessage(QString("Setting initial target framerate to %1 FPS").arg(initialFPSSetting));

    connect(&FPSTimer, SIGNAL(timeout()), this, SLOT(computeFPS()));
    FPSElapsedTimer.start();
    FPSTimer.setInterval(1000);
    FPSTimer.setSingleShot(false);


    if(options.headless && (!options.wfPreviewEnabled)) {
        statusMessage("Not starting waterfall display update timer for headless mode without waterfall previews.");
    } else {
        //statusMessage("Starting waterfall");
        //rendertimer.start();
        //FPSTimer.start();
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

void waterfall::process()
{
    statusMessage("Thread started");
}

QImage* waterfall::getImage() {
    statusMessage("Returning specImage from basic waterfall widget.");
    return specImage;
}

void waterfall::setSpecImage(QImage *extSpecImage) {
    if(extSpecImage != NULL) {
        statusMessage("Switching to extSpecImage");
        statusMessage(QString("Using extSpecImage with height %1 and width %2.").arg(extSpecImage->height()).arg(extSpecImage->width()));

        statusMessage("Pausing render timer");
        rendertimer.stop();

        this->priorSpecImage = this->specImage;
        this->specImage = extSpecImage;
        followingExternalSpecImage = true;

        //connect(&rendertimer, SIGNAL(timeout()), this, SLOT(cheapRedraw()));
        statusMessage("Starting render timer.");

        rendertimer.start(); // EHL TODO: The problem is that this is called directly,
                            // and this the timer will start in the wrong thread.
        resetFPS(this->TARGET_WF_FRAMERATE);

        //FPSTimer.start(); // skip for now.
    } else {
        statusMessage("Error, priorSpecImage is NULL. Not sure what to do.");
        // do nothing
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

    //painter.setWindow(QRect(0, 0, hSize/4.0, 1024));
    painter.setWindow(QRect(0, 0, hSize, 1024));

    // Target: Where the image ultimately goes
    // Source: A rectangle which defines the portion of the specImage
    //         to copy into the target.

    // The source is stretched out over the target.

    // For length trimming, adjust the wflength parameter on the source
    // for Left-Right trimming, adjust the first two parameters of source.

    //QRectF target(0, 0, hSize/4.0, vSize);
    QRectF target(0, 0, hSize, vSize);

    QRectF source(0.0f, 0.0f, hSize, wflength); // use source geometry to "crop" the waterfall image
    painter.drawImage(target, *specImage, source);
}

void waterfall::cheapRedraw() {
    // Redraws from a given spec image, does not compute much.
    framesDelivered++;
    this->repaint(); // calls paintEvent
}

void waterfall::setSecondaryWF(bool isSecondary) {
    this->isSecondary = isSecondary;
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
        if(fps < 2)
            return; // no point or measurement is in error.

        if(isSecondary) {
            if(fps < TARGET_WF_FRAMERATE_SECONDARY*0.90) {
                fpsUnderEvents++;
                if(fpsUnderEvents > fpsUEThreshold) {
                    unsigned int newFR = (TARGET_WF_FRAMERATE_SECONDARY + fps) / 2;
                    if(newFR > minimumFPS) {
                        TARGET_WF_FRAMERATE_SECONDARY = newFR;
                        WF_DISPLAY_PERIOD_MSECS_SECONDARY = 1000 / TARGET_WF_FRAMERATE_SECONDARY;
                        rendertimer.setInterval(WF_DISPLAY_PERIOD_MSECS_SECONDARY);
                        debugMessage(QString("Adjusting FPS down to %1 FPS. Minimum allowed is %2, observed is %3.")
                                     .arg(TARGET_WF_FRAMERATE_SECONDARY)
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
                    metFPS = 0;
                }
            } else {
                fpsUnderEvents = 0; //reset, only care about FPS under in a row.
                if(fps > (((float)TARGET_WF_FRAMERATE_SECONDARY)*0.95)) {
                    metFPS = metFPS>1024?1024:metFPS+1; // clamp at 1024
                    debugMessage(QString("Meeting 95% FPS. Thresh: %1, got %2, metFPS count: %3")
                                 .arg(TARGET_WF_FRAMERATE_SECONDARY*0.95).arg(fps).arg(metFPS));
                } else {
                    metFPS = 0;
                }
            }

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
                        rendertimer.setInterval(WF_DISPLAY_PERIOD_MSECS);
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

void waterfall::resetFPS(int desiredFPS) {
    if(desiredFPS >= (int)minimumFPS/5) {
        if(isSecondary) {
            TARGET_WF_FRAMERATE_SECONDARY = desiredFPS;
            WF_DISPLAY_PERIOD_MSECS_SECONDARY = 1000 / TARGET_WF_FRAMERATE_SECONDARY;
            rendertimer.setInterval(WF_DISPLAY_PERIOD_MSECS_SECONDARY);
            statusMessage(QString("Adjusting FPS setpoint to %1 FPS.").arg(TARGET_WF_FRAMERATE_SECONDARY));
        } else {
            TARGET_WF_FRAMERATE = desiredFPS;
            WF_DISPLAY_PERIOD_MSECS = 1000 / TARGET_WF_FRAMERATE;
            rendertimer.setInterval(WF_DISPLAY_PERIOD_MSECS);
            statusMessage(QString("Adjusting FPS setpoint to %1 FPS.").arg(TARGET_WF_FRAMERATE));
        }
        fpsUnderEvents = 0;
        //metFPS = 0;
    }
}

void waterfall::debugThis()
{
    statusMessage("In debugThis function.");
}

void waterfall::debugMessage(QString m) {
#ifndef QT_DEBUG
    return;
#endif
    if(isSecondary) {
        m.prepend(QString("DBG WATERFALL (2): "));
    } else {
        m.prepend(QString("DBG WATERFALL (1): "));
    }
    std::cout << m.toLocal8Bit().toStdString() << std::endl; fflush(stdout);
    emit statusMessageOut(m);
}

void waterfall::statusMessage(QString m)
{
    if(isSecondary) {
        m.prepend(QString("WATERFALL (2): "));
    } else {
        m.prepend(QString("WATERFALL (1): "));
    }
    // Note: Messages made during the constructor might get emitted before
    // the console log is ready. Uncomment the next line to see them anyway:
#ifdef QT_DEBUG
    std::cout << m.toLocal8Bit().toStdString() << std::endl; fflush(stdout);
#endif
    emit statusMessageOut(m);
}
