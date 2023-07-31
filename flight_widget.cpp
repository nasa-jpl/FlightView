#include "flight_widget.h"

flight_widget::flight_widget(frameWorker *fw, startupOptionsType options, QWidget *parent) : QWidget(parent)
{
    //connect(this, SIGNAL(statusMessage(QString)), this, SLOT(showDebugMessage(QString)));

    qDebug() << "Running flight widget constructor";
    emit statusMessage(QString("Starting flight screen widget"));
    fi = new flightIndicators();
    fiUI_t flightDisplayElements = fi->getElements();

    if(flightDisplayElements.lastIssueLabel == NULL) {
        qDebug() << "ERROR lastIssueLabel is NULL!!";
    }

    stickyFPSError = false;
    FPSErrorCounter = 0;
    this->fw = fw;
    this->options = options;
    useAvionicsWidgets = false;
    gpsPlotSplitter = new QSplitter();

    waterfall_widget = new waterfall(fw, 1, 1024, this);
//    waterfall_widget = new waterfall(fw, 1, 1024, NULL);

    wfThread = new QThread(this);
    wfThread->setObjectName("lv:wfThread");
    dsf_widget = new frameview_widget(fw, DSF, this);

    gpsMessageCycleTimer = new QTimer(this);
    gpsMessageCycleTimer->setInterval(1500);

    gpsMessageToLogReporterTimer = new QTimer(this);
    gpsMessageToLogReporterTimer->setInterval(60*1000);

    if(options.flightMode)
    {
        connect(gpsMessageToLogReporterTimer, SIGNAL(timeout()),
                this, SLOT(gpsMessageToLogReporterSlot()));
        gpsMessageToLogReporterTimer->start();
    }

    startedPrimaryGPSLog = false;
    gps = new gpsManager();

    if(useAvionicsWidgets)
    {
        // NOTE: If a widget is set to NULL, it will not
        // be updated and will not be added to the layout.
        // Simply comment out the widgets not desired.
    }

    // Group Box "Flight Instrument Controls" items:
    resetStickyErrorsBtn.setText("Clear Errors");
    resetStickyErrorsBtn.setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    gpsLED.setState(QLedLabel::StateOkBlue);

    if(options.rtpCam)
    {
        updateLabel(cameraLinkLEDLabel, "RTP Link");
    } else if (options.xioCam) {
        updateLabel(cameraLinkLEDLabel, "XIO Files");
    } else {
        updateLabel(cameraLinkLEDLabel, "C Link");
    }

    if(cameraLinkLED != NULL) {
        cameraLinkLED->setState(QLedLabel::StateOk);
    }
    diskLEDLabel.setText("Disk:");
    if(diskLED != NULL) {
        diskLED->setState(QLedLabel::StateOkBlue);
    }

    // Format is &item, row, col, rowSpan, colSpan. -1 = to "edge"

    flightControlLayout.addWidget(fi, 0, 0, 1,-1);



    // Avionics Widget Layout Placement
    if(useAvionicsWidgets)
    {
        // Note: If widget size is > 150, widgets will generally
        // span larger than one column. A window resize
        // will cause the widgets to re-draw and fit nicely without
        // having to fine-tune the layout.

        // Arguments are: widgetPointer, ROW, COL, ROW SPAN, COL SPAN
        int avColumn = 7; // starting at column 7
        if(EADI!=NULL) flightControlLayout.addWidget(EADI, 4, avColumn--, -1, 1); // col 7
        if(EHSI!=NULL) flightControlLayout.addWidget(EHSI, 4, avColumn--, -1, 1); // col 6
        if(ASI!=NULL)  flightControlLayout.addWidget(ASI,  4, avColumn--, -1, 1); // col 5
        if(VSI!=NULL)  flightControlLayout.addWidget(VSI,  4, avColumn--, -1, 1); // col 4
    }


    // Group Box "Flight Instrument Controls"
    flightControls.setTitle("Instrumentation Status");
    flightControls.setLayout(&flightControlLayout);

    rhSplitter.setOrientation(Qt::Vertical);
    rhSplitter.addWidget(dsf_widget);
    rhSplitter.addWidget(&flightControls);
    rhSplitter.setHandleWidth(10);

    lrSplitter.addWidget(waterfall_widget);
    lrSplitter.addWidget(&rhSplitter);

    lrSplitter.setHandleWidth(10);

    layout.addWidget(&lrSplitter);

    this->setLayout(&layout);

    connect(waterfall_widget, SIGNAL(statusMessageOut(QString)), this, SLOT(showDebugMessage(QString)));

    // Connections to GPS:
    connect(gps, SIGNAL(gpsStatusMessage(QString)), this, SLOT(showDebugMessage(QString)));
    connect(gps, SIGNAL(gpsConnectionError(int)), this, SLOT(handleGPSConnectionError(int)));
    connect(gps, SIGNAL(statusMessagesSig(QStringList,QStringList)), this, SLOT(handleGPSStatusMessages(QStringList,QStringList)));
    gps->insertLEDs(flightDisplayElements.gpsLinkLED, flightDisplayElements.gpsTroubleLED);

    connect(gpsMessageCycleTimer, SIGNAL(timeout()), this, SLOT(cycleGPSStatusMessagesViaTimer()));
    gpsMessageCycleTimer->start();

    diskLED = flightDisplayElements.diskLED;
    cameraLinkLED = flightDisplayElements.imageLED;

    // Unused labels are Roll, Pitch, and Rate of Climb
    gps->insertLabels(flightDisplayElements.latLabel , flightDisplayElements.longLabel, flightDisplayElements.altitudeLabel,
                      NULL, NULL, NULL,
                      flightDisplayElements.groundSpeedLabel,
                      flightDisplayElements.headingLabel, NULL, NULL,
                      NULL, flightDisplayElements.alignmentLabel,
                      NULL);

    if(useAvionicsWidgets)
    {
        gps->insertAvionicsWidgets(ASI, VSI, EADI, EHSI);
    }

    gps->prepareElements();

    if(options.flightMode && !options.disableGPS)
    {
        emit statusMessage(QString("Starting liveview in FLIGHT mode."));
        // Connecto to GPS immediately and start logging.
        if(options.gpsIPSet && options.gpsPortSet && options.dataLocationSet)
        {
            this->startGPS(options.gpsIP,
                            options.gpsPort,
                            options.dataLocation);
        } else {
            emit statusMessage(QString("ERROR, cannot start GPS in flight mode with incomplete GPS settings."));
        }
    } else {
        emit statusMessage(QString("Starting liveview in LAB mode."));
        if(options.gpsIPSet && !options.disableGPS && options.dataLocationSet && (!options.disableGPS))
        {
            emit statusMessage(QString("Starting GPS in LAB mode."));
            this->startGPS(options.gpsIP,
                            options.gpsPort,
                            options.dataLocation);
        } else {
            emit statusMessage(QString("Not starting GPS."));
        }
    }

    // TODO:
    // set base filename for master (primary) log

    // connect start and stop signals for secondary log
    connect(this, &flight_widget::beginSecondaryLog, gps, &gpsManager::handleStartsecondaryLog);
    connect(this, &flight_widget::stopSecondaryLog, gps, &gpsManager::handleStopSecondaryLog);


    connect(fi, SIGNAL(clearErrors()), this, SLOT(resetFPSError()));
    connect(fi, SIGNAL(clearErrors()), gps, SLOT(clearStickyError()));
    connect(fi, SIGNAL(clearErrors()), this, SLOT(clearStickyErrors()));
    connect(this, SIGNAL(haveGPSErrorWarningMessage(QString)), fi, SLOT(updateLastIssue(QString)));

    diskCheckerTimer = new QTimer();
    diskCheckerTimer->setInterval(1000);
    diskCheckerTimer->setSingleShot(false);
    connect(diskCheckerTimer, SIGNAL(timeout()), this, SLOT(checkDiskSpace()));
    diskCheckerTimer->start();

    hideRGBTimer.setInterval(30000);
    hideRGBTimer.setSingleShot(true);
    hideRGBTimer.stop();
    connect(&hideRGBTimer, SIGNAL(timeout()), this, SLOT(hideRGB()));

    waterfall_widget->moveToThread(wfThread);
    connect(wfThread, SIGNAL(started()), waterfall_widget, SLOT(process()));
    connect(wfThread, SIGNAL(finished()), waterfall_widget, SLOT(deleteLater()));
    wfThread->start();

    setupWFConnections();    
    QList <int>rhSS;
    rhSS.append(514);
    rhSS.append(197);
    rhSplitter.setSizes(rhSS);
    rhSplitter.setStretchFactor(0, 2);
    rhSplitter.setStretchFactor(1, 0); // do not stretch the indicators
    QList <int>lrSS;
    lrSS.append(830);
    lrSS.append(684);
    lrSplitter.setSizes(lrSS);
    lrSplitter.setStretchFactor(0, 2);
    lrSplitter.setStretchFactor(1, 0); // do not stretch the indicators
    emit statusMessage(QString("Finished flight constructor."));
}

flight_widget::~flight_widget()
{
    qDebug() << "Running flight_widget destructor.";
    if(wfThread != NULL)
    {
        wfThread->quit();
        wfThread->wait();
        //usleep(10000);
    }

//    if(gps != NULL)
//    {
//        gps->initiateGPSDisconnect();
//        usleep(1000);
//        gps->deleteLater();
//        usleep(1000);
//        //delete gps;
//    }
}

void flight_widget::setupWFConnections()
{
    connect(this, SIGNAL(changeWFLengthSignal(int)), waterfall_widget, SLOT(changeWFLength(int)));
    connect(this, SIGNAL(updateCeilingSignal(int)), waterfall_widget, SLOT(updateCeiling(int)));
    connect(this, SIGNAL(updateFloorSignal(int)), waterfall_widget, SLOT(updateFloor(int)));
    connect(this, SIGNAL(setRGBLevelsSignal(double,double,double,double)), waterfall_widget, SLOT(setRGBLevels(double,double,double,double)));
    connect(this, SIGNAL(updateRGBbandSignal(int,int,int)), waterfall_widget, SLOT(changeRGB(int,int,int)));
}

double flight_widget::getCeiling()
{
    return dsf_widget->getCeiling();
}

double flight_widget::getFloor()
{
    return dsf_widget->getFloor();
}

void flight_widget::setUseDSF(bool useDSF)
{
    waterfall_widget->setUseDSF(useDSF);
    dsf_widget->setUseDSF(useDSF);
}

void flight_widget::toggleDisplayCrosshair()
{
    dsf_widget->toggleDisplayCrosshair();
}

void flight_widget::hideRGB()
{
    // Timer signal "timeout" hits here.
    hideRGBTimer.stop();
    if(!showRGBp)
        dsf_widget->toggleDrawRGBRow(false);
}

void flight_widget::handleNewFrame()
{
    dsf_widget->handleNewFrame();
    waterfall_widget->handleNewFrame();
}

void flight_widget::updateFPS()
{
    if(cameraLinkLED != NULL) {
        if(fw->delta < 12.8f)
        {
            processFPSError();
        } else if (fw->delta < 13.0f) {
            this->cameraLinkLED->setState(QLedLabel::StateWarning);
        } else if ((fw->delta > 13.0f) && !stickyFPSError)
        {
            // to reset the warning, but not the sticky error:
            this->cameraLinkLED->setState(QLedLabel::StateOk);
        }
    }
}

void flight_widget::checkDiskSpace()
{
    if(options.dataLocationSet)
    {
        diskSpace = fs::space(options.dataLocation.toLocal8Bit().constData());
    } else {
        diskSpace = fs::space("/");
    }
    // Note:
    // diskSpace.free is the total amount not being used.
    // diskSpace.available is the amount not being used that is available to non-root processes.
    // We are using the "avaliable" space, which may be the same as "free" for an extra non-OS disk.

    //emit statusMessage(QString("Capacity: %1, available: %2 free: %3").arg(diskSpace.capacity).arg(diskSpace.available).arg(diskSpace.free));
    emit sendDiskSpaceAvailable((quint64)diskSpace.capacity, (quint64)diskSpace.available);

    int percent = (100.0 * (diskSpace.capacity - diskSpace.available) / diskSpace.capacity);

    if(havePrefs)
    {
        if(diskLED != NULL) {
            if(percent > prefs.percentDiskStop)
            {
                diskLED->setState(QLedLabel::StateError);
                stickyDiskFull = true;
                //emit statusMessage(QString("[Flight Widget]: ERROR: Disk too full to use at percent %1").arg(percent));
            } else if (percent > prefs.percentDiskWarning)
            {
                diskLED->setState(QLedLabel::StateWarning);
                //emit statusMessage(QString("[Flight Widget]: Warning: Disk quite full at percent %1").arg(percent));
            } else {
                diskLED->setState(QLedLabel::StateOk);
            }
        }
    }

}

void flight_widget::processFPSError()
{
    if(FPSErrorCounter % 20 == 0)
    {
        emit statusMessage(QString("FPS Error, FPS: %1, Error Count: %2").\
                           arg(fw->delta).\
                           arg(FPSErrorCounter));
    }
    FPSErrorCounter++;


    if((FPSErrorCounter > 0) && !stickyFPSError)
    {
        if(cameraLinkLED != NULL) {
            this->cameraLinkLED->setState(QLedLabel::StateError);
        }
        stickyFPSError = true;
    }
}

void flight_widget::resetFPSError()
{
    if(cameraLinkLED != NULL) {
        this->cameraLinkLED->setState(QLedLabel::StateOk);
    }
    stickyFPSError = false;
}

void flight_widget::handleNewColorScheme(int scheme, bool useDarkThemeVal)
{
    // It should be ok to call these directly:
    //waterfall_widget->handleNewColorScheme(scheme);
    //dsf_widget->handleNewColorScheme(scheme, useDarkThemeVal);
    gps->setPlotTheme(useDarkThemeVal);
}

void flight_widget::handlePrefs(settingsT prefs)
{
    this->prefs = prefs;
    havePrefs = true;
    emit statusMessage("[Flight Widget]: Have preferences inside flight_widget.");
}

void flight_widget::colorMapScrolledX(const QCPRange &newRange)
{

}

void flight_widget::colorMapScrolledY(const QCPRange &newRange)
{

}

void flight_widget::setScrollX(bool Yenabled)
{

}

void flight_widget::setScrollY(bool Xenabled)
{

}

void flight_widget::updateCeiling(int c)
{
    //waterfall_widget->updateCeiling(c);
    emit updateCeilingSignal(c);
    dsf_widget->updateCeiling(c);
}

void flight_widget::updateFloor(int f)
{
    //waterfall_widget->updateFloor(f);
    emit updateFloorSignal(f);
    dsf_widget->updateFloor(f);
}

void flight_widget::rescaleRange()
{
    //waterfall_widget->rescaleRange();
    dsf_widget->rescaleRange();
}

void flight_widget::changeRGB(int r, int g, int b)
{
    //waterfall_widget->changeRGB(r,g,b);
    emit updateRGBbandSignal(r,g,b);
    dsf_widget->showRGB(r,g,b);
    hideRGBTimer.start();
    //emit statusMessage(QString("Updated RGB lines: r:%1, g:%2, b:%3").arg(r).arg(g).arg(b));
}

void flight_widget::setRGBLevels(double r, double g, double b, double gamma)
{
    //waterfall_widget->setRGBLevels(r, g, b, gamma);
    emit setRGBLevelsSignal(r,g,b,gamma);
    //emit statusMessage(QString("Updated RGB levels: r:%1, g:%2, b:%3").arg(r).arg(g).arg(b));
}

void flight_widget::setShowRGBLines(bool showLines)
{
    //emit statusMessage(QString("Showline status: %1").arg(showLines));
    hideRGBTimer.stop();
    showRGBp = showLines;
    dsf_widget->toggleDrawRGBRow(showLines);
}

void flight_widget::changeWFLength(int length)
{
    emit changeWFLengthSignal(length);
    //waterfall_widget->changeWFLength(length);
}

void flight_widget::setCrosshairs(QMouseEvent *event)
{
    dsf_widget->setCrosshairs(event);
}

void flight_widget::startDataCollection(QString secondaryLogFilename)
{
    emit statusMessage(QString("[Flight Widget]: User pressed START Recording button"));
    // Example filename:
    // /tmp/flighttest/AV320230719t191438_gps
    if(options.flightMode)
    {
        QString hhmm = secondaryLogFilename.mid(secondaryLogFilename.length()-10, 4);
        hhmm.insert(2, ':');
        fi->updateLastRec(hhmm);
    } else {
        // Can't rely on these filenames for non-flight recordings.
        fi->updateLastRec();
    }
    if(!options.disableGPS)
        emit beginSecondaryLog(secondaryLogFilename);
}

void flight_widget::stopDataCollection()
{
    emit statusMessage(QString("[Flight Widget]: User pressed STOP Recording button"));
    fi->doneRecording();
    if(!options.disableGPS)
        emit stopSecondaryLog();
}

void flight_widget::startGPS(QString gpsHostname, uint16_t gpsPort, QString primaryLogLocation)
{
    this->gpsHostname = gpsHostname;
    this->primaryGPSLogLocation = primaryLogLocation;
    this->gpsPort = gpsPort;

    if(!startedPrimaryGPSLog)
    {
        gps->initiateGPSConnection(gpsHostname, gpsPort, primaryLogLocation);
        startedPrimaryGPSLog = true;

        emit statusMessage(QString("[GPS]: Connecting to GPS host %1:%2 with primary log location %3")\
                       .arg(gpsHostname)\
                       .arg(gpsPort)\
                       .arg(primaryLogLocation));
    } else {
        emit statusMessage(QString("[GPS]: Error, asked to connect to GPS twice."));
    }
}

void flight_widget::handleGPSConnectionError(int errorNum)
{
    // This usually means we could not connect to the GPS
    // The error string is already handled.
    gpsLED.setState(QLedLabel::StateError);
    (void)errorNum;
}

void flight_widget::handleGPSStatusMessages(QStringList errorMessages, QStringList warningMessages)
{
    // New messages from the gps manager.
    // Copy in the new ones and update the big message thing.
    QMutexLocker locker(&gpsMessageMutex);
    bool update = false;

    int newErrorMessagesSize = errorMessages.size();
    int newWarningMessagesSize = warningMessages.size();
    int currentErrorMessageSize = priorGPSErrorMessages.size();
    int currentWarningMessageSize = priorGPSWarningMessages.size();

    if( (newErrorMessagesSize+newWarningMessagesSize==0) &&
            (!recentlyClearedErrors) &&
            (currentErrorMessageSize+currentWarningMessageSize!=0) ) {
        // The new messages are empty,
        // But there are some old messages.
        return;
    }

    if(priorGPSErrorMessages != errorMessages)
    {
        priorGPSErrorMessages = errorMessages;
        update = true;
    }

    if(priorGPSWarningMessages != warningMessages)
    {
        priorGPSWarningMessages = warningMessages;
        update = true;
    }

    if(update) {
        recentlyClearedErrors = false; // reset this flag
        totalGPSStatusMessages.clear(); // possibly clear the entire thing
        totalGPSStatusMessages << errorMessages;
        totalGPSStatusMessages << warningMessages;
        messageIndex = 0;
    }

}

void flight_widget::cycleGPSStatusMessagesViaTimer()
{
    QMutexLocker locker(&gpsMessageMutex);
    QString messageStr;
    int size = totalGPSStatusMessages.size();
    if(size) {
        messageStr = totalGPSStatusMessages.at(messageIndex%size);
        messageIndex++;
    } else {
        messageStr = "None";
    }
    emit haveGPSErrorWarningMessage(messageStr);
}

void flight_widget::gpsMessageToLogReporterSlot()
{
    QMutexLocker locker(&gpsMessageMutex);

    QString messageLogWarnings = QString("GPS Warnings: ");
    QString messageLogErrors = QString("GPS Errors: ");

    int warSize = priorGPSWarningMessages.size();
    int erSize = priorGPSErrorMessages.size();

    for(int i=0; i < warSize; i++) {
        messageLogWarnings.append(priorGPSWarningMessages.at(i));
        if(i<warSize-1)
            messageLogWarnings.append(", ");
    }

    for(int i=0; i < erSize; i++) {
        messageLogErrors.append(priorGPSErrorMessages.at(i));
        if(i<erSize-1)
            messageLogErrors.append(", ");
    }

    if(warSize!=0)
        emit statusMessage(messageLogWarnings);
    if(erSize!=0)
        emit statusMessage(messageLogErrors);
}

void flight_widget::clearStickyErrors()
{
    stickyDiskFull = false;
    if(diskLED != NULL) {
        diskLED->setState(QLedLabel::StateOk);
    }

    gpsMessageToLogReporterSlot();

    QMutexLocker locker(&gpsMessageMutex);
    priorGPSErrorMessages.clear();
    priorGPSWarningMessages.clear();
    totalGPSStatusMessages.clear();
    recentlyClearedErrors = true;
    emit statusMessage("[Flight Widget]: User cleared sticky errors.");
}

void flight_widget::updateLabel(QLabel *label, QString text)
{
    if(label != NULL)
    {
        label->setText(text);
    }
}

void flight_widget::showDebugMessage(QString debugMessage)
{
    // This is the location to log and/or display debug messages
    // related to flight operations, including GPS and data recording.
    //std::cout << "DEBUG MESSAGE IN FLIGHT WIDGET: " << debugMessage.toLocal8Bit().toStdString() << std::endl;
    emit statusMessage("[Flight Widget]: " + debugMessage);
}

void flight_widget::debugThis()
{
    qDebug() << "in debug function using qDebug()";
    emit statusMessage("Debug function inside flight widget pressed.");
    qDebug() << "Current GPS warnings: " << priorGPSWarningMessages;
    qDebug() << "Current GPS errors: " << priorGPSErrorMessages;
}
