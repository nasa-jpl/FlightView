#include "flight_widget.h"

flight_widget::flight_widget(frameWorker *fw, startupOptionsType options, QWidget *parent) : QWidget(parent)
{
    connect(this, SIGNAL(statusMessage(QString)), this, SLOT(showDebugMessage(QString)));

    emit statusMessage(QString("Starting flight screen widget"));
    stickyFPSError = false;
    FPSErrorCounter = 0;
    this->fw = fw;
    this->options = options;

    waterfall_widget = new waterfall(fw, 1, 1024, this);
    dsf_widget = new frameview_widget(fw, DSF, this);

    startedPrimaryGPSLog = false;
    gps = new gpsManager();

    // Group Box "Flight Instrument Controls" items:
    resetStickyErrorsBtn.setText("Clear Errors");
    instBtn.setText("Init");
    aircraftLbl.setText("AVIRIS-III");
    gpsLatText.setText("GPS Latitude: ");
    gpsLatData.setText("########");
    gpsLongText.setText("GPS Longitude: ");
    gpsLongData.setText("########");
    gpsLEDLabel.setText("GPS Status: ");
    gpsHeadingText.setText("Heading: ");
    gpsHeadingData.setText("###.###");
    gpsLED.setState(QLedLabel::StateOkBlue);
    cameraLinkLEDLabel.setText("CameraLink Status:");
    cameraLinkLED.setState(QLedLabel::StateOk);
    diskLEDLabel.setText("Disk:");
    diskLED.setState(QLedLabel::StateOkBlue);

    // Format is &item, row, col, rowSpan, colSpan. -1 = to "edge"
    int row=0;

    flightControlLayout.addWidget(&gpsPitchRollPlot, row,0,4,8);

    row += 4;

    flightControlLayout.addWidget(&resetStickyErrorsBtn, ++row,0,1,1);
    flightControlLayout.addWidget(&instBtn, row,1,1,1);

    flightControlLayout.addWidget(&gpsLEDLabel, ++row,0,1,1);
    flightControlLayout.addWidget(&gpsLED, row,1,1,1);
    flightControlLayout.addWidget(&diskLEDLabel, row, 2, 1, 1);
    flightControlLayout.addWidget(&diskLED, row, 3, 1, 1, Qt::AlignLeft);

    flightControlLayout.addWidget(&cameraLinkLEDLabel, ++row,0,1,1);
    flightControlLayout.addWidget(&cameraLinkLED, row,1,1,1);

    flightControlLayout.addWidget(&gpsLatText, ++row,0,1,1);
    flightControlLayout.addWidget(&gpsLatData, row,1,1,1);

    flightControlLayout.addWidget(&gpsLongText, ++row,0,1,1);
    flightControlLayout.addWidget(&gpsLongData, row,1,1,1);

    flightControlLayout.addWidget(&gpsHeadingText, ++row,0,1,1);
    flightControlLayout.addWidget(&gpsHeadingData, row,1,1,1);

    flightControlLayout.setColumnStretch(3,2);
    flightControlLayout.setRowStretch(3,2);



    // Group Box "Flight Instrument Controls"
    flightControls.setTitle("Flight Instrument Controls");
    flightControls.setLayout(&flightControlLayout);

    rhSplitter.setOrientation(Qt::Vertical);
    rhSplitter.addWidget(dsf_widget);
    rhSplitter.addWidget(&flightControls);



    lrSplitter.addWidget(waterfall_widget);
    lrSplitter.addWidget(&rhSplitter);

    lrSplitter.setHandleWidth(5);

    layout.addWidget(&lrSplitter);

    this->setLayout(&layout);

    connect(waterfall_widget, SIGNAL(statusMessageOut(QString)), this, SLOT(showDebugMessage(QString)));


    // Connections to GPS:
    connect(gps, SIGNAL(gpsStatusMessage(QString)), this, SLOT(showDebugMessage(QString)));
    gps->insertLEDs(&gpsLED);
    gps->insertLabels(&gpsLatData, &gpsLongData,
                      &gpsAltitudeData, NULL,
                      NULL, NULL, NULL,
                      &gpsHeadingData, NULL,
                      NULL, NULL, NULL);
    gps->insertPlots(&gpsPitchRollPlot);
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
            emit statusMessage(QString("Error, cannot start flight mode with incomplete GPS settings."));
        }
    } else {
        emit statusMessage(QString("Starting liveview in LAB mode."));
    }

    // TODO:
    // set base filename for master (primary) log

    // connect start and stop signals for secondary log
    connect(this, &flight_widget::beginSecondaryLog, gps, &gpsManager::handleStartsecondaryLog);
    connect(this, &flight_widget::stopSecondaryLog, gps, &gpsManager::handleStopSecondaryLog);



    connect(&resetStickyErrorsBtn, SIGNAL(clicked(bool)), gps, SLOT(clearStickyError()));
    connect(&resetStickyErrorsBtn, SIGNAL(clicked(bool)), this, SLOT(resetFPSError()));

    connect(&resetStickyErrorsBtn, SIGNAL(clicked(bool)), this, SLOT(clearStickyErrors()));

    diskCheckerTimer = new QTimer();
    diskCheckerTimer->setInterval(1000);
    diskCheckerTimer->setSingleShot(false);
    connect(diskCheckerTimer, SIGNAL(timeout()), this, SLOT(checkDiskSpace()));
    diskCheckerTimer->start();

    emit statusMessage(QString("Finished flight constructor."));
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

void flight_widget::handleNewFrame()
{
    dsf_widget->handleNewFrame();
    waterfall_widget->handleNewFrame();
}

void flight_widget::updateFPS()
{
    if(fw->delta < 12.8f)
    {
        processFPSError();
    } else if (fw->delta < 13.0f) {
        this->cameraLinkLED.setState(QLedLabel::StateWarning);
    } else if ((fw->delta > 13.0f) && !stickyFPSError)
    {
        // to reset the warning, but not the sticky error:
        this->cameraLinkLED.setState(QLedLabel::StateOk);
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
        if(percent > prefs.percentDiskStop)
        {
            diskLED.setState(QLedLabel::StateError);
            stickyDiskFull = true;
        } else if (percent > prefs.percentDiskWarning)
        {
            diskLED.setState(QLedLabel::StateWarning);
        } else {
            diskLED.setState(QLedLabel::StateOk);
        }
    }

}

void flight_widget::processFPSError()
{
    FPSErrorCounter++;
    emit statusMessage(QString("FPS Error, FPS: %1, Error Count: %2").\
                       arg(fw->delta).\
                       arg(FPSErrorCounter));
    if((FPSErrorCounter > 0) && !stickyFPSError)
    {
        this->cameraLinkLED.setState(QLedLabel::StateError);
        stickyFPSError = true;
    }
}

void flight_widget::resetFPSError()
{
    this->cameraLinkLED.setState(QLedLabel::StateOk);
    stickyFPSError = false;
    FPSErrorCounter = 0;
}

void flight_widget::handleNewColorScheme(int scheme)
{
    // It should be ok to call these directly:
    //waterfall_widget->handleNewColorScheme(scheme);
    dsf_widget->handleNewColorScheme(scheme);
}

void flight_widget::handlePrefs(settingsT prefs)
{
    this->prefs = prefs;
    havePrefs = true;
    emit statusMessage("Have preferences inside flight_widget.");
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
    waterfall_widget->updateCeiling(c);
    dsf_widget->updateCeiling(c);
}

void flight_widget::updateFloor(int f)
{
    waterfall_widget->updateFloor(f);
    dsf_widget->updateFloor(f);
}

void flight_widget::rescaleRange()
{
    //waterfall_widget->rescaleRange();
    dsf_widget->rescaleRange();
}

void flight_widget::changeRGB(int r, int g, int b)
{
    waterfall_widget->changeRGB(r,g,b);
}

void flight_widget::changeWFLength(int length)
{
    waterfall_widget->changeWFLength(length);
}

void flight_widget::setCrosshairs(QMouseEvent *event)
{
    dsf_widget->setCrosshairs(event);
}

void flight_widget::startDataCollection(QString secondaryLogFilename)
{
    emit statusMessage(QString("User pressed START Recording button"));
    emit beginSecondaryLog(secondaryLogFilename);
}

void flight_widget::stopDataCollection()
{
    emit statusMessage(QString("User pressed STOP Recording button"));
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

        emit statusMessage(QString("Connecting to GPS host %1:%2 with primary log location %3")\
                       .arg(gpsHostname)\
                       .arg(gpsPort)\
                       .arg(primaryLogLocation));
    } else {
        emit statusMessage(QString("Error, asked to connect to GPS twice."));
    }
}

void flight_widget::clearStickyErrors()
{
    stickyDiskFull = false;
    diskLED.setState(QLedLabel::StateOk);
}

void flight_widget::showDebugMessage(QString debugMessage)
{
    // This is the location to log and/or display debug messages
    // related to flight operations, including GPS and data recording.
    std::cout << "DEBUG MESSAGE IN FLIGHT WIDGET: " << debugMessage.toLocal8Bit().toStdString() << std::endl;
}

void flight_widget::debugThis()
{
    qDebug() << "in debug function using qDebug()";
    emit statusMessage("Debug function inside flight widget pressed.");
    //gps->initiateGPSConnection("10.0.0.6", 8111, "");
    //waterfall_widget->debugThis();
    waterfall_widget->handleNewFrame();
}
