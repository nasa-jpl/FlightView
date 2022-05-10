#include "flight_widget.h"

flight_widget::flight_widget(frameWorker *fw, startupOptionsType options, QWidget *parent) : QWidget(parent)
{
    //connect(this, SIGNAL(statusMessage(QString)), this, SLOT(showDebugMessage(QString)));

    emit statusMessage(QString("Starting flight screen widget"));
    stickyFPSError = false;
    FPSErrorCounter = 0;
    this->fw = fw;
    this->options = options;
    useAvionicsWidgets = true;

    waterfall_widget = new waterfall(fw, 1, 1024, this);
    dsf_widget = new frameview_widget(fw, DSF, this);

    startedPrimaryGPSLog = false;
    gps = new gpsManager();

    if(useAvionicsWidgets)
    {
        // NOTE: If a widget is set to NULL, it will not
        // be updated and will not be added to the layout.
        // Simply comment out the widgets not desired.

        // See the "Avionics Widget Layout Placement" section
        // for layout issues.

        // QFlightInstruments Avionics Widgets are by Marek Cel:
        // http://marekcel.pl/qflightinstruments
        // https://github.com/marek-cel/QFlightinstruments

        int avBase = 100;
        int avMax = 150; // Larger values may require a window resize

        // EADI: Electronic Attitude Direction Indicator
        // shows speed, climb, heading, and ground orientation
        EADI = new qfi_EADI();
        EADI->setBaseSize(avBase,avBase);
        EADI->setMaximumSize(avMax,avMax);
        //EADI->setMinimumSize(200,200);
        EADI->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        EADI->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        EADI->setInteractive(false);
        EADI->setEnabled(false);

        // EHSI: Electronic Horizontal Situation Indicator
        // shows heading and course
        EHSI = new qfi_EHSI();
        EHSI->setBaseSize(avBase,avBase);
        EHSI->setMaximumSize(avMax,avMax);
        EHSI->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        EHSI->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        EHSI->setInteractive(false);
        EHSI->setEnabled(false);

        // ASI: Air Speed Indicator
        // air speed from GPS unit of course
        ASI = new qfi_ASI();
        ASI->setBaseSize(avBase,avBase);
        ASI->setMaximumSize(avMax,avMax);
        ASI->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        ASI->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        ASI->setInteractive(false);
        ASI->setEnabled(false);

        // VSI: Vertical Speed Indicator
        // Shows rate of climb
        VSI = new qfi_VSI();
        VSI->setBaseSize(avBase,avBase);
        VSI->setMaximumSize(avMax,avMax);
        VSI->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        VSI->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        VSI->setInteractive(false);
        VSI->setEnabled(false);
    }

    // Group Box "Flight Instrument Controls" items:
    resetStickyErrorsBtn.setText("Clear Errors");
    resetStickyErrorsBtn.setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    aircraftLbl.setText("AVIRIS-III");
    gpsLatText.setText("GPS Latitude:");
    gpsLatData.setText("########");
    gpsAltitudeText.setText("GPS Altitude:");
    gpsAltitudeData.setText("########");
    gpsLongText.setText("GPS Longitude:");
    gpsLongData.setText("########");
    gpsLEDLabel.setText("GPS Status:");
    gpsHeadingText.setText("Heading:");
    gpsHeadingData.setText("###.###");

    gpsUTCtimeText.setText("UTC Time:");
    gpsUTCtimeData.setText("###TIME##");
    gpsUTCdateText.setText("UTC Date:");
    gpsUTCdateData.setText("########");
    gpsUTCValidityText.setText("UTC Validity:");
    gpsUTCValidityData.setText("##VAL TIME##");

    gpsGroundSpeedText.setText("Ground Speed:");
    gpsGroundSpeedData.setText("########");
    gpsQualityText.setText("GPS Quality:");
    gpsQualityData.setText("########");


    gpsLED.setState(QLedLabel::StateOkBlue);
    cameraLinkLEDLabel.setText("CameraLink Status:");
    cameraLinkLED.setState(QLedLabel::StateOk);
    diskLEDLabel.setText("Disk:");
    diskLED.setState(QLedLabel::StateOkBlue);

    // Format is &item, row, col, rowSpan, colSpan. -1 = to "edge"
    int row=0;
    //flightPlotsLayout.addWidget(&gpsPitchRollPlot, 0,0,4,8);
    flightControlLayout.addWidget(&gpsPitchRollPlot, row,0,4,8);

    row += 4;

    // First row of widgets:
    ++row;
    flightControlLayout.addWidget(&gpsLEDLabel,   row,0,1,1);
    flightControlLayout.addWidget(&gpsLED,        row,1,1,1);
    flightControlLayout.addWidget(&diskLEDLabel,  row,2,1,1);
    flightControlLayout.addWidget(&diskLED,       row,3,1,1, Qt::AlignLeft);

    // Second row:
    row++;
    flightControlLayout.addWidget(&cameraLinkLEDLabel,  row,0,1,1);
    flightControlLayout.addWidget(&cameraLinkLED,       row,1,1,1);
    flightControlLayout.addWidget(&resetStickyErrorsBtn,row,2,1,1);

    // Third row:
    row++;
    flightControlLayout.addWidget(&gpsLatText,  row,0,1,1);
    flightControlLayout.addWidget(&gpsLatData,  row,1,1,1);
    flightControlLayout.addWidget(&gpsLongText, row,2,1,1);
    flightControlLayout.addWidget(&gpsLongData, row,3,1,1);

    // Fourth row:
    row++;
    flightControlLayout.addWidget(&gpsAltitudeText,    row,0,1,1);
    flightControlLayout.addWidget(&gpsAltitudeData,    row,1,1,1);
    flightControlLayout.addWidget(&gpsGroundSpeedText, row,2,1,1);
    flightControlLayout.addWidget(&gpsGroundSpeedData, row,3,1,1);

    // Fifth row:
    row++;
    flightControlLayout.addWidget(&gpsHeadingText,     row,0,1,1);
    flightControlLayout.addWidget(&gpsHeadingData,     row,1,1,1);
    flightControlLayout.addWidget(&gpsUTCValidityText, row,2,1,1);
    flightControlLayout.addWidget(&gpsUTCValidityData, row,3,1,1);

    // Sixth row:
    row++;
    flightControlLayout.addWidget(&gpsUTCdateText, row,0,1,1);
    flightControlLayout.addWidget(&gpsUTCdateData, row,1,1,1);
    flightControlLayout.addWidget(&gpsUTCtimeText, row,2,1,1);
    flightControlLayout.addWidget(&gpsUTCtimeData, row,3,1,1);

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

    flightControlLayout.setColumnStretch(0,0);
    flightControlLayout.setColumnStretch(1,0);
    flightControlLayout.setColumnStretch(2,0);
    flightControlLayout.setColumnStretch(3,1); // this is between the text and the avionics widgets

    flightControlLayout.setColumnStretch(4,0);
    flightControlLayout.setColumnStretch(5,0);
    flightControlLayout.setColumnStretch(6,0);
    flightControlLayout.setColumnStretch(7,0);

    flightControlLayout.setRowStretch(3,2); // stretch the plot area

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
    connect(gps, SIGNAL(gpsConnectionError(int)), this, SLOT(handleGPSConnectionError(int)));
    gps->insertLEDs(&gpsLED);
    gps->insertLabels(&gpsLatData, &gpsLongData, &gpsAltitudeData,
                      &gpsUTCtimeData, &gpsUTCdateData, &gpsUTCValidityData,
                      &gpsGroundSpeedData,
                      &gpsHeadingData, NULL, NULL,
                      &gpsQualityData,
                      NULL);
    gps->insertPlots(&gpsPitchRollPlot);
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



    connect(&resetStickyErrorsBtn, SIGNAL(clicked(bool)), gps, SLOT(clearStickyError()));
    connect(&resetStickyErrorsBtn, SIGNAL(clicked(bool)), this, SLOT(resetFPSError()));

    connect(&resetStickyErrorsBtn, SIGNAL(clicked(bool)), this, SLOT(clearStickyErrors()));

    diskCheckerTimer = new QTimer();
    diskCheckerTimer->setInterval(1000);
    diskCheckerTimer->setSingleShot(false);
    connect(diskCheckerTimer, SIGNAL(timeout()), this, SLOT(checkDiskSpace()));
    diskCheckerTimer->start();

    hideRGBTimer.setInterval(30000);
    hideRGBTimer.stop();
    connect(&hideRGBTimer, SIGNAL(timeout()), this, SLOT(hideRGB()));


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

void flight_widget::hideRGB()
{
    hideRGBTimer.stop();
    dsf_widget->toggleDrawRGBRow(false);
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
            //emit statusMessage(QString("[Flight Widget]: ERROR: Disk too full to use at percent %1").arg(percent));
        } else if (percent > prefs.percentDiskWarning)
        {
            diskLED.setState(QLedLabel::StateWarning);
            //emit statusMessage(QString("[Flight Widget]: Warning: Disk quite full at percent %1").arg(percent));
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
    dsf_widget->showRGB(r,g,b);
    hideRGBTimer.start();
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
    emit statusMessage(QString("[Flight Widget]: User pressed START Recording button"));
    if(!options.disableGPS)
        emit beginSecondaryLog(secondaryLogFilename);
}

void flight_widget::stopDataCollection()
{
    emit statusMessage(QString("[Flight Widget]: User pressed STOP Recording button"));
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

void flight_widget::clearStickyErrors()
{
    stickyDiskFull = false;
    diskLED.setState(QLedLabel::StateOk);
    emit statusMessage("[Flight Widget]: User cleared sticky errors.");
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
    //gps->initiateGPSConnection("10.0.0.6", 8111, "");
    //waterfall_widget->debugThis();
    waterfall_widget->handleNewFrame();
}
