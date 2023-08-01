#include "gpsmanager.h"

gpsManager::gpsManager()
{
    qRegisterMetaType<gpsMessage>();
    statusLinkStickyError = false;

    // May override later:
    baseSaveDirectory = QString("/tmp/gps");
    //createLoggingDirectory();
    filenamegen.setMainDirectory(baseSaveDirectory);
    filenamegen.setFilenameExtension("log");

    prepareVectors(); // size the vectors
    prepareGPS(); // get ready to connect the GPS

}

gpsManager::~gpsManager()
{
    if(gpsThread != NULL)
    {
        gpsThread->quit();
        gpsThread->wait();
        //delete gpsThread;
    }
    //delete gps;
}

void gpsManager::initiateGPSConnection(QString host = "192.168.2.101", int port=(int)8112, QString gpsBinaryLogFilename = "")
{
    if (gpsBinaryLogFilename.isEmpty())
    {
        createLoggingDirectory();
        gpsBinaryLogFilename=filenamegen.getNewFullFilename(this->baseSaveDirectory, QString(""), QString("-gpsPrimary"), QString("bin"));
    } else {
        baseSaveDirectory = gpsBinaryLogFilename;
        createLoggingDirectory();
        gpsBinaryLogFilename = filenamegen.getNewFullFilename(gpsBinaryLogFilename, QString(""), QString("-gpsPrimary"), QString("bin"));
    }
    this->host = host;
    this->port = port;
    this->gpsBinaryLogFilename = gpsBinaryLogFilename;
    emit connectToGPS(host, port, gpsBinaryLogFilename);
    gpsReconnectTimer.setInterval(1000);
    gpsReconnectTimer.setSingleShot(true);
    connect(&gpsReconnectTimer, SIGNAL(timeout()), this, SLOT(handleGPSReconnectTimer()));
    gnssStatusTime.restart();
}

void gpsManager::initiateGPSDisconnect()
{
    emit disconnectFromGPS();
    firstMessage = true;
}

bool gpsManager::createLoggingDirectory()
{
    if(baseSaveDirectory.isEmpty())
    {
        emit gpsStatusMessage(QString("ERROR! GPS Base Directory requested for primary log is blank!"));
        return false;
    }
    QDir dir(baseSaveDirectory);
    if (!dir.exists())
        dir.mkpath(".");

    if(!dir.exists())
    {
        emit gpsStatusMessage(QString("ERROR! Could not create gps primary log directory %1!").arg(baseSaveDirectory));
    }
    return dir.exists();
}

void gpsManager::prepareElements()
{
    // This function prepares the available optional elements
    // such as LEDs, Labels, and Plots:
    preparePlots();
    prepareLEDs();
    prepareLabels();
}

void gpsManager::prepareVectors()
{
    // Resize
    vecSize = 450;

    headingsMagnetic.resize(vecSize);
    headingsCourse.resize(vecSize);
    rolls.resize(vecSize);
    pitches.resize(vecSize);
    lats.resize(vecSize);
    longs.resize(vecSize);
    alts.resize(vecSize);
    nVelos.resize(vecSize);
    eVelos.resize(vecSize);
    upVelos.resize(vecSize);
    timeAxis.resize(vecSize);

}

void gpsManager::prepareLEDs()
{
    // Initial state
    if(gpsLinkLED != NULL)gpsLinkLED->setText("");
    if(gpsLinkLED != NULL)gpsLinkLED->setState(QLedLabel::StateOkBlue);
}

void gpsManager::prepareLabels()
{
    // This function is currently not needed but may be populated in the future.
}

void gpsManager::prepareGPS()
{
    // Threads and connections
    msgsReceivedCount = 0;
    firstMessage = true;
    gps = new gpsNetwork();
    gpsThread = new QThread(this);

    gps->moveToThread(gpsThread);

    connect(gpsThread, &QThread::finished, gps, &QObject::deleteLater);

    connect(this, SIGNAL(connectToGPS(QString,int,QString)), gps, SLOT(connectToGPS(QString,int,QString)));
    connect(this, SIGNAL(disconnectFromGPS()), gps, SLOT(disconnectFromGPS()));
    connect(this, SIGNAL(getDebugInfo()), gps, SLOT(debugThis()));
    connect(gps, SIGNAL(haveGPSString(QString)), this, SLOT(handleGPSDataString(QString)));
    connect(gps, SIGNAL(statusMessage(QString)), this, SLOT(handleGPSStatusMessage(QString)));
    connect(gps, SIGNAL(connectionError(int)), this, SLOT(handleGPSConnectionError(int)));
    connect(gps, SIGNAL(connectionGood()), this, SLOT(handleGPSConnectionGood()));

    connect(gps, SIGNAL(haveGPSMessage(gpsMessage)), this, SLOT(receiveGPSMessage(gpsMessage)));

    connect(this, SIGNAL(startSecondaryLog(QString)), gps, SLOT(beginSecondaryBinaryLog(QString)));
    connect(this, SIGNAL(stopSecondaryLog()), gps, SLOT(stopSecondaryBinaryLog()));

    gpsThread->setObjectName(name + "gps");
    gpsThread->start();

    gpsMessageHeartbeat.setInterval(500); // half second, expected is 5ms.
    connect(&gpsMessageHeartbeat, SIGNAL(timeout()), this, SLOT(handleGPSTimeout()));
}

void gpsManager::preparePlots()
{
    bool ok = false;
    // Time axis:
    uint16_t t=0;
    for(int i=vecSize; i > 0; i--)
    {
        timeAxis[i-1] = t++;
    }

    if(plotRollPitch != NULL)
    {
        plotRollPitch->addGraph();
        plotRollPitch->yAxis->setRange(-10, 10);
        plotRollPitch->addGraph(plotRollPitch->xAxis, plotRollPitch->yAxis );
        plotRollPitch->addGraph(plotRollPitch->xAxis, plotRollPitch->yAxis );
        setTimeAxis(plotRollPitch->xAxis);
        plotRollPitch->yAxis->setLabel("Degrees");
        if(titleRollPitch==NULL)
        {
            titleRollPitch = new QCPPlotTitle(plotRollPitch);
        }
        setPlotTitle(plotRollPitch, titleRollPitch, "Pitch and Roll");
        if(usePlotTitle)
        {
            ok = plotRollPitch->plotLayout()->addElement(1,1,titleRollPitch);
            if(!ok){
                handleGPSStatusMessage("Error, could not add element to gps plot.");
                qDebug() << "Error, could not add element to gps plot.";
            }
        }
        plotRollPitch->graph(0)->setPen(QPen(Qt::red)); // roll
        plotRollPitch->graph(1)->setPen(QPen(Qt::green)); // pitch
        plotRollPitch->graph(0)->setName("Roll");
        plotRollPitch->graph(1)->setName("Pitch");
        plotRollPitch->removeGraph(2);
        plotRollPitch->legend->setVisible(true);
        setPlotColors(plotRollPitch, true);
    }

    if(plotHeading != NULL)
    {


        plotHeading->addGraph(); // magnetic
        plotHeading->addGraph(0,0); // course
        plotHeading->graph(0)->setName("Magnetic");
        plotHeading->graph(1)->setName("Course");
        plotHeading->yAxis->setRange(0, 360*1.1); // raw data is 0-360 but we will show it like this.
        setTimeAxis(plotHeading->xAxis);
        if(titleHeading == NULL)
        {
            titleHeading = new QCPPlotTitle(plotHeading);
        }
        setPlotTitle(plotHeading, titleHeading, "Pitch and Roll");

        plotHeading->graph(0)->setPen(QPen(Qt::yellow));
        plotHeading->graph(1)->setPen(QPen(Qt::red));
        plotHeading->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignTop);
        setPlotColors(plotHeading, true);

        //plotHeading->legend->setBrush(QColor("#85ffffff")); // 85% opaque white
        plotHeading->legend->setVisible(true);
    }
    updatePlots(); // initial draw
}

void gpsManager::updateUIelements()
{
    // new messages ultimately update the UI through this.

}

void gpsManager::updatePlots()
{
    if(plotRollPitch != NULL)
    {
        this->plotRollPitch->graph(0)->setData(timeAxis, rolls);
        this->plotRollPitch->graph(1)->setData(timeAxis, pitches);
        this->plotRollPitch->replot();
    }

    if(plotHeading != NULL)
    {
        this->plotHeading->graph(0)->setData(timeAxis, headingsMagnetic);
        this->plotHeading->graph(1)->setData(timeAxis, headingsCourse);
        this->plotHeading->replot();
    }
}

void gpsManager::setTimeAxis(QCPAxis *x)
{ 
    if(x != NULL)
    {
        x->setRange(0, vecSize);
        x->setLabel("Time (relative units)");
    }
}

void gpsManager::setPlotTitle(QCustomPlot *p, QCPPlotTitle *t, QString title)
{
    // Does not seem to quite work, oh well. Fix as you go.
    if(p!=NULL)
    {
        if(t==NULL)
        {
            t = new QCPPlotTitle(p);
        }
        t->setText(title);
    }
    // The insert to the plot happens later.
}

void gpsManager::setPlotTheme(bool isDark)
{
    setPlotColors(plotRollPitch, isDark);
}

void gpsManager::setPlotColors(QCustomPlot *p, bool dark)
{
    if(p == NULL)
        return;

    if(dark)
    {
        p->setBackground(QBrush(Qt::black));
        p->xAxis->setBasePen(QPen(Qt::green)); // lower line of axis
        p->yAxis->setBasePen(QPen(Qt::green)); // left line of axis
        p->xAxis->setTickPen(QPen(Qt::green));
        p->yAxis->setTickPen(QPen(Qt::green));
        p->xAxis->setLabelColor(QColor(Qt::white));
        p->yAxis->setLabelColor(QColor(Qt::white));
        p->yAxis->setTickLabelColor(Qt::white);
        p->xAxis->setTickLabelColor(Qt::white);
        //p->graph()->setPen(QPen(Qt::red));
        p->graph(0)->setPen(QPen(Qt::yellow));
        p->graph(1)->setPen(QPen(Qt::red));
        p->legend->setBrush(QBrush(Qt::black));
        p->legend->setTextColor(Qt::white);
        //p->graph()->setBrush(QBrush(Qt::yellow)); // sets an underfill for the line
    } else {
        p->setBackground(QBrush(Qt::white));
        p->xAxis->setBasePen(QPen(Qt::black)); // lower line of axis
        p->yAxis->setBasePen(QPen(Qt::black)); // left line of axis
        p->xAxis->setTickPen(QPen(Qt::black));
        p->yAxis->setTickPen(QPen(Qt::black));
        p->xAxis->setLabelColor(QColor(Qt::black));
        p->yAxis->setLabelColor(QColor(Qt::black));
        p->graph(0)->setPen(QPen(Qt::blue));
        p->graph(1)->setPen(QPen(Qt::red));
        p->legend->setBrush(QBrush(Qt::white));
        p->legend->setTextColor(Qt::black);
        p->yAxis->setTickLabelColor(Qt::black);
        p->xAxis->setTickLabelColor(Qt::black);
        //p->graph()->setBrush(QBrush(Qt::black)); // sets an underfill for the line
    }
    p->replot();
}

void gpsManager::showStatusMessage(QString s)
{
    qDebug() << "GPS MANAGER DEBUG: " << s;
    (void)s;
}

void gpsManager::insertLEDs(QLedLabel *gpsLinkLED, QLedLabel *gpsTroubleLED)
{
    this->gpsLinkLED = gpsLinkLED;
    this->gpsTroubleLED = gpsTroubleLED;
}

void gpsManager::insertPlots(QCustomPlot *gpsRollPitchPlot, QCustomPlot *gpsHeadingPlot)
{
    this->plotRollPitch = gpsRollPitchPlot;
    this->plotHeading = gpsHeadingPlot;
}

void gpsManager::insertLabels(QLabel *gpsLat, QLabel *gpsLong, QLabel *gpsAltitude,
                              QLabel *gpsUTCtime, QLabel *gpsUTCdate, QLabel *gpsUTCValid,
                              QLabel *gpsGroundSpeed,
                              QLabel *gpsHeading, QLabel *gpsRoll, QLabel *gpsPitch,
                              QLabel *gpsQuality, QLabel *gpsAlignment,
                              QLabel *gpsRateClimb)
{

    this->gpsLat = gpsLat;
    this->gpsLong = gpsLong;
    this->gpsAltitude = gpsAltitude;
    this->gpsUTCtime = gpsUTCtime;
    this->gpsUTCdate = gpsUTCdate;
    this->gpsUTCValidity = gpsUTCValid;
    this->gpsGroundSpeed = gpsGroundSpeed;
    this->gpsHeading = gpsHeading;
    this->gpsRoll = gpsRoll;
    this->gpsPitch = gpsPitch;
    this->gpsQuality = gpsQuality;
    this->gpsAlignment = gpsAlignment;
    this->gpsRateClimb = gpsRateClimb;
}

void gpsManager::insertAvionicsWidgets(qfi_ASI *asi, qfi_VSI *vsi,
                                       qfi_EADI *eadi, qfi_EHSI *ehsi)
{
    this->asi = asi;
    this->vsi = vsi;
    this->eadi = eadi;
    this->ehsi = ehsi;
}

void gpsManager::updateLabel(QLabel *label, QString text)
{
    if(label != NULL)
    {
        label->setText(text);
    }
}

void gpsManager::updateLED(QLedLabel *led, QLedLabel::State s)
{
    if(led != NULL) {
        led->setState(s);
    }
}

unsigned char gpsManager::getBit(uint32_t d, unsigned char bit)
{
    return((d & ( 1 << bit )) >> bit);
}

void gpsManager::receiveGPSMessage(gpsMessage m)
{
    // Entry point for new messages
    float longitude;

    if(m.validDecode)
    {
        this->m = m;
        statusMessageDecodeOk = true;
        consecutiveDecodeErrors = 0;
    } else {
        statusMessageDecodeOk = false;
        consecutiveDecodeErrors++;
        processStatus();
        return;
    }

    msgsReceivedCount++;

    if(firstMessage)
    {
        gnssStatusTime.start();
    }

    gpsMessageHeartbeat.start();
    statusGPSHeartbeatOk = true;

    if(m.haveGNSSInfo1 || m.haveGNSSInfo2 || m.haveGNSSInfo3)
    {
        gnssStatusTime.restart();
        statusGNSSReceptionOk = true;
        statusGNSSReceptionWarning = false;
    } else {
        if(gnssStatusTime.elapsed() > 5*1000)
        {
            statusGNSSReceptionOk = false;
        } else if (gnssStatusTime.elapsed() > 2*1000)
        {
            statusGNSSReceptionWarning = true;
        }
    }

    if( (m.numberDropped > 0) && (!firstMessage) )
    {
        statusGPSMessagesDropped = true;
    } else {
        statusGPSMessagesDropped = false;
    }

    //bool doPlotUpdate = (msgsReceivedCount%updatePlotsInverval)==0;
    //bool doWidgetPaint = (msgsReceivedCount%updateAvionicsWidgetsInterval)==0;
    bool doPlotUpdate = false; // no more plots
    bool doWidgetPaint = false; // no more fancy widgets
    bool doLabelUpdate = (msgsReceivedCount%updateLabelsInterval)==0;

    if(m.haveUTC)
    {
        // This is only once per second, but we don't want to miss it.
        utcTime UTCdataValidityTime = processUTCstamp(m.UTCdataValidityTime);
        updateLabel(gpsUTCtime, UTCdataValidityTime.UTCstr);
        utcTime navValidity =  processUTCstamp(m.navDataValidityTime);
        updateLabel(gpsUTCValidity, navValidity.UTCValidityStr);
    }

    if(doLabelUpdate)
    {
        if(!m.haveUTC)
        {
            // If we just did this due to m.haveUTC, then no need to do it again.
            // The validity time is always available, there isn't a boolean to check.
            utcTime navValidity =  processUTCstamp(m.navDataValidityTime);
            updateLabel(gpsUTCValidity, navValidity.UTCValidityStr);
        }

        if(m.haveAltitudeHeading)
        {
            updateLabel(gpsRoll, QString("%1").arg(m.roll));
            updateLabel(gpsHeading, QString(" %1").arg(m.heading));
            updateLabel(gpsPitch, QString("%1").arg(m.pitch));

        }
        if(m.havePosition)
        {
            if(m.longitude > 180)
            {
                longitude = -360+m.longitude;
            } else {
                longitude = m.longitude;
            }
            // Format specifier is number, TOTAL DIGITS (including the dot), float, and the number of decimal places desired (8), and the filler character for any front filling
            // Example: -118.94771 57:
            updateLabel(gpsLat, QString("%1").arg(m.latitude, 10, 'f', 5, QChar(' ')));
            updateLabel(gpsLong, QString("%1").arg(longitude, 10, 'f', 5, QChar(' ')));
            // Native altitude is meters
            // Converting to feet by multiplying by 3.28084
            updateLabel(gpsAltitude, QString("%1 ft").arg(m.altitude * 3.28084, 6, 'f', 1, QChar('0')));
        }
        if(m.haveCourseSpeedGroundData)
        {
            // The native units are meters per second.
            // To convert to knots, multiply by 1.94384
            // To convert to MPH, multiply by 2.23694
            // To convert to KPH, multiply by 3.6
            // The abbreviation for knot or knots is "kt" or "kts", respectively.
            updateLabel(gpsGroundSpeed, QString("%1 kts").arg(m.speedOverGround * 1.94384, 6, 'f', 2, QChar('0')));
        }
        if(m.haveSpeedData)
        {
            updateLabel(gpsRateClimb, QString("%1").arg(m.upVelocity * 196.85));
        }
        if(m.haveSystemDateData)
        {
            QString date = QString("%1-%2-%3").arg(m.systemYear).arg(m.systemMonth, 2, 10, QChar('0')).arg(m.systemDay, 2, 10, QChar('0'));
            updateLabel(gpsUTCdate, date);
        }
    }

    if(m.haveINSAlgorithmStatus) {
        if( (firstMessage) || (priorAlgorithmStatus1 !=m.algorithmStatus1) ) {
            bool courseAlignment = false;
            bool fineAlignment = false;
            if(getBit(m.algorithmStatus1, 1)) {
                gnssAlignmentPhase = "COURSE";
                //updateLabel(gpsAlignment, "COURSE");
                courseAlignment = true;
                gnssAlignmentComplete = false;
            }
            if(getBit(m.algorithmStatus1, 2)) {
                //updateLabel(gpsAlignment, "FINE");
                gnssAlignmentPhase = "FINE";
                fineAlignment = true;
                gnssAlignmentComplete = false;
            }
            if( (!courseAlignment) && (!fineAlignment) ) {
                    //updateLabel(gpsAlignment, "COMPLETE");
                    gnssAlignmentPhase = "Done";
                    gnssAlignmentComplete = true;
            }
            if(getBit(m.algorithmStatus1, 0)) {
                navPhase = true;
            } else {
                navPhase = false;
            }
            if(getBit(m.algorithmStatus1, 12)) {
                gpsReceived = true;
            } else {
                gpsReceived = false;
            }
            if(getBit(m.algorithmStatus1, 13)) {
                gpsValid = true;
            } else {
                gpsValid = false;
            }
            if(getBit(m.algorithmStatus1, 14)) {
                gpsWaiting = true;
            } else {
                gpsWaiting = false;
            }
            if(getBit(m.algorithmStatus1, 15)) {
                gpsRejected = true;
            } else {
                gpsRejected = false;
            }
            if(getBit(m.algorithmStatus1, 28)) {
                altitudeSaturation = true;
            } else {
                altitudeSaturation = false;
            }
            if(getBit(m.algorithmStatus1, 29)) {
                speedSaturation = true;
            } else {
                speedSaturation = false;
            }
            if(getBit(m.algorithmStatus1, 30)) {
                interpolationMissed = true;
            } else {
                interpolationMissed = false;
            }
        }
        if( (firstMessage) || (priorAlgorithmStatus2 !=m.algorithmStatus2) ) {
            if(getBit(m.algorithmStatus2, 15)) {
                altitudeRejected = true;
            } else {
                altitudeRejected = false;
            }
            if(getBit(m.algorithmStatus2, 16)) {
                zuptActive = true;
            } else {
                zuptActive = false;
            }
            if( getBit(m.algorithmStatus2, 17) ||
                    getBit(m.algorithmStatus2, 18) ||
                    getBit(m.algorithmStatus2, 19)) {
                zuptOther = true;
            } else {
                zuptOther = false;
            }

        }
        if( (firstMessage) || (priorAlgorithmStatus3 !=m.algorithmStatus3) ) {
            // Currently not very interesting
        }
        if( (firstMessage) || (priorAlgorithmStatus4 !=m.algorithmStatus4) ) {
            if(getBit(m.algorithmStatus4, 28)) {
                flashWriteError = true;
            } else {
                flashWriteError = false;
            }
            if(getBit(m.algorithmStatus4, 29)) {
                flashEraseError = true;
            } else {
                flashEraseError = false;
            }
        }

        priorAlgorithmStatus1 = m.algorithmStatus1;
        priorAlgorithmStatus2 = m.algorithmStatus2;
        priorAlgorithmStatus3 = m.algorithmStatus3;
        priorAlgorithmStatus4 = m.algorithmStatus4;
    }

    if(m.haveINSSystemStatus) {
        if( (firstMessage) || (priorSystemStatus1 != m.systemStatus1) ) {
            if(getBit(m.systemStatus1, 17)) {
                outputAFull = true;
            } else {
                outputAFull = false;
            }
            if(getBit(m.systemStatus1, 18)) {
                outputBFull = true;
            } else {
                outputBFull = false;
            }
        }
        if( (firstMessage) || (priorSystemStatus1 != m.systemStatus1) ) {
            if(getBit(m.systemStatus2, 2)) {
                gpsDetected = true;
            } else {
                gpsDetected = false;
            }
        }
        if( (firstMessage) || (priorSystemStatus1 != m.systemStatus1) ) {
            if(getBit(m.systemStatus3, 18)) {
                systemReady = true;
            } else {
                systemReady = false;
            }
        }

        priorSystemStatus1 = m.systemStatus1;
        priorSystemStatus2 = m.systemStatus2;
        priorSystemStatus3 = m.systemStatus3;
    }

    if((m.haveGNSSInfo1 && (m.gnss[0].gnssGPSQuality != gnssQualPrior)) || firstMessage) {
        gnssInfo i = m.gnss[0];
        gpsQualityKinds q = i.gnssGPSQuality;
        switch(q)
        {
        case gpsQualityNatural_10m:
            gnssQualStr = "Nat 10M";
            gnssQualShortStr = "[10M]";
            break;
        case gpsQualityDifferential_3m:
            gnssQualStr = "Diff 3M";
            gnssQualShortStr = "[3M]";
            break;
        case gpsQualityMilitary_10m:
            gnssQualStr = "Mil 10M";
            gnssQualShortStr = "[10M]";
            break;
        case gpsQualityRTK_0p1m:
            gnssQualStr = "RTK 0.1M";
            gnssQualShortStr = "[0.1M]";
            break;
        case gpsQualityFloatRTK_0p3m:
            gnssQualStr = "RTK 0.3M";
            gnssQualShortStr = "[0.3M]";
            break;
        case gpsQualityOther:
        case gpsQualityInvalid:
        default:
            gnssQualStr = "INVALID";
            gnssQualShortStr = " ?";
            break;
        }
        gnssQualPrior = q;
    }

    if(doLabelUpdate) {
        QString alignmentText;
        alignmentText.append(gnssAlignmentPhase);
        if(gnssAlignmentComplete) {
            alignmentText.append(QString(" %1").arg(gnssQualShortStr));
        }
        updateLabel(gpsAlignment, alignmentText);
    }

    if(doPlotUpdate)
    {
        if(m.haveHeadingRollPitchRate)
        {
            headingsMagnetic.push_front(m.heading);
            headingsMagnetic.pop_back();

            headingsCourse.push_front(m.courseOverGround);
            headingsCourse.pop_back();

            rolls.push_front(m.roll);
            rolls.pop_back();

            pitches.push_front(m.pitch);
            pitches.pop_back();
        }
        if(m.havePosition)
        {

            if(m.longitude > 180)
            {
                longitude = -360+m.longitude;
            } else {
                longitude = m.longitude;
            }

            lats.push_front(m.latitude);
            lats.pop_back();

            longs.push_front(longitude);
            longs.pop_back();

            alts.push_front(m.altitude);
            alts.pop_back();
        }
        if(m.haveSpeedData)
        {
            nVelos.push_front(m.northVelocity);
            nVelos.pop_back();

            eVelos.push_front(m.eastVelocity);
            eVelos.pop_back();

            upVelos.push_front(m.upVelocity);
            upVelos.pop_back();
        }
    }

    if(doWidgetPaint)
    {
        if(m.haveHeadingRollPitchRate)
        {
            if(eadi != NULL)
            {
                eadi->setHeading(m.heading);
                eadi->setPitch(m.pitch * -1 );
                eadi->setRoll(m.roll);
            }

            if(ehsi != NULL)
            {
                ehsi->setHeading(m.heading);
                ehsi->setBearing(m.heading);
            }
        }
        if(m.haveCourseSpeedGroundData)
        {
            if(ehsi != NULL) ehsi->setCourse(m.courseOverGround);
            if(eadi != NULL) eadi->setAirspeed(m.speedOverGround * 1.94384); // knots
            if(asi  != NULL) asi->setAirspeed(m.speedOverGround * 1.94384);
        }
        if(m.haveSpeedData)
        {
            if(vsi  != NULL) vsi->setClimbRate(m.upVelocity * 196.85); // feet per 100 minutes
            if(eadi != NULL) eadi->setClimbRate(m.upVelocity * 196.85);
        }
    }



    if(doWidgetPaint)
    {
        if(ehsi!=NULL) ehsi->redraw();
        if(eadi!=NULL) eadi->redraw();
        if(asi !=NULL) asi->redraw();
        if(vsi !=NULL) vsi->redraw();
    }
    if(doPlotUpdate)
        updatePlots();

    // Every time, process the status:
    processStatus();

    if(firstMessage)
    {
        firstMessage = false;
    }
}

void gpsManager::handleStartsecondaryLog(QString filename)
{
    emit gpsStatusMessage(QString("About to start secondary logging to file %1").arg(filename));
    emit startSecondaryLog(filename);
}

void gpsManager::handleStopSecondaryLog()
{
    emit gpsStatusMessage(QString("Stopping secondary GPS log."));
    emit stopSecondaryLog();
}

void gpsManager::handleGPSStatusMessage(QString message)
{
    emit gpsStatusMessage(QString("GPS Message: ") + message);
}

void gpsManager::handleGPSDataString(QString gpsString)
{
    emit gpsStatusMessage(QString("GPS message: ") + gpsString);
}

void gpsManager::handleGPSConnectionError(int error)
{
    switch(error) {
        case 1:
            {
                emit gpsStatusMessage(QString("Error code from GPS connection: %1 (lost connection). Reconnecting in 1 second.").arg(error));
                break;
            }
        case 7:
            {
                emit gpsStatusMessage(QString("Error code from GPS connection: %1 (could not connect). Reconnecting in 1 second.").arg(error));
                break;
            }
        case 5:
        {
            emit gpsStatusMessage(QString("Error code from GPS connection: %1 (connection timeout). Reconnecting in 1 second.").arg(error));
            break;
        }
        default:
            {
                emit gpsStatusMessage(QString("Error code from GPS connection: %1 (unknown). Reconnecting in 1 second.").arg(error));
                break;
            }
    }

    emit gpsConnectionError(error);
    statusLinkStickyError = true;
    statusConnectedToGPS = false;
    processStatus();
    gpsReconnectTimer.start();
}

void gpsManager::handleGPSReconnectTimer()
{
    // Going here means we have failed to connect and are trying again.
    emit gpsStatusMessage(QString("Attempting to reconnect to GPS."));
    emit connectToGPS(host, port, gpsBinaryLogFilename);
    // If we fail, handleGPSConnectionError is called automatically
}

void gpsManager::handleGPSConnectionGood()
{
    emit gpsStatusMessage(QString("GPS: Connection good"));
    statusLinkStickyError = false; // Safe to clear when a new connection has been made.
    statusConnectedToGPS = true;
    hbErrorCount = 0;
}

void gpsManager::handleGPSTimeout()
{
    // Heartbeat fail
    hbErrorCount++;
    statusGPSHeartbeatOk = false;
    processStatus();
    if(hbErrorCount > 120)
    {
        hbErrorCount = 0;
        // About 2 minutes of missed data
        emit gpsStatusMessage(QString("Error, heartbeat errors exceed threshold, establishing new GPS connection."));
        // disconnect
        initiateGPSDisconnect();
        // reconnect
        handleGPSReconnectTimer();
    }
}

utcTime gpsManager::processUTCstamp(uint64_t t)
{
    utcTime tObj;
    tObj.hour = t / ((float)1E4)/60.0/60.0;
    tObj.minute = ( t / ((float)1E4)/60.0 ) - (tObj.hour*60) ;
    tObj.secondFloat = ( t / ((float)1E4) ) - (tObj.hour*60.0*60.0) - (tObj.minute*60.0);
    tObj.second = ( t / ((float)1E4) ) - (tObj.hour*60.0*60.0) - (tObj.minute*60.0);

    tObj.UTCValidityStr = QString("%1:%2:%3 UTC").arg(tObj.hour, 2, 10, QChar('0')).arg(tObj.minute, 2, 10, QChar('0')).arg(tObj.secondFloat, 6, 'f', 3, QChar('0'));
    tObj.UTCstr = QString("%1:%2:%3 UTC").arg(tObj.hour, 2, 10, QChar('0')).arg(tObj.minute, 2, 10, QChar('0')).arg(tObj.second, 2, 10, QChar('0'));

    return tObj;
}

void gpsManager::processStatus()
{
    // Central function to evaluate the status
    // of the GPS

    // Error status is an "OR" of prior error ('sticky') and current error conditions.

    // Link errors and warnings relate to our GPS decode and connection to the unit over TCP/IP.
    bool gpsLinkError = statusLinkStickyError || !statusGPSHeartbeatOk || !statusConnectedToGPS;
    bool gpsLinkWarning = statusGPSMessagesDropped;

    if(consecutiveDecodeErrors > 10) {
        gpsLinkWarning = true;
    }
    if(consecutiveDecodeErrors > 100) {
        gpsLinkError = true;
    }

    // GPS Trouble errors and warnings relate to status received from the GPS
    // Trouble: something really is badly wrong
    // Warning: Not ready yet
    bool gpsTroubleError = statusTroubleStickyError || !statusGNSSReceptionOk ||
            !gpsValid || gpsRejected || altitudeSaturation || speedSaturation ||
            interpolationMissed || altitudeRejected || zuptActive || zuptOther ||
            flashWriteError || flashEraseError || outputAFull || outputBFull;

    bool gpsTroubleWarning = statusGNSSReceptionWarning || !gpsReceived ||
            gpsWaiting || !gpsDetected || !systemReady || !navPhase || !statusConnectedToGPS;

    // Warnings are not sticky (automatically cleared)

    // Always show the current state of the above variables, which include the stickyness already:

    if(gpsLinkError) {
        updateLED(gpsLinkLED, QLedLabel::StateError);
        statusLinkStickyError = true;
    } else if (gpsLinkWarning) {
        updateLED(gpsLinkLED, QLedLabel::StateWarning);
    } else {
        updateLED(gpsLinkLED, QLedLabel::StateOk);
    }

    if(gpsTroubleError) {
        updateLED(gpsTroubleLED, QLedLabel::StateError);
        statusTroubleStickyError = true;
    } else if (gpsTroubleWarning) {
        updateLED(gpsTroubleLED, QLedLabel::StateWarning);
    } else {
        updateLED(gpsTroubleLED, QLedLabel::StateOk);
    }

    // Clear errors if we were asked to and if there isn't a new reason to
    // flag errors...
    if(statusJustCleared && (!gpsTroubleError) && (!gpsTroubleWarning)) {
        // Reset
        updateLED(gpsTroubleLED, QLedLabel::StateOkBlue);
    }
    if(statusJustCleared && (!gpsLinkError) && (!gpsLinkWarning)) {
        // Reset
        updateLED(gpsLinkLED, QLedLabel::StateOkBlue);
    }

    genStatusMessages();

    if(statusJustCleared)
        statusJustCleared = false;
}

void gpsManager::genStatusMessages()
{
    QMutexLocker locker(&messageMutex);
    errorMessages.clear();
    warningMessages.clear();

    // Warnings:
    if(!navPhase)
        warningMessages << "Not NavPhase";
    if(!gpsReceived)
        warningMessages << "No GPS Received";
    if(gpsWaiting)
        warningMessages << "GPS Waiting";
    if(!gpsDetected)
        warningMessages << "GPS NOT Detected";
    if(!systemReady)
        warningMessages << "SYS NOT Ready";
    if(!statusGPSHeartbeatOk)
        warningMessages << "GPS Msg Cadence";
    if(statusGNSSReceptionWarning)
        warningMessages << "Poor Sat RX";
    if(statusGPSMessagesDropped)
        warningMessages << "Msg Dropped";
    if(consecutiveDecodeErrors > 100) {
        errorMessages << "Many NG Decodes";
    } else if (consecutiveDecodeErrors > 10) {
        warningMessages << "NG Decode";
    }

    // Errors:
    if(!statusConnectedToGPS)
        errorMessages << "Not Connected";
    if(!statusGNSSReceptionOk)
        errorMessages << "No SAT RX";
    if(!gpsValid)
        errorMessages << "GPS invalid";
    if(gpsRejected)
        errorMessages << "GPS Rejected";
    if(altitudeRejected)
        errorMessages << "Alt Reject";
    if(zuptActive)
        errorMessages << "ZUpT Active";
    if(zuptOther)
        errorMessages << "ZUpT Other";
    if(flashWriteError)
        errorMessages << "Flash Write Er";
    if(flashEraseError)
        errorMessages << "Flash Erase Er";
    if(outputAFull)
        errorMessages << "Out A Full";
    if(outputBFull)
        errorMessages << "Out B Full";

    emit statusMessagesSig(errorMessages,
                           warningMessages);
}

void gpsManager::clearStickyError()
{
    statusLinkStickyError = false;
    statusTroubleStickyError = false;
    statusJustCleared = true;
    processStatus();
}
