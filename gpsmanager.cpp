#include "gpsmanager.h"

gpsManager::gpsManager()
{
    qRegisterMetaType<gpsMessage>();
    statusStickyError = false;

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
    gpsThread->quit();
    gpsThread->wait();
    //delete gpsThread;
    //delete gps;
}

void gpsManager::initiateGPSConnection(QString host = "10.0.0.6", int port=(int)8111, QString gpsBinaryLogFilename = "")
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
    emit connectToGPS(host, port, gpsBinaryLogFilename);
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

    headings.resize(vecSize);
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
    if(gpsOkLED != NULL)gpsOkLED->setText("");
    if(gpsOkLED != NULL)gpsOkLED->setState(QLedLabel::StateOkBlue);
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

    gpsThread->start();

    gpsMessageHeartbeat.setInterval(500); // half second, expected is 5ms.
    connect(&gpsMessageHeartbeat, SIGNAL(timeout()), this, SLOT(handleGPSTimeout()));
}

void gpsManager::preparePlots()
{
    // Time axis:
    uint16_t t=0;
    for(int i=vecSize; i > 0; i--)
    {
        timeAxis[i-1] = t++;
    }

    if(plotRollPitch != NULL)
    {
        plotRollPitch->addGraph();
        plotRollPitch->yAxis->setRange(-10, 10); // Lat
        plotRollPitch->addGraph(plotRollPitch->xAxis, plotRollPitch->yAxis ); // Lat
        plotRollPitch->addGraph(plotRollPitch->xAxis, plotRollPitch->yAxis ); // Long
        setTimeAxis(plotRollPitch->xAxis);
        plotRollPitch->yAxis->setLabel("Degrees");
        setPlotTitle(plotRollPitch, titleLatLong, "Pitch and Roll");
        plotRollPitch->plotLayout()->addElement(0,-1,titleLatLong);
        plotRollPitch->graph(0)->setPen(QPen(Qt::red)); // roll
        plotRollPitch->graph(1)->setPen(QPen(Qt::green)); // pitch
        setPlotTitle(plotRollPitch, titleLatLong, "Pitch and Roll");
        plotRollPitch->graph(0)->setName("Roll");
        plotRollPitch->graph(1)->setName("Pitch");
        plotRollPitch->removeGraph(2);
        plotRollPitch->legend->setVisible(true);
        setPlotColors(plotRollPitch, true);
    }
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
        this->plotRollPitch->graph(1)->setData(timeAxis, pitches); // should be longs
        this->plotRollPitch->replot();
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

void gpsManager::insertLEDs(QLedLabel *gpsOk)
{
    this->gpsOkLED = gpsOk;
}

void gpsManager::insertPlots(QCustomPlot *gpsRollPitchplot)
{
    this->plotRollPitch = gpsRollPitchplot;
}

void gpsManager::insertLabels(QLabel *gpsLat, QLabel *gpsLong, QLabel *gpsAltitude,
                              QLabel *gpsUTCtime, QLabel *gpsUTCdate, QLabel *gpsUTCValid,
                              QLabel *gpsGroundSpeed,
                              QLabel *gpsHeading, QLabel *gpsRoll, QLabel *gpsPitch,
                              QLabel *gpsQuality,
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

void gpsManager::receiveGPSMessage(gpsMessage m)
{
    // Entry point for new messages
    float longitude;

    if(m.validDecode)
    {
        this->m = m;
        statusMessageDecodeOk = true;
    } else {
        statusMessageDecodeOk = false;
        processStatus();
        return;
    }

    msgsReceivedCount++;

    if(firstMessage)
    {
        gnssStatusTime.start();
        firstMessage = false;
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

    if(m.numberDropped > 0)
    {
        statusGPSMessagesDropped = true;
    } else {
        statusGPSMessagesDropped = false;
    }

    bool doPlotUpdate = (msgsReceivedCount%updatePlotsInverval)==0;
    bool doWidgetPaint = (msgsReceivedCount%updateAvionicsWidgetsInterval)==0;
    bool doLabelUpdate = (msgsReceivedCount%updateLabelsInterval)==0;

    if(doLabelUpdate)
    {
        utcTime navValidity =  processUTCstamp(m.navDataValidityTime);
        updateLabel(gpsUTCValidity, navValidity.UTCValidityStr);

        if(m.haveAltitudeHeading)
        {
            updateLabel(gpsRoll, QString("%1").arg(m.roll));
            updateLabel(gpsHeading, QString("%1").arg(m.heading));
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
            updateLabel(gpsLat, QString("%1").arg(m.latitude, 12, 'f', 8, QChar('0')));
            updateLabel(gpsLong, QString("%1").arg(longitude, 12, 'f', 8, QChar('0')));
            updateLabel(gpsAltitude, QString("%1").arg(m.altitude));
        }
        if(m.haveCourseSpeedGroundData)
        {
            updateLabel(gpsGroundSpeed, QString("%1").arg(m.speedOverGround * 1.94384));
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
        if(m.haveUTC)
        {
            utcTime UTCdataValidityTime = processUTCstamp(m.UTCdataValidityTime);
            updateLabel(gpsUTCdate, UTCdataValidityTime.UTCstr);
        }
    }

    if(doPlotUpdate)
    {
        if(m.haveHeadingRollPitchRate)
        {
            headings.push_front(m.heading);
            headings.pop_back();

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
            if(ehsi != NULL)
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
    emit gpsStatusMessage(QString("Error code from GPS connection: %1").arg(error));
}

void gpsManager::handleGPSConnectionGood()
{
    emit gpsStatusMessage(QString("GPS: Connection good"));
    statusConnectedToGPS = true;
}

void gpsManager::handleGPSTimeout()
{
    // Heartbeat fail
    statusGPSHeartbeatOk = false;
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

    bool trouble = statusStickyError || !statusGPSHeartbeatOk || !statusUTCok || !statusGNSSReceptionOk;
    bool warning = statusGNSSReceptionWarning;

    // Warnings are not sticky (automatically cleared)
    if(warning && !trouble)
    {
        if(gpsOkLED != NULL)gpsOkLED->setState(QLedLabel::StateWarning);
    }

    // Trouble is sticky (clearning is manually done by the user)
    if(trouble)
    {
        if(gpsOkLED != NULL)gpsOkLED->setState(QLedLabel::StateError);
        statusStickyError = true;
    } else {
        if(gpsOkLED != NULL)gpsOkLED->setState(QLedLabel::StateOk);
    }
    if(statusJustCleared && !trouble)
    {
        if(gpsOkLED != NULL)gpsOkLED->setState(QLedLabel::StateOkBlue);
        statusJustCleared = false;
    }
}

void gpsManager::clearStickyError()
{
    statusStickyError = false;
    statusJustCleared = true;
    processStatus();
}
