/* Qt includes */
#include <QDebug>
#include <cstdio>
#include <QVBoxLayout>

/* Standard includes */
#include <memory>

/* Live View includes */
#include "mainwindow.h"
#include "image_type.h"

MainWindow::MainWindow(startupOptionsType *optionsIn, QThread *qth, frameWorker *fw, QWidget *parent)
    : QMainWindow(parent)
{
    qRegisterMetaType<frame_c*>("frame_c*");    
    const QString name = "lv:";
    this->fw = fw;
    this->options = optionsIn;

    startDateTime =  QDateTime::currentDateTimeUtc();
    QString startDate;
    startDate.append(startDateTime.toString("yyyyMMdd"));

    if(options->dataLocationSet)
    {
        // Append today's date
        // This will go in to the consoleLog, gps, and raw data saving functions
        if(!options->dataLocation.endsWith("/"))
            options->dataLocation.append("/");

        this->options->dataLocation.append(startDate);
        cLog = new consoleLog(this->options->dataLocation, options->flightMode);
    } else {
        cLog = new consoleLog();
    }

    if(options->flightMode)
        handleMainWindowStatusMessage(QString("This version of FlightView was compiled on %1 at %2 using gcc version %3").arg(QString(__DATE__)).arg(QString(__TIME__)).arg(__GNUC__));
    else
        handleMainWindowStatusMessage(QString("This version of LiveView was compiled on %1 at %2 using gcc version %3").arg(QString(__DATE__)).arg(QString(__TIME__)).arg(__GNUC__));

    handleMainWindowStatusMessage(QString("The compilation was performed by %1@%2.").arg(QString(UNAME)).arg(QString(HOST)));


    connect(fw, SIGNAL(sendStatusMessage(QString)), this, SLOT(handleMainWindowStatusMessage(QString)));


    /*! start the workerThread from main */
    qth->setObjectName(name + "worker");
    qth->start();

#ifdef VERBOSE
    qDebug() << "fw passed to MainWindow";
#endif
    //this->resize(1280, 1024);
    mainwidget = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout;

    /* Create tabs */
    tabWidget = new QTabWidget;

    save_server = new saveServer(fw);

    /*! \note Care should be taken to ensure that tabbed widgets are ordered by the value of their image_type enum
     * signals/slots (currentChanged) make use of this relation */
    unfiltered_widget = new frameview_widget(fw, BASE);
    //dsf_widget = new frameview_widget(fw, DSF);
    dsf_widget = NULL;
    waterfall_widget = new frameview_widget(fw, WATERFALL);

    //flight_screen = new flight_widget(fw, *options, this);
    flight_screen = new flight_widget(fw, *options);

    connect(flight_screen, SIGNAL(statusMessage(QString)), this, SLOT(handleGeneralStatusMessage(QString)));

    std_dev_widget = new frameview_widget(fw, STD_DEV);
    hist_widget = new histogram_widget(fw);
    vert_mean_widget = new profile_widget(fw, VERTICAL_MEAN);
    horiz_mean_widget = new profile_widget(fw, HORIZONTAL_MEAN);
    vert_cross_widget = new profile_widget(fw, VERTICAL_CROSS);
    horiz_cross_widget = new profile_widget(fw, HORIZONTAL_CROSS);
    vert_overlay_widget = new profile_widget(fw, VERT_OVERLAY);
    fft_mean_widget = new fft_widget(fw);

    connect(unfiltered_widget, SIGNAL(statusMessage(QString)), this, SLOT(handleMainWindowStatusMessage(QString)));
    connect(waterfall_widget, SIGNAL(statusMessage(QString)), this, SLOT(handleMainWindowStatusMessage(QString)));
    connect(std_dev_widget, SIGNAL(statusMessage(QString)), this, SLOT(handleMainWindowStatusMessage(QString)));
    connect(save_server, SIGNAL(sigMessage(QString)), this, SLOT(handleGeneralStatusMessage(QString)));
    raw_play_widget = NULL;
    if(!options->flightMode)
    {
        //raw_play_widget = new playback_widget(fw);
        setWindowTitle("Ground Mode");
    } else {
        setWindowTitle("Flight Mode");
    }

    /* Add tabs in order */
    tabWidget->addTab(unfiltered_widget, QString("FPA")); // check commit log here
    //tabWidget->addTab(dsf_widget, QString("Dark Subtraction"));
    tabWidget->addTab(waterfall_widget, QString("Waterfall"));
    tabWidget->addTab(flight_screen, QString("Flight"));
    tabWidget->addTab(std_dev_widget, QString("Std. Deviation"));
    tabWidget->addTab(hist_widget, QString("Histogram View"));
    tabWidget->addTab(vert_mean_widget, QString("Vertical Mean Profile"));
    tabWidget->addTab(horiz_mean_widget, QString("Horizontal Mean Profile"));
    tabWidget->addTab(vert_cross_widget, QString("Vertical Crosshair Profile"));
    tabWidget->addTab(horiz_cross_widget, QString("Horizontal Crosshair Profile"));
    tabWidget->addTab(vert_overlay_widget, QString("Vertical Overlay"));
    tabWidget->addTab(fft_mean_widget, QString("FFT Profile"));
    if(!options->flightMode)
    {
        //tabWidget->addTab(raw_play_widget, QString("Playback View"));
    } else {
        raw_play_widget = NULL;
    }

    layout->addWidget(tabWidget, 3);

    /* Create controls box */
    controlbox = new ControlsBox(fw, tabWidget, *options);
    connect(controlbox, SIGNAL(statusMessage(QString)), this, SLOT(handleMainWindowStatusMessage(QString)));
    layout->addWidget(controlbox, 1);

    mainwidget->setLayout(layout);
    this->setCentralWidget(mainwidget);

    // Keyboard commands will be active at all times unless a text box is selected. You can revert to mainwindow focus by clicking or pressing tab
    this->setFocusPolicy(Qt::StrongFocus);

    /* Connections */
    connect(tabWidget,SIGNAL(currentChanged(int)),controlbox,SLOT(tab_changed_slot(int)));
    controlbox->tab_changed_slot(0);

    connect(flight_screen, SIGNAL(sendDiskSpaceAvailable(quint64,quint64)), controlbox, SLOT(updateDiskSpace(quint64,quint64)));

    if(options->flightMode)
    {
        // Flight Tab
        int flightIndex = tabWidget->indexOf(flight_screen);
        if(flightIndex >-1)
        {
            tabWidget->setCurrentIndex(flightIndex);
        } else {
            handleMainWindowStatusMessage("ERROR: Could not set current screen to Flight Screen.");
        }
    }

    // connect(fw, SIGNAL(newFrameAvailable()), unfiltered_widget, SLOT(handleNewFrame())); // should not be needed with render timers running.
    // connect(fw, SIGNAL(newFrameAvailable()), flight_screen, SLOT(handleNewFrame())); // depreciated
    connect(controlbox, SIGNAL(startDSFMaskCollection()), fw,SLOT(startCapturingDSFMask()));
    connect(controlbox, SIGNAL(stopDSFMaskCollection()), fw, SLOT(finishCapturingDSFMask()));
    connect(controlbox, SIGNAL(startSavingFinite(unsigned int, QString, unsigned int)), fw, SLOT(startSavingRawData(unsigned int, QString, unsigned int)));
    connect(controlbox, SIGNAL(stopSaving()),fw,SLOT(stopSavingRawData()));
    connect(controlbox->std_dev_N_slider, SIGNAL(valueChanged(int)), fw, SLOT(setStdDev_N(int)));
    connect(controlbox, SIGNAL(toggleStdDevCalculation(bool)), fw, SLOT(enableStdDevCalculation(bool)));
    connect(controlbox, SIGNAL(haveReadPreferences(settingsT)), this, SLOT(handlePreferenceRead(settingsT)));
    connect(controlbox, SIGNAL(haveReadPreferences(settingsT)), flight_screen, SLOT(handlePrefs(settingsT)));
    connect(fw, SIGNAL(savingFrameNumChanged(unsigned int)), controlbox, SLOT(updateSaveFrameNum_slot(unsigned int)));
    connect(fw, SIGNAL(updateFrameCountDisplay(int)), controlbox, SLOT(setFrameNumber(int)));
    controlbox->fps_label.setStyleSheet("QLabel {color: green;}");
    if(save_server->isListening()) {
        handleMainWindowStatusMessage(QString("SaveServer IP: %1").arg(save_server->ipAddress.toString()));
        handleMainWindowStatusMessage(QString("SaveServer Port: %1").arg(save_server->port));
    }

    if(options->rtpCam) {
        if(options->rtpNextGen) {
            handleMainWindowStatusMessage("Camera: RTP NextGen");
        } else {
            handleMainWindowStatusMessage("Camera: RTP gstreamer");
        }

        QString listeningString = "RTP SRC";

        if(options->havertpInterface) {
            listeningString.append(QString(": %1").arg(options->rtpInterface));
        }
        if(options->havertpAddress) {
            listeningString.append(QString(": %1").arg(options->rtpAddress));
        }

        controlbox->server_ip_label.setText(listeningString);
        handleMainWindowStatusMessage(listeningString);
        controlbox->server_port_label.setText(QString("RTP Port: %1").arg(options->rtpPort));
        handleMainWindowStatusMessage(QString("RTP Port: %1").arg(options->rtpPort));
    } else if(options->xioCam){
        controlbox->server_ip_label.setText("XIO active");
        handleMainWindowStatusMessage("Camera: XIO (files)");
    } else {
        controlbox->server_ip_label.setText("CameraLink active");
        handleMainWindowStatusMessage("Camera: CameraLink");
    }

    connect(controlbox, SIGNAL(debugSignal()), this, SLOT(debugThis()));

    connect(controlbox, SIGNAL(startDataCollection(QString)), flight_screen, SLOT(startDataCollection(QString)));
    connect(controlbox, SIGNAL(stopDataCollection()), flight_screen, SLOT(stopDataCollection()));
    connect(fw, SIGNAL(setColorScheme_signal(int,bool)), flight_screen, SLOT(handleNewColorScheme(int,bool)));
    connect(this, SIGNAL(toggleStdDevCalc(bool)), fw, SLOT(enableStdDevCalculation(bool)));
    connect(controlbox, SIGNAL(sendRGBLevels(double,double,double,double,bool)), flight_screen, SLOT(setRGBLevels(double,double,double,double,bool)));
    connect(controlbox, SIGNAL(setWFTargetFPS_render(int)), flight_screen, SLOT(setWFFPS_render(int)));
    connect(controlbox, SIGNAL(setWFTargetFPS_primary(int)), flight_screen, SLOT(setWFFPS_primary(int)));
    connect(controlbox, SIGNAL(setWFTargetFPS_secondary(int)), flight_screen, SLOT(setWFFPS_secondary(int)));

    controlbox->getPrefsExternalTrig();
    connect(controlbox, &ControlsBox::showConsoleLog,
            [=]() {
            cLog->show();
            cLog->setWindowState(Qt::WindowActive);
            cLog->raise();
    });
    if(options->dataLocationSet)
    {
        handleMainWindowStatusMessage(QString("Data storage location: [%1]").arg(options->dataLocation));
    }
    if(options->runStdDevCalculation)
    {
        emit toggleStdDevCalc(true);
        handleMainWindowStatusMessage(QString("Standard Deviation calculation enabled"));
    } else {
        emit toggleStdDevCalc(false);
        handleMainWindowStatusMessage(QString("Standard Deviation calculation disabled"));
    }

    if(options->flightMode)
    {
        handleMainWindowStatusMessage("Flight Mode ENABLED.");
    } else {
        handleMainWindowStatusMessage("Flight Mode DISABLED.");
    }
    if(options->er2mode) {
        handleMainWindowStatusMessage("ER2 Mode ENABLED.");
    }
    if(options->headless) {
        handleMainWindowStatusMessage("Headless Mode ENABLED.");
    }

    connect(this->controlbox, &ControlsBox::setCameraPause,
            [=](bool paused) {
        fw->setCameraPaused(paused);
        if(paused)
        {
            handleMainWindowStatusMessage("Pausing Camera");
        }
        else {
            handleMainWindowStatusMessage("Unpausing Camera");
        }
    });


    connect(this->controlbox, &ControlsBox::loadDarkFile,
            [=](QString filename, fileFormat_t fileformat) {
            fw->loadDarkFile(filename, fileformat);
    });

    // Control functions needed for ARTIC:
    connect(save_server, SIGNAL(startTakingDarks()), controlbox, SLOT(startTakingDarks()));
    connect(save_server, SIGNAL(stopTakingDarks()), controlbox, SLOT(stopTakingDarks()));
    connect(save_server, SIGNAL(startSavingFlightData()), controlbox, SLOT(save_finite_button_slot()));
    connect(save_server, SIGNAL(stopSavingData()), controlbox, SLOT(stopSavingData()));
    //this->setWindowState( (windowState() & ~Qt::WindowMinimized ) | Qt::WindowActive);
    //this->raise();
    //this->activateWindow();
    handleMainWindowStatusMessage("Started");
}

void MainWindow::closeEvent(QCloseEvent *e)
{

    int reply = QMessageBox::question(this, "Are you sure?", "Are you sure you want to quit?", QMessageBox::Ok, QMessageBox::Cancel, QMessageBox::NoButton);
    handleMainWindowStatusMessage("User requested to close main window.");
    if(reply==QMessageBox::Ok)
    {
        handleMainWindowStatusMessage("User confirmed quit.");
        flight_screen->setStop();
        cLog->close();
        QList<QWidget*> allWidgets = findChildren<QWidget*>();
        for(int i = 0; i < allWidgets.size(); ++i)
            allWidgets.at(i)->close();
        //e->accept();
        QApplication::exit();
    } else {
        handleMainWindowStatusMessage("User canceled close request.");
        e->ignore();
    }
}

void MainWindow::handleMainWindowStatusMessage(QString message)
{
    //std::cout << "STDOUT: Status Message: " << message.toLocal8Bit().toStdString() << std::endl;
    //qDebug() << __PRETTY_FUNCTION__ << "Status message: " << message;
    cLog->insertText(QString("[MainWindow]: ") + message);
}

void MainWindow::handleGeneralStatusMessage(QString message)
{
    //std::cout << "STDOUT: Status Message: " << message.toLocal8Bit().toStdString() << std::endl;
    //qDebug() << __PRETTY_FUNCTION__ << "Status message: " << message;
    cLog->insertText(message);
}

void MainWindow::enableStdDevTabs()
{
    //qDebug() << "enabling std. dev. tabs";
    tabWidget->setTabEnabled(2, true);
    tabWidget->setTabEnabled(3, true);
    disconnect(fw, SIGNAL(std_dev_ready()), this, SLOT(enableStdDevTabs()));
}

// protected
void MainWindow::keyPressEvent(QKeyEvent *c)
{
    /*! \brief Contains all keyboard shortcuts for LiveView.
     * \param c The key from the keyboard buffer.
     * \paragraph
     *
     * This function checks which widget is currently being displayed to check for widget-specific controls using the
     * qobject_cast method.
     * \paragraph
     *
     * ------------------------------
     * Keyboard Controls
     * ------------------------------
     * \paragraph
     *
     * p - Toggle the Precision Slider
     * m - Toggle the Dark Subtraction Mask (if one is present)
     * , - Begin recording Dark Frames
     * . - Stop recording dark frames
     * \paragraph
     *
     * FOR FRAME VIEWS (RAW IMAGE, DARK SUBTRACTION, STANDARD DEVIATION)
     * left click - profile the data at the specified coordinate
     * esc - reset the crosshairs
     * d - Toggle display of the crosshairs
     * \paragraph
     * FOR THE HISTOGRAM WIDGET
     * r - reset the range of the display. Zooming may make it difficult to return to the original scale of the plot.
     * \paragraph
     * FOR THE PLAYBACK WIDGET
     * 'drag and drop onto the viewing window' - load the selected file. WARNING: Any filetype is accepted. This means if the filetype is not data, garbage will be displayed in the viewing window.
     * s - Stop playback and return to the first frame
     * return - Play/Pause
     * f - Fast Forward. Multiple presses increase the fast forward multiplier up to 64x faster.
     * r - Rewind. Multple presses inreas the rewind multiplier up to 64x faster.
     * a - Move back one frame. Only works when playback is paused.
     * d - Move forward one frame. Only works when playback is paused.
     *
     * \author Jackie Ryan
     */

    if(options->headless) {
        return;
    }

    // This function is somewhat unsafe and needs to be rewritten.
    return;

    QWidget* current_tab = tabWidget->widget(tabWidget->currentIndex());
    profile_widget *ppw;
    frameview_widget *fvw;
    if (!c->modifiers()) {
        if (current_tab == raw_play_widget) {
            if (c->key() == Qt::Key_Space || c->key() == Qt::Key_Return) {
                //raw_play_widget->playPause();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_A)
            {
                //raw_play_widget->moveBackward();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_D) {
                //raw_play_widget->moveForward();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_S) {
                //raw_play_widget->stop();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_R) {
                //raw_play_widget->fastRewind();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_F) {
                //raw_play_widget->fastForward();
                c->accept();
                return;
            }
            // More key mappings can be provided here
        } else if (current_tab == unfiltered_widget || current_tab == dsf_widget) {
            fvw = qobject_cast<frameview_widget*>(current_tab);
            if (c->key() == Qt::Key_Escape) {
                fw->crosshair_x = -1;
                fw->crosshair_y = -1;
                fw->crossStartCol = -1;
                fw->crossWidth = -1;
                fw->crossStartRow = -1;
                fw->crossHeight = -1;
                vert_cross_widget->hideCallout();
                horiz_cross_widget->hideCallout();

                c->accept();
                return;
            } else if (c->key() == Qt::Key_D) {
                fvw->toggleDisplayCrosshair();
                c->accept();
                return;
            }
        } else if (current_tab == hist_widget) {
        /*! \example How to add a keyboard shortcut to Live View
         * @{ */
            if(c->key() == Qt::Key_R) {
                hist_widget->resetRange();
                c->accept();
                return;
            }
        /*! @} */
        } else if ((ppw = qobject_cast<profile_widget*>(current_tab))) {
            if (c->key() == Qt::Key_Escape) {
                ppw->hideCallout();
                c->accept();
                return;
            } else if (c->key() == Qt::Key_R) {
                ppw->updateCeiling(fw->base_ceiling);
                ppw->updateFloor(0);
                c->accept();
                return;
            }
        }
        if (c->key() == Qt::Key_P) {
            controlbox->low_increment_cbox.setChecked(!controlbox->low_increment_cbox.isChecked());
            c->accept();
            return;
        } else if ((controlbox->use_DSF_cbox.isEnabled()) && (c->key() == Qt::Key_M)) {
            controlbox->use_DSF_cbox.setChecked(!controlbox->use_DSF_cbox.isChecked());
        } else if (c->key() == Qt::Key_Comma) {
            controlbox->collect_dark_frames_button.click();
        } else if (c->key() == Qt::Key_Period) {
            controlbox->stop_dark_collection_button.click();
        }
    }
}

//void MainWindow::closeEvent(QCloseEvent *e)
//{
//    Q_UNUSED(e);


//}

void MainWindow::handlePreferenceRead(settingsT prefs)
{
    if(options->flightMode)
    {
        if(prefs.hideFFT)
            removeTab("FFT Profile");
        if(prefs.hideVerticalOverlay)
            removeTab("Vertical Overlay");
        if(prefs.hideVertMeanProfile)
            removeTab("Vertical Mean Profile");
        if(prefs.hideVertCrosshairProfile)
            removeTab("Vertical Crosshair Profile");
        if(prefs.hideHorizontalMeanProfile)
            removeTab("Horizontal Mean Profile");
        if(prefs.hideHorizontalCrosshairProfile)
            removeTab("Horizontal Crosshair Profile");
        if(prefs.hideHistogramView)
            removeTab("Histogram View");
        if(prefs.hideStddeviation)
            removeTab("Std. Deviation");
        if(prefs.hideWaterfallTab)
            removeTab("Waterfall");
    }

    if(!options->runStdDevCalculation)
    {
        prefs.hideStddeviation = true;
        removeTab("Std. Deviation");
        prefs.hideHistogramView = true;
        removeTab("Histogram View");
    }

    // One of these two widgets may be the first widget shown,
    // which will not receive the "tab changed" signal to update
    // the levels. Thus, we update here:

    if(unfiltered_widget) {
        unfiltered_widget->updateCeiling(prefs.frameViewCeiling);
        unfiltered_widget->updateFloor(prefs.frameViewFloor);
    }

    if(flight_screen) {
        flight_screen->updateCeiling(prefs.flightCeiling);
        flight_screen->updateFloor(prefs.flightFloor);
    }

    handleMainWindowStatusMessage(QString("2s compliment setting: %1").arg(prefs.use2sComp?"Enabled":"Disabled"));

    if(options->headless) {
        this->resize(1558, 1024);
    } else {
        if( (prefs.preferredWindowWidth < 4096) && (prefs.preferredWindowHeight < 4096) && (prefs.preferredWindowWidth > 0) && (prefs.preferredWindowHeight > 0))
        {
            this->resize(prefs.preferredWindowWidth, prefs.preferredWindowHeight);
        } else {
            handleMainWindowStatusMessage(QString("Warning, preferred window size out of range: width %1, height %2").arg(prefs.preferredWindowWidth).arg(prefs.preferredWindowHeight));
        }
    }

    // Note: These prefs are not saved correctly currently, so do not restore.
    // restoreGeometry(prefs.windowGeometry);
    // restoreState(prefs.windowState);
}

void MainWindow::removeTab(QString tabTitle)
{
    // Note: This isn't optimal, and the tab cannot be added again without
    // some interesting consequences.
    QString title;
    for(int t=0; t < tabWidget->count(); t++)
    {
        title = tabWidget->tabText(t);
        if(title == tabTitle)
        {
            tabWidget->removeTab(t);
            break;
        }
    }
}

void MainWindow::debugThis()
{
    handleMainWindowStatusMessage("Debug function reached.");
    qDebug() << __PRETTY_FUNCTION__ << ": Debug reached inside MainWindow class.";
    flight_screen->debugThis();
    fw->debugThis();
}
