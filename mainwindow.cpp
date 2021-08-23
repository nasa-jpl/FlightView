/* Qt includes */
#include <QDebug>
#include <cstdio>
#include <QVBoxLayout>

/* Standard includes */
#include <memory>

/* Live View includes */
#include "mainwindow.h"
#include "image_type.h"

MainWindow::MainWindow(startupOptionsType options, QThread *qth, frameWorker *fw, QWidget *parent)
    : QMainWindow(parent)
{
    qRegisterMetaType<frame_c*>("frame_c*");
    //qRegisterMetaType<QVector<double>>("QVector<double>");
    //qRegisterMetaType<QSharedPointer<QVector<double>>>("QSharedPointer<QVector<double>>");

    this->fw = fw;
    this->options = options;

    /*! start the workerThread from main */
    qth->start();

#ifdef VERBOSE
    qDebug() << "fw passed to MainWindow";
#endif
    this->resize(1280, 1024);
    mainwidget = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout;

    /* Create tabs */
    tabWidget = new QTabWidget;

    save_server = new saveServer(fw);

    /*! \note Care should be taken to ensure that tabbed widgets are ordered by the value of their image_type enum
     * signals/slots (currentChanged) make use of this relation */
    unfiltered_widget = new frameview_widget(fw, BASE);
    dsf_widget = new frameview_widget(fw, DSF);
    waterfall_widget = new frameview_widget(fw, WATERFALL);
    flight_screen = new flight_widget(fw, options);
    std_dev_widget = new frameview_widget(fw, STD_DEV);
    hist_widget = new histogram_widget(fw);
    vert_mean_widget = new profile_widget(fw, VERTICAL_MEAN);
    horiz_mean_widget = new profile_widget(fw, HORIZONTAL_MEAN);
    vert_cross_widget = new profile_widget(fw, VERTICAL_CROSS);
    horiz_cross_widget = new profile_widget(fw, HORIZONTAL_CROSS);
    vert_overlay_widget = new profile_widget(fw, VERT_OVERLAY);
    fft_mean_widget = new fft_widget(fw);
    raw_play_widget = new playback_widget(fw);

    /* Add tabs in order */
    tabWidget->addTab(unfiltered_widget, QString("Live View"));
    tabWidget->addTab(dsf_widget, QString("Dark Subtraction"));
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
    tabWidget->addTab(raw_play_widget, QString("Playback View"));

    layout->addWidget(tabWidget, 3);

    /* Create controls box */
    controlbox = new ControlsBox(fw, tabWidget);
    layout->addWidget(controlbox, 1);

    mainwidget->setLayout(layout);
    this->setCentralWidget(mainwidget);

    // Keyboard commands will be active at all times unless a text box is selected. You can revert to mainwindow focus by clicking or pressing tab
    this->setFocusPolicy(Qt::StrongFocus);

    /* Connections */
    connect(tabWidget,SIGNAL(currentChanged(int)),controlbox,SLOT(tab_changed_slot(int)));
    controlbox->tab_changed_slot(0);

    if(options.flightMode)
    {
        // Flight Tab
        tabWidget->setCurrentIndex(3);
    }

    connect(fw, SIGNAL(newFrameAvailable()), unfiltered_widget, SLOT(handleNewFrame()));
    connect(controlbox, SIGNAL(startDSFMaskCollection()), fw,SLOT(startCapturingDSFMask()));
    connect(controlbox, SIGNAL(stopDSFMaskCollection()), fw, SLOT(finishCapturingDSFMask()));
    connect(controlbox, SIGNAL(startSavingFinite(unsigned int, QString, unsigned int)), fw, SLOT(startSavingRawData(unsigned int, QString, unsigned int)));
    connect(controlbox, SIGNAL(stopSaving()),fw,SLOT(stopSavingRawData()));
    connect(controlbox->std_dev_N_slider, SIGNAL(valueChanged(int)), fw, SLOT(setStdDev_N(int)));
    connect(fw, SIGNAL(savingFrameNumChanged(unsigned int)), controlbox, SLOT(updateSaveFrameNum_slot(unsigned int)));

    controlbox->fps_label.setStyleSheet("QLabel {color: green;}");
    if(save_server->isListening()) {
        controlbox->server_ip_label.setText(tr("Server IP: %1").arg(save_server->ipAddress.toString()));
        controlbox->server_port_label.setText(tr("Server Port: %1").arg(save_server->port));
    }

    connect(controlbox, SIGNAL(debugSignal()), this, SLOT(debugThis()));

    connect(controlbox, SIGNAL(startDataCollection(QString)), flight_screen, SLOT(startDataCollection(QString)));
    connect(controlbox, SIGNAL(stopDataCollection()), flight_screen, SLOT(stopDataCollection()));
    connect(flight_screen, SIGNAL(statusMessage(QString)), this, SLOT(handleStatusMessage(QString)));
    qDebug() << __PRETTY_FUNCTION__ << "started";
}

void MainWindow::handleStatusMessage(QString message)
{
    std::cout << "STDOUT: Status Message: " << message.toLocal8Bit().toStdString() << std::endl;
    qDebug() << __PRETTY_FUNCTION__ << "Status message: " << message;
}

void MainWindow::enableStdDevTabs()
{
    qDebug() << "enabling std. dev. tabs";
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
    QWidget* current_tab = tabWidget->widget(tabWidget->currentIndex());
    profile_widget *ppw;
    frameview_widget *fvw;
    if (!c->modifiers()) {
        if (current_tab == raw_play_widget) {
            if (c->key() == Qt::Key_Space || c->key() == Qt::Key_Return) {
                raw_play_widget->playPause();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_A)
            {
                raw_play_widget->moveBackward();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_D) {
                raw_play_widget->moveForward();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_S) {
                raw_play_widget->stop();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_R) {
                raw_play_widget->fastRewind();
                c->accept();
                return;
            }
            if (c->key() == Qt::Key_F) {
                raw_play_widget->fastForward();
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
void MainWindow::closeEvent(QCloseEvent *e)
{
    Q_UNUSED(e);

    QList<QWidget*> allWidgets = findChildren<QWidget*>();
    for(int i = 0; i < allWidgets.size(); ++i)
        allWidgets.at(i)->close();
}

void MainWindow::debugThis()
{
    qDebug() << __PRETTY_FUNCTION__ << ": Debug reached inside MainWindow class.";
    flight_screen->debugThis();
}
