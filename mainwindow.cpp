#include <QVBoxLayout>
#include <QDebug>
#include "mainwindow.h"
#include "image_type.h"
#include <memory>
MainWindow::MainWindow(QThread* qth, frameWorker* fw, QWidget* parent)
    : QMainWindow(parent)
{
    qRegisterMetaType<frame_c*>("frame_c*");
    qRegisterMetaType<QVector<double>>("QVector<double>");
    qRegisterMetaType<QSharedPointer<QVector<double>>>("QSharedPointer<QVector<double>>");

    this->fw = fw;

    //start worker Thread
    qth->start();

#ifdef VERBOSE
    qDebug() << "fw passed to MainWindow";
#endif
    this->resize(1440,900);
    mainwidget = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout;

    //Create tabs
    tabWidget = new QTabWidget;

    save_server = new saveServer(fw);

    // NOTE: Care should be taken to ensure that tabbed widgets are ordered by the value of their image_type enum
    // signals/slots (currentChanged) make use of this relation
    unfiltered_widget = new frameview_widget(fw, BASE);
    dsf_widget = new frameview_widget(fw, DSF);
    std_dev_widget = new frameview_widget(fw, STD_DEV);
    hist_widget = new histogram_widget(fw);
    vert_mean_widget = new profile_widget(fw,VERTICAL_MEAN);
    horiz_mean_widget = new profile_widget(fw,HORIZONTAL_MEAN);
    vert_cross_widget = new profile_widget(fw,VERTICAL_CROSS);
    horiz_cross_widget = new profile_widget(fw,HORIZONTAL_CROSS);
    fft_mean_widget = new fft_widget(fw);
    raw_play_widget = new playback_widget(fw);

    tabWidget->addTab(unfiltered_widget, QString("Live View"));
    tabWidget->addTab(dsf_widget, QString("Dark Subtraction"));
    tabWidget->addTab(std_dev_widget, QString("Std. Deviation"));
    tabWidget->addTab(hist_widget,QString("Histogram View"));
    tabWidget->addTab(vert_mean_widget,QString("Vertical Mean Profile"));
    tabWidget->addTab(horiz_mean_widget,QString("Horizontal Mean Profile"));
    tabWidget->addTab(vert_cross_widget,QString("Vertical Crosshair Profile"));
    tabWidget->addTab(horiz_cross_widget,QString("Horizontal Crosshair Profile"));
    tabWidget->addTab(fft_mean_widget,QString("FFT Profile"));
    tabWidget->addTab(raw_play_widget,QString("Playback View"));

    layout->addWidget(tabWidget,3);

    //Create controls box
    controlbox = new ControlsBox(fw,tabWidget);
    layout->addWidget(controlbox,1);

    mainwidget->setLayout(layout);
    this->setCentralWidget(mainwidget);

    // Connections
    connect(tabWidget,SIGNAL(currentChanged(int)),controlbox,SLOT(tab_changed_slot(int)));
    controlbox->tab_changed_slot(0);

    connect(fw,SIGNAL(newFrameAvailable()), unfiltered_widget, SLOT(handleNewFrame()));
    connect(controlbox,SIGNAL(startDSFMaskCollection()),fw,SLOT(startCapturingDSFMask()));
    connect(controlbox,SIGNAL(stopDSFMaskCollection()),fw,SLOT(finishCapturingDSFMask()));
    connect(controlbox,SIGNAL(startSavingFinite(unsigned int,QString)),fw,SLOT(startSavingRawData(unsigned int,QString)));
    connect(controlbox,SIGNAL(stopSaving()),fw,SLOT(stopSavingRawData()));
    connect(&controlbox->std_dev_N_slider,SIGNAL(valueChanged(int)),fw,SLOT(setStdDev_N(int)));
    connect(fw,SIGNAL(savingFrameNumChanged(unsigned int)),controlbox,SLOT(updateSaveFrameNum_slot(unsigned int)));

    controlbox->fps_label.setStyleSheet("QLabel {color: green;}");
    if( save_server->isListening() )
    {
        controlbox->server_ip_label.setText( tr("Server IP: %1").arg(save_server->ipAddress) );
        controlbox->server_port_label.setText( tr("Server Port: %1").arg(save_server->port) );
    }
}
void MainWindow::enableStdDevTabs()
{
    qDebug() << "enabling std. dev. tabs";
    tabWidget->setTabEnabled(2,true);
    tabWidget->setTabEnabled(3,true);
    disconnect(fw,SIGNAL(std_dev_ready()),this,SLOT(enableStdDevTabs()));
}

// protected
void MainWindow::keyPressEvent(QKeyEvent* c)
{
    /*!
     * Contains all keyboard shortcuts for liveview2
     */
    playback_widget* pbw = qobject_cast<playback_widget*>(tabWidget->widget(tabWidget->currentIndex()));
    if(pbw) // inside playback widget
    {
        if(!c->modifiers())
        {
            if(c->key() == Qt::Key_Space || c->key() == Qt::Key_Return)
            {
                pbw->playPause();
                c->accept();
                return;
            }
            if(c->key() == Qt::Key_A)
            {
                pbw->moveBackward();
                c->accept();
                return;
            }
            if(c->key() == Qt::Key_D)
            {
                pbw->moveForward();
                c->accept();
                return;
            }
            if(c->key() == Qt::Key_S)
            {
                pbw->stop();
                c->accept();
                return;
            }
            if(c->key() == Qt::Key_R)
            {
                pbw->fastRewind();
                c->accept();
                return;
            }
            if(c->key() == Qt::Key_F)
            {
                pbw->fastForward();
                c->accept();
                return;
            }
            // More key mappings can be provided here
        }
    }
}
