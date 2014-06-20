#include <QVBoxLayout>
#include <QDebug>
#include "mainwindow.h"
#include "image_type.h"
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{

    createBackend();
    this->resize(1324,830);
    mainwidget = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout;

    //Create tabs
    tabWidget = new QTabWidget;
    unfiltered_widget = new frameview_widget(fw, BASE);
    dsf_widget = new frameview_widget(fw, DSF);
    std_dev_widget = new frameview_widget(fw, STD_DEV);

    tabWidget->addTab(unfiltered_widget, QString("Live View"));
    tabWidget->addTab(dsf_widget, QString("Dark Subtraction"));
    tabWidget->addTab(std_dev_widget, QString("Std. Deviation"));


    layout->addWidget(tabWidget,3);
    //Create controls box

    controlbox = new ControlsBox();
    layout->addWidget(controlbox,1);


    mainwidget->setLayout(layout);


    this->setCentralWidget(mainwidget);

    //Connect everything together



    //connect(controlbox->collect_dark_frames_button,SIGNAL(clicked()),this,SLOT(testslot()));
    connect(controlbox->run_display_button, SIGNAL(clicked()), this, SLOT(connectAndStartBackend()));
    connect(controlbox->stop_display_button, SIGNAL(clicked()), this, SLOT(destroy10Backend()));


}

void MainWindow::createBackend()
{
    workerThread = new QThread();
    fw = new frameWorker();

    fw->moveToThread(workerThread);



}
void MainWindow::connectAndStartBackend()
{
    connect(workerThread,SIGNAL(started()), fw, SLOT(captureFrames()));

    connect(fw,SIGNAL(newFrameAvailable()), unfiltered_widget, SLOT(handleNewFrame()));
    connect(fw,SIGNAL(newFrameAvailable()), dsf_widget, SLOT(handleNewFrame()));
    connect(fw,SIGNAL(newFrameAvailable()), std_dev_widget, SLOT(handleNewFrame()));
    connect(controlbox->collect_dark_frames_button,SIGNAL(clicked()),fw,SLOT(startCapturingDSFMask()));
    connect(controlbox->stop_dark_collection_button,SIGNAL(clicked()),fw,SLOT(finishCapturingDSFMask()));
    connect(controlbox,SIGNAL(mask_selected(const char *)),fw,SLOT(loadDSFMask(const char *)));

    connect(unfiltered_widget->toggleGrayScaleButton,SIGNAL(clicked()),unfiltered_widget,SLOT(toggleGrayScale()));

    connect(controlbox->ceiling_slider, SIGNAL(valueChanged(int)), dsf_widget, SLOT(updateCeiling(int)));
    connect(controlbox->floor_slider, SIGNAL(valueChanged(int)), dsf_widget, SLOT(updateFloor(int)));

    //connect(controlbox->save_frames_button,SIGNAL(clicked()),fw,SLOT(startSavingRawData(const char*)));
    //connect(controlbox->save_frames_button,SIGNAL(clicked()),fw,SLOT(startSavingDSFData(const char*)));
    //connect(controlbox->save_frames_button,SIGNAL(clicked()),fw,SLOT(startSavingSTD_DEVData(const char*)));
    connect(controlbox, SIGNAL(startSaving(const char*)),fw,SLOT(startSavingRawData(const char*)));
    connect(controlbox->stop_saving_frames_button,SIGNAL(clicked()),fw,SLOT(stopSavingRawData()));
    //start worker Thread
    workerThread->start();

}

void MainWindow::destroyBackend()
{
    qDebug() << "attempting to stop backend";
    disconnect(fw,SIGNAL(newFrameAvailable()), unfiltered_widget, SLOT(handleNewFrame()));
    disconnect(fw,SIGNAL(newFrameAvailable()), dsf_widget, SLOT(handleNewFrame()));
    disconnect(fw,SIGNAL(newFrameAvailable()), std_dev_widget, SLOT(handleNewFrame()));
    disconnect(controlbox->collect_dark_frames_button,SIGNAL(clicked()),fw,SLOT(startCapturingDSFMask()));
    disconnect(controlbox->stop_dark_collection_button,SIGNAL(clicked()),fw,SLOT(finishCapturingDSFMask()));
    disconnect(controlbox,SIGNAL(mask_selected(const char *)),fw,SLOT(loadDSFMask(const char *)));
    delete fw;
    delete workerThread;
}

MainWindow::~MainWindow()
{

}
void MainWindow::testslot()
{
    qDebug() << "test slot hit";
}

void MainWindow::updateFPS(unsigned int fps)
{
    //controlbox->fps_label->setText();
}
