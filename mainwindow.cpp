#include <QVBoxLayout>
#include <QDebug>
#include "mainwindow.h"
#include "image_type.h"
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{

    workerThread = new QThread();
    fw = new frameWorker();

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


    fw->moveToThread(workerThread);
    connect(workerThread,SIGNAL(started()), fw, SLOT(captureFrames()));

    connect(fw,SIGNAL(newFrameAvailable()), unfiltered_widget, SLOT(handleNewFrame()));
    connect(fw,SIGNAL(newFrameAvailable()), dsf_widget, SLOT(handleNewFrame()));
    connect(controlbox->collect_dark_frames_button,SIGNAL(clicked()),this,SLOT(testslot()));


    connect(controlbox->collect_dark_frames_button,SIGNAL(clicked()),fw,SLOT(startCapturingDSFMask()));
    connect(controlbox->stop_dark_collection_button,SIGNAL(clicked()),fw,SLOT(finishCapturingDSFMask()));
    connect(controlbox,SIGNAL(mask_selected(const char *)),fw,SLOT(loadDSFMask(const char *)));
    connect(unfiltered_widget->toggleGrayScaleButton,SIGNAL(clicked()),unfiltered_widget,SLOT(toggleGrayScale()));

    connect(controlbox->ceiling_slider, SIGNAL(valueChanged(int)), dsf_widget, SLOT(updateCeiling(int)));
    connect(controlbox->floor_slider, SIGNAL(valueChanged(int)), dsf_widget, SLOT(updateFloor(int)));

    //start worker Thread
    workerThread->start();
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
