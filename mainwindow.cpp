#include <QVBoxLayout>
#include <QDebug>
#include "mainwindow.h"
#include "image_type.h"
#include <memory>
MainWindow::MainWindow(QThread *qth, frameWorker *fw, QWidget *parent)
    : QMainWindow(parent)
{
    qRegisterMetaType<frame_c*>("frame_c*");
    qRegisterMetaType<QVector<double>>("QVector<double>");
    qRegisterMetaType<QSharedPointer<QVector<double>>>("QSharedPointer<QVector<double>>");


    this->fw = fw;
    //start worker Thread
    qth->start();
    qDebug() << "fw passed to MainWindow";
    this->resize(1440,900);
    mainwidget = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout;

    //Create tabs
    tabWidget = new QTabWidget;

    //NOTE: Care should be taken to ensure that tabbed widgets are ordered by the value of their image_type enum
    //signals/slots (currentChanged) make use of this relation
    unfiltered_widget = new frameview_widget(fw, BASE);
    dsf_widget = new frameview_widget(fw, DSF);
    std_dev_widget = new frameview_widget(fw, STD_DEV);
    hist_widget = new histogram_widget(fw,STD_DEV_HISTOGRAM);

    vert_widget = new mean_profile_widget(fw,VERTICAL_MEAN);
    horiz_widget = new mean_profile_widget(fw,HORIZONTAL_MEAN);
    fft_mean_widget = new fft_widget(fw,FFT_MEAN);

    tabWidget->addTab(unfiltered_widget, QString("Live View"));
    tabWidget->addTab(dsf_widget, QString("Dark Subtraction"));
    tabWidget->addTab(std_dev_widget, QString("Std. Deviation"));
    tabWidget->addTab(hist_widget,QString("Histogram View"));
    tabWidget->addTab(vert_widget,QString("Vertical Mean Profile"));
    tabWidget->addTab(horiz_widget,QString("Horizontal Mean Profile"));
    tabWidget->addTab(fft_mean_widget,QString("FFT of Plane Mean"));

    layout->addWidget(tabWidget,3);
    //Create controls box

    controlbox = new ControlsBox(fw,tabWidget);
    layout->addWidget(controlbox,1);


    mainwidget->setLayout(layout);


    this->setCentralWidget(mainwidget);

    //Connect everything together

    connect(tabWidget,SIGNAL(currentChanged(int)),controlbox,SLOT(tabChangedSlot(int)));
    controlbox->tabChangedSlot(0);

    connect(fw,SIGNAL(newFrameAvailable()), unfiltered_widget, SLOT(handleNewFrame()));

    //connect(fw,SIGNAL(newFrameAvailable(frame_c *)), unfiltered_widget, SLOT(handleNewFrame(frame_c *)));
    //connect(fw,SIGNAL(newFrameAvailable(frame_c *)), dsf_widget, SLOT(handleNewFrame(frame_c *)));

   // connect(fw,SIGNAL(stdDevFrameCompleted(frame_c *)), std_dev_widget, SLOT(handleNewFrame(frame_c *)));

   // connect(fw,SIGNAL(newFrameAvailable(frame_c *)),vert_widget,SLOT(handleNewFrame(frame_c *)));
    //connect(fw,SIGNAL(newFrameAvailable(frame_c *)),horiz_widget,SLOT(handleNewFrame(frame_c *)));

   // connect(fw,SIGNAL(newFFTMagAvailable(QSharedPointer<QVector<double> >)),fft_mean_widget,SLOT(handleNewFrame(QSharedPointer<QVector<double> >)));
   // connect(fw,SIGNAL(newStdDevHistogramAvailable(QSharedPointer<QVector<double> >)),hist_widget,SLOT(handleNewFrame(QSharedPointer<QVector<double> >)));



    connect(controlbox,SIGNAL(startDSFMaskCollection()),fw,SLOT(startCapturingDSFMask()));
    connect(controlbox,SIGNAL(stopDSFMaskCollection()),fw,SLOT(finishCapturingDSFMask()));
    //connect(controlbox,SIGNAL(mask_selected(const char *)),fw,SLOT(loadDSFMask(const char *)));
    connect(&controlbox->useDSFCbox,SIGNAL(toggled(bool)),fw,SLOT(toggleUseDSF(bool)));

    connect(controlbox,SIGNAL(startSavingFinite(unsigned int,QString)),fw,SLOT(startSavingRawData(unsigned int,QString)));
    connect(controlbox,SIGNAL(stopSaving()),fw,SLOT(stopSavingRawData()));

    connect(&controlbox->std_dev_N_slider,SIGNAL(valueChanged(int)),fw,SLOT(setStdDev_N(int)));

    connect(fw,SIGNAL(savingFrameNumChanged(unsigned int)),controlbox,SLOT(updateSaveFrameNum_slot(unsigned int)));
    //connect(fw,SIGNAL(savingFrameNumChanged(uint)),&controlbox,SLOT(updateSaveFrameNum_slot(uint)));
    controlbox->fps_label.setText("Running");
    controlbox->fps_label.setStyleSheet("{color: green;}");




}


MainWindow::~MainWindow()
{
}
void MainWindow::testslot(int val)
{
    qDebug() << "test slot hit";
}

void MainWindow::updateFPS(unsigned int fps)
{
    //controlbox->fps_label->setText();
}
void MainWindow::enableStdDevTabs()
{

    qDebug() << "enabling std. dev. tabs";
    tabWidget->setTabEnabled(2,true);
    tabWidget->setTabEnabled(3,true);
    disconnect(fw,SIGNAL(std_dev_ready()),this,SLOT(enableStdDevTabs()));

}
