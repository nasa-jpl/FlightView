#include <QVBoxLayout>
#include "mainwindow.h"
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    this->resize(1324,830);
    mainwidget = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout;

    //Create tabs
    tabWidget = new QTabWidget;
    filtered_widget = new frameview_widget();
    unfiltered_widget = new frameview_widget();
    tabWidget->addTab(unfiltered_widget, QString("Live View"));
    tabWidget->addTab(filtered_widget, QString("Analytics"));

    layout->addWidget(tabWidget,3);
    //Create controls box

    controlbox = new ControlsBox();
    layout->addWidget(controlbox,1);


    mainwidget->setLayout(layout);


    this->setCentralWidget(mainwidget);
}

MainWindow::~MainWindow()
{
    
}
