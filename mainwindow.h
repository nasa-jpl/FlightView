#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "controlsbox.h"
#include "frameview_widget.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();
private:
    QThread * workerThread;
    frameWorker *fw;
    QTabWidget * tabWidget;
    QWidget * mainwidget;
    ControlsBox * controlbox;
    frameview_widget * unfiltered_widget;
    frameview_widget * dsf_widget;
    frameview_widget * std_dev_widget;
public slots:
    void updateFPS(unsigned int fps);
    void testslot();
    void createBackend();
    void destroyBackend();
    void connectAndStartBackend();
};

#endif // MAINWINDOW_H
