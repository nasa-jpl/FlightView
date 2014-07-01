#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "controlsbox.h"
#include "frameview_widget.h"
#include "histogram_widget.h"
#include "mean_profile_widget.h"

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
    histogram_widget * hist_widget;

    mean_profile_widget * vert_widget;
    mean_profile_widget * horiz_widget;
public slots:
    void updateFPS(unsigned int fps);
    void testslot(int val);
    void createBackend();
    void destroyBackend();
    void connectAndStartBackend();
    void enableStdDevTabs();
};

#endif // MAINWINDOW_H
