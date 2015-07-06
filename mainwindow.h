#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "controlsbox.h"
#include "frameview_widget.h"
#include "histogram_widget.h"
#include "profile_widget.h"
#include "fft_widget.h"
#include "saveserver.h"
#include "frame_c_meta.h"


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QThread * qth, frameWorker * fw, QWidget *parent = 0);
    ~MainWindow();

private:
    frameWorker *fw;
    QTabWidget * tabWidget;
    QWidget * mainwidget;
    ControlsBox * controlbox;
    saveServer* save_server;

    frameview_widget* unfiltered_widget;
    frameview_widget* dsf_widget;
    frameview_widget* std_dev_widget;
    histogram_widget* hist_widget;
    profile_widget* vert_mean_widget;
    profile_widget* horiz_mean_widget;
    profile_widget* vert_cross_widget;
    profile_widget* horiz_cross_widget;
    fft_widget* fft_mean_widget;

public slots:
    //void updateFPS(unsigned int fps);
    //void testslot(int val);

    void enableStdDevTabs();
};

#endif // MAINWINDOW_H
