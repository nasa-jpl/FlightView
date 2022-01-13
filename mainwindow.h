#ifndef MAINWINDOW_H
#define MAINWINDOW_H

/* Qt GUI incude */
#include <QMainWindow>

/* Live View includes */
#include "controlsbox.h"
#include "fft_widget.h"
#include "frame_c_meta.h"
#include "frameview_widget.h"
#include "flight_widget.h"
#include "histogram_widget.h"
#include "profile_widget.h"
#include "playback_widget.h"
#include "saveserver.h"
#include "startupOptions.h"
#include "preferences.h"

// GPS includes:

#include "gpsGUI/gpsbinaryreader.h"

/*! \file
 * \brief The main viewing window for Live View.
 */

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    /*! The main window must be passed a QThread to ensure that GUI commands are handled
     * separately from backend commands. This improves the overall responsiveness of the software. */
    MainWindow(startupOptionsType options, QThread *qth, frameWorker *fw, QWidget *parent = 0);

private:
    frameWorker *fw;
    QTabWidget *tabWidget;
    QWidget *mainwidget;
    ControlsBox *controlbox;
    saveServer *save_server; // Save Server is a non-GUI component that should be open regardless of the current view widget

    startupOptionsType options;
    QString settingsFilename;

    void prepareGPS();
    void processGPSMessage();

    /*! All widgets currently used in Live View
     * @{ */
    frameview_widget *unfiltered_widget;
    frameview_widget *dsf_widget;
    frameview_widget *waterfall_widget;
    flight_widget *flight_screen;
    frameview_widget *std_dev_widget;
    histogram_widget *hist_widget;
    profile_widget *vert_mean_widget;
    profile_widget *horiz_mean_widget;
    profile_widget *vert_cross_widget;
    profile_widget *horiz_cross_widget;
    profile_widget *vert_overlay_widget;
    fft_widget *fft_mean_widget;
    playback_widget *raw_play_widget;

    /*! @} */

public slots:
    void handleStatusMessage(QString message);
    void enableStdDevTabs();
    void debugThis();

signals:


protected:
    /*! \brief Defines keyboard controls for all components of Live View */
    void keyPressEvent(QKeyEvent *c);
    void closeEvent(QCloseEvent *e);

};

#endif // MAINWINDOW_H
