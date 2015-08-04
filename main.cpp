/* Qt includes */
#include <QApplication>
#include <QDesktopWidget>
#include <QPixmap>
#include <QSplashScreen>
#include <QStyle>
#include <QThread>
#include <QTime>

/* liveview includes */
#include "mainwindow.h"
#include "qcustomplot.h"
#include "frame_worker.h"

/*! If the macros to define the development environment are not defined at compile time, use defaults */
#ifndef HOST
#define HOST "unknown location"
#endif
/*! If the macros to define the version author are not defined at compile time, use defaults */
#ifndef UNAME
#define UNAME "unknown person"
#endif

/*! \file */
/*! \mainpage  \header View live plots of focal plane data
 * Live View is a Qt frontend GUI for cuda_take, it displays focal plane data and basic analysis
 * (such as the std. dev, dark subtraction, FFT, Spectral Profile, and Video Savant-like playback). Plots are
 * implemented using the QCustomPlot (http://www.qcustomplot.com) library, which generates live color maps, bar graphs,
 * and line graphs within the Qt C++ environment.
 * \paragraph
 *
 * Live View is designed to be sufficiently modular that it will plot any data with known geometry, up to a maximum word size
 * of 16 bits. To implement new hardware or modify existing parameters, changes must be made to the backend cuda_take software.
 *
 * \author This documentation and most comments in Live View and cuda_take were written by JP Ryan
 */

int main(int argc, char *argv[])
{
    /*! Step 1: Setup this QApplication */
    QApplication::setGraphicsSystem("raster"); //This is intended to make 2D rendering faster
    QApplication a(argc, argv);

    /*! Step 2: Load the splash screen */
    QPixmap logo_pixmap(":images/aviris-logo-transparent.png");
    QSplashScreen splash(logo_pixmap);
    splash.show();
    splash.showMessage(QObject::tr("Loading AVIRIS-Next Generation LiveView2. Compiled on " __DATE__ ", " __TIME__ " PDT by " UNAME "@" HOST  ),
                       Qt::AlignCenter | Qt::AlignBottom , Qt::gray);

    /*! Step 3: Load the parallel worker object which will act as a "backend" for Live View */
    frameWorker * fw = new frameWorker();
    QThread * workerThread = new QThread();
    fw->moveToThread(workerThread);
    QObject::connect(workerThread,SIGNAL(started()),fw,SLOT(captureFrames()));

    /*! Step 4: Display version author message in the console */
    std::cout << "This version of liveview2 was compiled on " << __DATE__ << " at " << __TIME__<< " using gcc " << __GNUC__ << std::endl;
    std::cout << "The compilation was performed by " << UNAME << " @ " << HOST << std::endl;

    /*! Step 5: Open the main window (GUI/frontend) */
    MainWindow w(workerThread, fw);
    w.setGeometry(   QStyle::alignedRect(
                         Qt::LeftToRight,
                         Qt::AlignCenter,
                         w.size(),
                         a.desktop()->availableGeometry()
                         ));
    QPixmap icon_pixmap(":images/icon.png");
    w.setWindowIcon(QIcon(icon_pixmap));
    w.show();

    splash.finish(&w);
    /*! Step 6: Close out the backend after the frontend is closed */
    int retval = a.exec();
#ifdef VERBOSE
    qDebug() << "Goodbye!";
#endif
    fw->stop();

    delete fw;
#ifdef VERBOSE
    qDebug() << "about to exit workerThread";
#endif
    workerThread->exit(0);
#ifdef VERBOSE
    qDebug() << "waiting on workerThread";
#endif
    workerThread->wait();
#ifdef VERBOSE
    qDebug() << "deleting on workerThread";
#endif
    delete workerThread;

    /*! There is a QCustomPlot component in frameview_widget which causes a segmentation violation. This bug is very difficult to fix... */
    qDebug() << "Liveview2 will now finish unexpectedly!";
    return retval;
}
