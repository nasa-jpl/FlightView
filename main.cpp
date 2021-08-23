/* Qt includes */
#include <QApplication>
#include <QDesktopWidget>
#include <QPixmap>
#include <QSplashScreen>
#include <QStyle>
#include <QThread>
#include <QTime>

#include <cstdio>
#include <QDebug>

/* LiveView includes */
#include "mainwindow.h"
#include "qcustomplot.h"
#include "frame_worker.h"
#include "startupOptions.h"

/* If the macros to define the development environment are not defined at compile time, use defaults */
#ifndef HOST
#define HOST "unknown location"
#endif
/* If the macros to define the version author are not defined at compile time, use defaults */
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
 * \author This documentation and comments in Live View and cuda_take were written by Jackie Ryan
 */

int main(int argc, char *argv[])
{
    /* Step 1: Setup this QApplication */
    //QApplication::setGraphicsSystem("raster"); //This is intended to make 2D rendering faster
    QApplication a(argc, argv);

    QString cmdName = QString("%1").arg(argv[0]);
    QString helptext = QString("\nUsage: %1 -d --debug, -f --flight --no-gps "
                               "--no-camera --datastoragelocation /path/to/storage --gpsIP 10.0.0.6 "
                               "--deviceIHE /dev/ihe --deviceFPIED /dev/fpied")\
            .arg(cmdName);
    QString currentArg;
    startupOptionsType startupOptions;
    startupOptions.debug = false;
    startupOptions.flightMode = false;
    startupOptions.disableCamera = false;
    startupOptions.disableGPS = false;
    startupOptions.dataLocation = QString("/data");
    startupOptions.gpsIP = QString("10.0.0.6");
    startupOptions.gpsPort = 8111;
    startupOptions.gpsPortSet = true;
    startupOptions.deviceFPIED = QString("/dev/fpied");
    startupOptions.deviceIHE = QString("/dev/ihe");

    // Basic CLI argument parser:
    for(int c=1; c < argc; c++)
    {
        currentArg = QString(argv[c]).toLower();

        if(currentArg == "-d" || currentArg == "--debug")
        {
            startupOptions.debug = true;
        }
        if(currentArg == "-f" || currentArg == "--flight")
        {
            startupOptions.flightMode = true;
        }
        if(currentArg == "--no-gps")
        {
            startupOptions.disableGPS = true;
        }
        if(currentArg == "--no-camera")
        {
            startupOptions.disableCamera = true;
        }
        if(currentArg == "--datastoragelocation")
        {
            if(argc > c)
            {
                startupOptions.dataLocation = argv[c+1];
                startupOptions.dataLocationSet = true;
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }
        if(currentArg == "--gpsip")
        {
            // Only IPV4 supported, and no hostnames please, let's not depend upon DNS or resolv in the airplane...
            if((argc > c) && QString(argv[c+1]).contains(QRegularExpression("(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})")))
            {
                startupOptions.gpsIP = argv[c+1];
                startupOptions.gpsIPSet = true;
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }
        if(currentArg == "--deviceihe")
        {
            // Use UDEV rules to give nice names
            if((argc > c) && QString(argv[c+1]).contains("/dev"))
            {
                startupOptions.deviceIHE = argv[c+1];
                startupOptions.deviceIHESet = true;
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }
        if(currentArg == "--devicefpied")
        {
            // Use UDEV rules to give nice names
            if((argc > c) && QString(argv[c+1]).contains("/dev"))
            {
                startupOptions.deviceFPIED = argv[c+1];
                startupOptions.deviceFPIEDSet = true;
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }
        if(currentArg == "--help" || currentArg == "-h" || currentArg.contains("?"))
        {
            std::cout << helptext.toStdString() << std::endl;
            exit(-1);
        }

    }



    qDebug() << "test";

    /* Step 2: Load the splash screen */
    QPixmap logo_pixmap(":images/aviris-logo-transparent.png");
    QSplashScreen splash(logo_pixmap);
    splash.show();
    splash.showMessage(QObject::tr("Loading AVIRIS-Next Generation LiveView. Compiled on " __DATE__ ", " __TIME__ " PDT by " UNAME "@" HOST  ),
                       Qt::AlignCenter | Qt::AlignBottom, Qt::gray);

    /* Step 3: Load the parallel worker object which will act as a "backend" for LiveView */
    frameWorker *fw = new frameWorker(startupOptions);
    QThread *workerThread = new QThread();
    fw->moveToThread(workerThread);
    QObject::connect(workerThread, SIGNAL(started()), fw, SLOT(captureFrames()));

    /* Step 4: Display version author message in the console */
    std::cout << "This version of LiveView was compiled on " << __DATE__ << " at " << __TIME__<< " using gcc " << __GNUC__ << std::endl;
    std::cout << "The compilation was performed by " << UNAME << " @ " << HOST << std::endl;

    /* Step 5: Open the main window (GUI/frontend) */
    MainWindow w(startupOptions, workerThread, fw);
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
    /* Step 6: Close out the backend after the frontend is closed */
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

    return retval;
}
