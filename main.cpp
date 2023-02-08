/* Qt includes */
#include <QApplication>
#include <QDesktopWidget>
#include <QPixmap>
#include <QSplashScreen>
#include <QStyle>
#include <QThread>
#include <QTime>
#include <QTimer>

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
    /* Step 0: Set environment variables: */

#ifndef QT_DEBUG
    (void)putenv((char*)"QT_LOGGING_RULES=*=false");
#endif

    // If the GUI does not scale correctly,
    // this may be enabled to fix it.
    // (void)putenv((char*)"QT_AUTO_SCREEN_SCALE_FACTOR=1");


    /* Step 1: Setup this QApplication */
    //QApplication::setGraphicsSystem("raster"); //This is intended to make 2D rendering faster
    QApplication a(argc, argv);
    a.setOrganizationDomain("jpl.nasa.gov");
    a.setOrganizationName("FlightView");
    a.setApplicationDisplayName("FlightView");
    a.setApplicationName("FlightView");

    QString cmdName = QString("%1").arg(argv[0]);
    QString helptext = QString("\nUsage: %1 -d --debug, -f --flight --no-gps "
                               "--no-camera --datastoragelocation /path/to/storage --gpsIP 10.0.0.6 "
                               "--gpsport 5661 "
                               "--no-stddev --xiocam --rtpcam")\
            .arg(cmdName);
    QString currentArg;
    startupOptionsType startupOptions;
    startupOptions.debug = false;
    startupOptions.flightMode = false;
    startupOptions.disableCamera = false;
    startupOptions.xioCam = false;
    startupOptions.rtpCam = false;
    startupOptions.disableGPS = false;
    startupOptions.dataLocation = QString("/data");
    startupOptions.gpsIP = QString("10.0.0.6");
    startupOptions.gpsPort = 8111;
    startupOptions.gpsPortSet = true;
    //startupOptions.xioDirectory = new QString("");
    startupOptions.xioDirectoryArray = (char*)calloc(4096, sizeof(char));
    if(startupOptions.xioDirectoryArray == NULL)
        abort();

    startupOptions.targetFPS = 50.0;

    bool heightSet = false;
    bool widthSet = false;
    bool rtpInterfaceSet = false;
    bool rtpAddressSet = false;

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
                c++;
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }

        if(currentArg == "--rtpcam")
        {
            startupOptions.rtpCam = true;
        }

        if(currentArg == "--rtpheight")
        {
            if(argc > c)
            {
                int rtpheighttemp = 0;
                bool ok = false;
                rtpheighttemp = QString(argv[c+1]).toUInt(&ok);
                if(ok)
                {
                    heightSet = true;
                    startupOptions.rtpHeight = rtpheighttemp;
                    c++;
                } else {
                    std::cout << helptext.toStdString() << std::endl;
                    exit(-1);
                }
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }

        if(currentArg == "--rtpwidth")
        {
            if(argc > c)
            {
                int rtpwidthtemp = 0;
                bool ok = false;
                rtpwidthtemp = QString(argv[c+1]).toUInt(&ok);
                if(ok)
                {
                    widthSet = true;
                    startupOptions.rtpWidth = rtpwidthtemp;
                    c++;
                } else {
                    std::cout << helptext.toStdString() << std::endl;
                    exit(-1);
                }
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }

        if(currentArg == "--rtpinterface")
        {
            if(argc > c)
            {
                startupOptions.rtpInterface = argv[c+1];
                rtpInterfaceSet = true;
                c++;
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }

        if(currentArg == "--rtpaddress")
        {
            if(argc > c)
            {
                startupOptions.rtpAddress = argv[c+1];
                rtpAddressSet = true;
                c++;
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }

        if(currentArg == "--rtpport")
        {
            if(argc > c)
            {
                int rtpPortTemp = 0;
                bool ok=false;
                rtpPortTemp = QString(argv[c+1]).toUInt(&ok);
                if(ok)
                {
                    startupOptions.rtpPort = rtpPortTemp;
                    c++;
                } else {
                    std::cout << helptext.toStdString() << std::endl;
                    exit(-1);
                }
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }

        //

        if(currentArg == "--xiocam")
        {
            startupOptions.xioCam = true;
        }

        if(currentArg == "--xioheight")
        {
            if(argc > c)
            {
                int xioheighttemp = 0;
                bool ok = false;
                xioheighttemp = QString(argv[c+1]).toUInt(&ok);
                if(ok)
                {
                    heightSet = true;
                    startupOptions.xioHeight = xioheighttemp;
                    c++;
                } else {
                    std::cout << helptext.toStdString() << std::endl;
                    exit(-1);
                }
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }

        if(currentArg == "--xiowidth")
        {
            if(argc > c)
            {
                int xiowidthtemp = 0;
                bool ok = false;
                xiowidthtemp = QString(argv[c+1]).toUInt(&ok);
                if(ok)
                {
                    widthSet = true;
                    startupOptions.xioWidth = xiowidthtemp;
                    c++;
                } else {
                    std::cout << helptext.toStdString() << std::endl;
                    exit(-1);
                }
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }
        if(currentArg == "--targetfps")
        {
            if(argc > c)
            {
                float targetfpstemp = 0;
                bool ok = false;
                targetfpstemp = QString(argv[c+1]).toFloat(&ok);
                if(ok)
                {
                    startupOptions.targetFPS = targetfpstemp;
                    c++;
                } else {
                    std::cout << helptext.toStdString() << std::endl;
                    exit(-1);
                }
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
                c++;
            } else {
                std::cout << helptext.toStdString() << std::endl;
                exit(-1);
            }
        }
        if(currentArg == "--gpsport")
        {
            if(argc > c)
            {
                int portTemp=0;
                bool ok = false;
                portTemp = QString(argv[c+1]).toInt(&ok);
                if(ok)
                {
                    startupOptions.gpsPort = portTemp;
                    startupOptions.gpsPortSet = true;
                    c++;
                } else {
                    std::cerr << "Invalid GPS Port set." << std::endl;
                    std::cout << helptext.toStdString() << std::endl;
                    exit(-1);
                }
            }
        }

        if(currentArg == "--no-stddev")
        {
            startupOptions.runStdDevCalculation = false;
        }
        if(currentArg == "--help" || currentArg == "-h" || currentArg.contains("?"))
        {
            std::cout << helptext.toStdString() << std::endl;
            exit(-1);
        }
    }

    if(startupOptions.rtpCam)
    {
        if(!rtpAddressSet)
            startupOptions.rtpAddress = NULL;
        if(!rtpInterfaceSet)
            startupOptions.rtpInterface = NULL;

        std::cout << "Debug output for RTP camera: \n";
        std::cout << "rtpHeight:    " << startupOptions.rtpHeight << std::endl;
        std::cout << "rtpWidth:     " << startupOptions.rtpWidth << std::endl;
        std::cout << "rtpPort:      " << startupOptions.rtpPort << std::endl;
        if(startupOptions.rtpInterface != NULL)
            std::cout << "rtpInterface: " << startupOptions.rtpInterface << std::endl;

        if(startupOptions.rtpAddress != NULL)
            std::cout << "rtpAddress:    " << startupOptions.rtpAddress << std::endl;
        if(widthSet && heightSet)
        {
            startupOptions.heightWidthSet = true;
        } else {
            startupOptions.heightWidthSet = false;
        }
    }

    if(startupOptions.flightMode && !startupOptions.dataLocationSet)
    {
        std::cerr << "Error, flight mode specified without data storage location." << std::endl;
        std::cout << helptext.toStdString() << std::endl;
        exit(-1);
    }

    if(startupOptions.flightMode && startupOptions.dataLocation.isEmpty())
    {
        std::cerr << "Error, flight mode specified without complete data storage location." << std::endl;
        std::cout << helptext.toStdString() << std::endl;
        exit(-1);
    }


    if(startupOptions.flightMode && !startupOptions.gpsIPSet && !startupOptions.disableGPS)
    {
        std::cerr << "Error, flight mode specified without GPS IP address." << std::endl;
        std::cout << helptext.toStdString() << std::endl;
        exit(-1);
    }

    if(startupOptions.flightMode && startupOptions.disableGPS)
    {
        std::cout << "WARNING:, flight mode specified with disabled GPS." << std::endl;
    }

    if(heightSet ^ widthSet)
    {
        // XOR, only one was set
        std::cerr << "Error, height and width must both be specified." << std::endl;
        exit(-1);
    }

    if(heightSet && widthSet)
    {
        startupOptions.heightWidthSet = true;
    }


    /* Step 2: Load the splash screen */


    QPixmap logo_pixmap(":images/aviris-logo-transparent.png");
    QSplashScreen splash(logo_pixmap);
    if(!startupOptions.xioCam)
    {
        // On some displays, the splash screen covers the setup dialog box
        splash.show();
        splash.showMessage(QObject::tr("Loading AVIRIS-Next Generation LiveView. Compiled on " __DATE__ ", " __TIME__ " PDT by " UNAME "@" HOST  ),
                           Qt::AlignCenter | Qt::AlignBottom, Qt::gray);
    }

    /* Step 3: Load the parallel worker object which will act as a "backend" for LiveView */
    frameWorker *fw = new frameWorker(startupOptions);
    QThread *workerThread = new QThread();
    fw->moveToThread(workerThread);
    QObject::connect(workerThread, SIGNAL(started()), fw, SLOT(captureFrames()));

    /* Step 4: Display version author message in the console */
    std::cout << "This version of LiveView was compiled on " << __DATE__ << " at " << __TIME__<< " using gcc " << __GNUC__ << std::endl;
    std::cout << "The compilation was performed by " << UNAME << " @ " << HOST << std::endl;

    /* Step 5: Open the main window (GUI/frontend) */
    MainWindow w(&startupOptions, workerThread, fw);
    //QObject::connect(fw, SIGNAL(sendStatusMessage(QString)), w, SLOT(handleStatusMessage(QString)));
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

    if(startupOptions.xioDirectoryArray != NULL)
        free(startupOptions.xioDirectoryArray);
    return retval;
}
