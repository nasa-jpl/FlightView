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
#include <stdlib.h>

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

// These are here for the purpose of easily finding the strings within the binary.
const volatile char* COMPILE_INFO_STR =     "---------- Compile-time information (CTIF) string: ----------";
const volatile char* DATE_COMPILE_STR =     "  CTIF      Compile Date: " __DATE__;
const volatile char* GIT_BRANCH_STR =       "  CTIF      Git Branch: " GIT_BRANCH;
const volatile char* GIT_CURRENT_SHA1_STR = "  CTIF      Git SHA1: " GIT_CURRENT_SHA1;
const volatile char* SRC_DIR_STR =          "  CTIF      Source Directory: " SRC_DIR;
const volatile char* COMPILE_INFO_END_STR = "-------------------------------------------------------------";


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
    QString helptext = QString("\nUsage: %1 -d --debug, -f --flight --no-gps \n"
                               "--no-camera --datastoragelocation /path/to/storage --gpsIP 10.0.0.6 \n"
                               "--gpsport 5661 \n"
                               "--no-stddev --xiocam --rtpcam \n"
                               "--rtpnextgen \n"
                               "--rtpheight 480 \n"
                               "--rtpwidth 1280 \n"
                               "--rtpaddress 1.2.3.4 \n"
                               "--rtpinterface eth2 \n"
                               "--er2 --headless \n"
                               "--wfpreview \n"
                               "--wfpreviewcontinuous \n"
                               "--wfpreviewlocation /path/to/waterfallpreview/files/ \n"
                               "-v --version \n"
                               "--rotate \n"
                               "--remap \n"
                               "--darkreffile /path/to/dark_file.raw (uint16 frames)\n"
                               )\
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
        if(currentArg == "-v" || currentArg == "--version")
        {
            printf("%s\n", COMPILE_INFO_STR);
            printf("%s\n", DATE_COMPILE_STR);
            printf("%s\n", GIT_BRANCH_STR);
            printf("%s\n", GIT_CURRENT_SHA1_STR);
            printf("  Link to commit: https://github.com/nasa-jpl/LiveViewLegacy/tree/" GIT_CURRENT_SHA1_SHORT "\n");
            printf("%s\n", SRC_DIR_STR);
            printf("%s\n", COMPILE_INFO_END_STR);
            exit(0);
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
        if(currentArg == "--er2") {
            startupOptions.er2mode = true;
        }
        if(currentArg == "--headless") {
            startupOptions.headless = true;
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
        if( (currentArg == "--darkreffile") || (currentArg == "--darkreferencefile") ) {
            if(argc > c)
            {
                startupOptions.darkReferenceFileLocation = argv[c+1];
                startupOptions.darkRefFileSet = true;
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
        if(currentArg == "--rtpnextgen") {
            startupOptions.rtpNextGen = true;
        }

        if(currentArg == "--rtprgb") {
            startupOptions.rtprgb = true;
        }

        if( (currentArg == "--rtpgray") || (currentArg == "--rtpgrey") ) {
            startupOptions.rtprgb = false;
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
                startupOptions.havertpInterface = true;
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
                startupOptions.havertpAddress = true;
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
        if(currentArg == "--laggy") {
            startupOptions.laggy = true;
            std::cout << "WARNING, laggy mode enabled." << std::endl;
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

        if( (currentArg == "--no-stddev") || (currentArg == "--no-stdev")
                || (currentArg == "--nostdev") || (currentArg == "--nostddev") )
        {
            startupOptions.runStdDevCalculation = false;
        }

        if( (currentArg == "--shm")) {
            startupOptions.useSHM = true;
        }

        if( (currentArg == "--no-gpu") || (currentArg == "--nogpu") ) {
            startupOptions.noGPU = true;
            startupOptions.runStdDevCalculation = false;
        }

        if( currentArg == "--rotate") {
            startupOptions.rotate = true;
        }

        if( currentArg == "--remap") {
            startupOptions.remapPixels = true;
        }

        if(currentArg == "--wfpreview") {
            startupOptions.wfPreviewEnabled = true;
        }
        if(currentArg == "--wfpreviewcontinuous") {
            startupOptions.wfPreviewContinuousMode = true;
            startupOptions.wfPreviewEnabled = true;
        }
        if(currentArg == "--wfpreviewlocation") {
            if(argc > c)
            {
                startupOptions.wfPreviewLocation = argv[c+1];
                startupOptions.wfPreviewlocationset = true;
                c++;
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
            std::cout << "rtpAddress:   " << startupOptions.rtpAddress << std::endl;

        if(startupOptions.rtpNextGen) {
            std::cout << "Using RTP NextGen code" << std::endl;
        }

        if(widthSet && heightSet)
        {
            startupOptions.heightWidthSet = true;
        } else {
            startupOptions.heightWidthSet = false;
        }
    }

    if(startupOptions.flightMode && !startupOptions.dataLocationSet)
    {
        system("xmessage \"Error, flight mode requires --datastoragelocation\"");
        std::cerr << "Error, flight mode specified without data storage location." << std::endl;
        std::cout << helptext.toStdString() << std::endl;
        exit(-1);
    }

    if(startupOptions.flightMode && startupOptions.dataLocation.isEmpty())
    {
        system("xmessage \"Error, flight mode requires --datastoragelocation with valid path set\"");
        std::cerr << "Error, flight mode specified without complete data storage location." << std::endl;
        std::cout << helptext.toStdString() << std::endl;
        exit(-1);
    }


    if(startupOptions.flightMode && !startupOptions.gpsIPSet && !startupOptions.disableGPS)
    {
        system("xmessage \"Error, flight mode requires --gpsip with GPS ip address specified.\"");
        std::cerr << "Error, flight mode specified without GPS IP address." << std::endl;
        std::cout << helptext.toStdString() << std::endl;
        exit(-1);
    }

    if(startupOptions.flightMode && startupOptions.disableGPS)
    {
        std::cout << "WARNING:, flight mode specified with disabled GPS." << std::endl;
    }

    if( startupOptions.rtpCam && ((int)heightSet + (int)widthSet != 2))
    {
        // XOR, only one was set
        system("xmessage \"Error, RTP requires both height and width to be specified.\"");
        std::cerr << "Error, RTP mode requires both height and width to be specified." << std::endl;
        std::cout << helptext.toStdString() << std::endl;
        exit(-1);
    }

    if(heightSet && widthSet)
    {
        startupOptions.heightWidthSet = true;
    }


    if(startupOptions.wfPreviewEnabled && (!startupOptions.wfPreviewlocationset)) {
        std::cerr << "Warning, waterfall preview option enabled but --wfpreviewlocation was not set." << std::endl;
    }

    /* Step 2: Load the splash screen */

    QString logoPath;
    if(startupOptions.flightMode)
    {
        logoPath = ":images/aviris3-logo.png";
    } else {
        logoPath = ":images/aviris-logo-transparent.png";
    }

    QPixmap logo_pixmap(logoPath);

    QSplashScreen *splash = new QSplashScreen(logo_pixmap);
    if(! (startupOptions.xioCam || startupOptions.headless))
    {
        // On some displays, the splash screen covers the setup dialog box
        splash->show();
        splash->showMessage(QObject::tr(" "),
                           Qt::WindowStaysOnTopHint | Qt::AlignCenter | Qt::AlignBottom, Qt::black);
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
    if(!startupOptions.headless) {
        w.setGeometry(   QStyle::alignedRect(
                             Qt::LeftToRight,
                             Qt::AlignCenter,
                             w.size(),
                             QGuiApplication::screens().at(0)->availableGeometry()
                             ));
    }
    QPixmap icon_pixmap(":images/icon.png");
    w.setWindowIcon(QIcon(icon_pixmap));
    if(! (startupOptions.xioCam || startupOptions.headless))
        splash->raise();
    w.show();
    if(! (startupOptions.xioCam || startupOptions.headless))
        splash->raise();


    splash->finish(&w);
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
