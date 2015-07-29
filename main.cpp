// Qt includes
#include <QApplication>
#include <QDesktopWidget>
#include <QPixmap>
#include <QSplashScreen>
#include <QStyle>
#include <QThread>
#include <QTime>

// liveview includes
#include "mainwindow.h"
#include "qcustomplot.h"
#include "frame_worker.h"

// If the macros to define the version author are not defined at compile time, use defaults
#ifndef HOST
#define HOST "unknown location"
#endif

#ifndef UNAME
#define UNAME "unknown person"
#endif

int main(int argc, char *argv[])
{
    QApplication::setGraphicsSystem("raster"); //This is intended to make 2D rendering faster
    QApplication a(argc, argv);

    // Load the splash screen
    QPixmap logo_pixmap(":images/aviris-logo-transparent.png");
    QSplashScreen splash(logo_pixmap);
    splash.show();
    splash.showMessage(QObject::tr("Loading AVIRIS-Next Generation Live View2..."),
                       Qt::AlignCenter | Qt::AlignBottom , Qt::gray);

    // Display version author message
    std::cout << "This version of liveview2 was compiled on " << __DATE__ << " at " << __TIME__<< " using gcc " << __GNUC__ << std::endl;
    std::cout << "The compilation was performed by " << UNAME << " @ " << HOST << std::endl;

    frameWorker * fw = new frameWorker();
    QThread * workerThread = new QThread();
    fw->moveToThread(workerThread);
    QObject::connect(workerThread,SIGNAL(started()),fw,SLOT(captureFrames()));

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

    qDebug() << "Liveview2 will now finish unexpectedly!";
    return retval;
}
