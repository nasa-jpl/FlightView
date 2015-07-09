#include <QApplication>
#include <QPixmap>
#include <QSplashScreen>
#include <QDesktopWidget>
#include <QTime>
#include <QStyle>
#include "mainwindow.h"
#include "qcustomplot.h"
#include "frame_worker.h"
#include <QThread>

#ifndef HOST
#define HOST "unknown location"
#endif

#ifndef UNAME
#define UNAME "unknown person"
#endif

// void delay(int);
int main(int argc, char *argv[])
{
    QApplication::setGraphicsSystem("raster"); //This is intended to make 2D rendering faster
    //    QApplication::setGraphicsSystem("opengl"); //This was not working

    QApplication a(argc, argv);
    QPixmap logo_pixmap(":images/aviris-logo-transparent.png");
    QSplashScreen splash(logo_pixmap);
    splash.show();
    splash.showMessage(QObject::tr("Loading AVIRIS-Next Generation Live View2..."),
                       Qt::AlignCenter | Qt::AlignBottom , Qt::gray);
    frameWorker * fw = new frameWorker();

    QThread * workerThread = new QThread();
    //frameWorker * fw = new frameWorker();

    fw->moveToThread(workerThread);
    QObject::connect(workerThread,SIGNAL(started()),fw,SLOT(captureFrames()));
    //delay(500);

    std::cout << "This version of liveview2 was compiled on " << __DATE__ << " at " << __TIME__<< " using gcc " << __GNUC__ << std::endl;
    std::cout << "The compilation was performed by " << UNAME << "  @ " << HOST << std::endl;
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
    //qDebug() << "Goodbye!";
    fw->stop();


    delete fw;
    //qDebug() << "about to exit workerThread";
    workerThread->exit(0);
    //qDebug() << "waiting on workerThread";
    workerThread->wait();
    //qDebug() << "deleting on workerThread";

    delete workerThread;
    qDebug() << "Liveview2 will now finish unexpectedly!";

    return retval;
}

/*
void delay( int millisecondsToWait )
{
    QTime dieTime = QTime::currentTime().addMSecs( millisecondsToWait );
    while( QTime::currentTime() < dieTime )
    {
        QCoreApplication::processEvents( QEventLoop::AllEvents, 100 );
    }
}
*/
