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
void delay(int);
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    a.setGraphicsSystem("opengl"); //This is intended to make 2D rendering faster
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
    delay(500);
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
    qDebug() << "Goodbye!";
    fw->stop();

    delete fw;
    qDebug() << "about to exit workerThread";
    workerThread->exit(0);
    qDebug() << "waiting on workerThread";
    //workerThread->wait();
    qDebug() << "deleting on workerThread";

    delete workerThread;
    return retval;
}


void delay( int millisecondsToWait )
{
    QTime dieTime = QTime::currentTime().addMSecs( millisecondsToWait );
    while( QTime::currentTime() < dieTime )
    {
        QCoreApplication::processEvents( QEventLoop::AllEvents, 100 );
    }
}
