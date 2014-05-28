#include <QApplication>
#include <QPixmap>
#include <QSplashScreen>
#include <QTime>

#include "mainwindow.h"

void delay(int);
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QPixmap logo_pixmap(":images/aviris-logo-transparent.png");
    QSplashScreen splash(logo_pixmap);
    splash.show();
    splash.showMessage(QObject::tr("Loading AVIRIS-Next Generation Live View..."),
                          Qt::AlignCenter | Qt::AlignBottom , Qt::gray);

    delay(500);
    MainWindow w;
    w.show();
    splash.finish(&w);
    return a.exec();
}


void delay( int millisecondsToWait )
{
    QTime dieTime = QTime::currentTime().addMSecs( millisecondsToWait );
    while( QTime::currentTime() < dieTime )
    {
        QCoreApplication::processEvents( QEventLoop::AllEvents, 100 );
    }
}
