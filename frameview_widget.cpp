#include "frameview_widget.h"
#include "take_object.hpp"
#include <QSize>
#include <QDebug>
#include <QRect>
frameview_widget::frameview_widget(QWidget *parent) :
    QWidget(parent)
{
    workerThread = new QThread();
    layout = new QVBoxLayout();
    fw = new frameWorker();
    fw->moveToThread(workerThread);
    //connect(fw,SIGNAL(newFrameAvailable()), this, SLOT(handleNewFrame()));
    connect(workerThread,SIGNAL(started()), fw, SLOT(captureFrames()));
   connect(fw,SIGNAL(newFrameAvailable()), this, SLOT(handleNewFrame()));

    //   QRect pictureRect(0,0,fw->getHeight(),fw->getWidth());
    //   QPixmap picturePixmap(QPixmap::fromImage(*image));
    imageLabel = new QLabel();
    //   imageLabel->setGeometry(pictureRect);
    //   imageLabel->setPixmap(picturePixmap);
    layout->addWidget(imageLabel);
    fpsLabel = new QLabel("FPS");
    layout->addWidget(fpsLabel);
    this->setLayout(layout);

    workerThread->start();

    qDebug() << "emitting capture signal, starting timer";
    fps = 0;
    fpstimer = new QTimer(this);
    connect(fpstimer, SIGNAL(timeout()), this, SLOT(updateFPS()));
    fpstimer->start(1000);
    //emit startCapturing(); //This sends a message to the frame worker to start capturing (in different thread)
    //fw->captureFrames();

}
void frameview_widget::handleNewFrame()
{
    //qDebug("handling\n");
    //qDebug() << fw->getWidth() << " " << fw->getHeight() << "\n";
    QImage temp(fw->getFrameImagePtr(), fw->getWidth(), fw->getHeight(), QImage::Format_RGB16);

    // QImage image;

    //  image = temp;

    //static QLabel * imageLabel
    imageLabel->setPixmap(QPixmap::fromImage(temp));
    fps++;
}
void frameview_widget::updateFPS()
{
    fpsLabel->setText(QString("fps: %1").arg(fps));
    fps = 0;
}
