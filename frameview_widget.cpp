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
    toggleGrayScaleButton = new QPushButton("Toggle grayscale output");
    outputGrayScale = true;
    fw->moveToThread(workerThread);
    //connect(fw,SIGNAL(newFrameAvailable()), this, SLOT(handleNewFrame()));
    connect(workerThread,SIGNAL(started()), fw, SLOT(captureFrames()));
    connect(fw,SIGNAL(newFrameAvailable()), this, SLOT(handleNewFrame()));
    connect(toggleGrayScaleButton,SIGNAL(clicked()),this,SLOT(toggleGrayScale()));
    //   QRect pictureRect(0,0,fw->getHeight(),fw->getWidth());
    //   QPixmap picturePixmap(QPixmap::fromImage(*image));
    imageLabel = new QLabel();
    //   imageLabel->setGeometry(pictureRect);
    //   imageLabel->setPixmap(picturePixmap);
    layout->addWidget(imageLabel);
    fpsLabel = new QLabel("FPS");
    layout->addWidget(fpsLabel);
    layout->addWidget(toggleGrayScaleButton);
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
    int height = fw->getHeight();
    int width = fw->getWidth();
    //quint16 * local_image_ptr = reinterpret_cast<quint16 *>(fw->getFrameImagePtr());
    u_char * local_image_ptr =  fw->getFrameImagePtr();
    //QImage temp(fw->getFrameImagePtr(), width, height, QImage::Format_RGB16);
    QImage temp(width, height, QImage::Format_RGB32);
    if(outputGrayScale)
    {
        QRgb value;
        local_image_ptr += 1; //We're only intereted in the MSB's of stuff for viewing purposes, little endian so this give us the MSB
        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                value = * local_image_ptr * 0x000101010; //Evil bithack found @ http://stackoverflow.com/questions/835753/convert-grayscale-value-to-rgb-representation
                temp.setPixel(x,y,value);
                local_image_ptr+=2; //increment to get to next pixel, skip le LSB
            }
        }
        // delete local_image_ptr;
    }
    else
    {
        temp = QImage(fw->getFrameImagePtr(), width, height, QImage::Format_RGB16);
    }
    imageLabel->setPixmap(QPixmap::fromImage(temp));
    fps++;
}
void frameview_widget::updateFPS()
{
    fpsLabel->setText(QString("fps: %1").arg(fps));
    fps = 0;
}
void frameview_widget::toggleGrayScale()
{
    outputGrayScale = !outputGrayScale;
    qDebug() << outputGrayScale;

}
