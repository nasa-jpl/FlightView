
#include "frameview_widget.h"
#include <QSize>
#include <QDebug>
#include <QtGlobal>
#include <QRect>
frameview_widget::frameview_widget(frameWorker *fw, image_t image_type, QWidget *parent) :
    QWidget(parent)
{


    this->fw = fw;
    this->image_type = image_type;
    this->ceiling = (1<<16)-1;
    this->floor = -1.0f*ceiling;
    layout = new QVBoxLayout();
    toggleGrayScaleButton = new QPushButton("Toggle grayscale output");
    outputGrayScale = true;
    imageLabel = new QLabel();
    //   imageLabel->setGeometry(pictureRect);
    //   imageLabel->setPixmap(picturePixmap);
    layout->addWidget(imageLabel);
    fpsLabel = new QLabel("FPS");
    layout->addWidget(fpsLabel);
    layout->addWidget(toggleGrayScaleButton);
    this->setLayout(layout);


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

    if(fps%4 == 0 && !this->isHidden())
    {


        int height = fw->getHeight();
        int width = fw->getWidth();
        QImage temp(width, height, QImage::Format_RGB32);

        if(image_type == BASE)
        {
            uint16_t * local_image_ptr =  fw->getFrameImagePtr();
            if(outputGrayScale)
            {
                QRgb value;
                for(int y = 0; y < height; y++)
                {
                    for(int x = 0; x < width; x++)
                    {

                        value = ((uint8_t) (local_image_ptr[x+width*y] >> 8)) * 0x000101010; //Evil bithack found @ http://stackoverflow.com/questions/835753/convert-grayscale-value-to-rgb-representation
                        temp.setPixel(x,y,value);
                    }
                }
            }
            else
            {
                temp = QImage(reinterpret_cast <uint8_t *>(fw->getFrameImagePtr()), width, height, QImage::Format_RGB16);
            }
        }
        else if(image_type == DSF)
        {
            float slope = 255.0f/(ceiling - floor);
            boost::shared_array< float > local_image_ptr = fw->getDSF();
            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {

                    uint8_t mag;
                    //value = ((uint8_t) (local_image_ptr[x+width*y]+10.0f)/divisor) * 0x000101010; //Evil bithack found @ http://stackoverflow.com/questions/835753/convert-grayscale-value-to-rgb-representation
                    //value = local_image_ptr[x+width*y] > 0 ? ((uint8_t) local_image_ptr[x+width*y]) * 0x000101010 : ((uint8_t) -1.0f*local_image_ptr[x+width*y]) * 0x000101010 ; //If its greater than 0, use its mag, otherwise use it's abs mag in 1 channel
                    //value = local_image_ptr[x+width*y] > 0 ? ((uint8_t) local_image_ptr[x+width*y]) * 0x000101010 : 0 ; //If its greater than 0, use its mag, otherwise use it's abs mag in 1 channel
                    mag = local_image_ptr[x+width*y] > floor ? (local_image_ptr[x+width*y] < ceiling ?  (local_image_ptr[x+width*y] - floor)*slope : 255) : 0;
                    QRgb value = mag * 0x00101010;
                    if(qrand() % 1000000 == 0 && DEBUG) //one in a 100000 chance
                    {
                        qDebug() << "@ x=" << x << " y=" << y << " pix val is d"<< local_image_ptr[x+width*y] << " mag is: " << mag << " float val is: " << local_image_ptr[x+width*y] << " value is: R=" << ((value & 0x00011000) >> 4) << "G=" << ((value & 0x00001100) >> 2) << "B=" << (value & 0x00000011) << " slope: " << slope ;
                    }
                    temp.setPixel(x,y,value);
                    //local_image_ptr++; //increment to get to next pixel, skip the little endian LSB
                }
            }
        }
        else if(image_type == STD_DEV)
        {
            float slope = 255.0f/(ceiling - floor);
            boost::shared_array< float > local_image_ptr = fw->getStdDevData();
            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {

                    uint8_t mag;
                    //value = ((uint8_t) (local_image_ptr[x+width*y]+10.0f)/divisor) * 0x000101010; //Evil bithack found @ http://stackoverflow.com/questions/835753/convert-grayscale-value-to-rgb-representation
                    //value = local_image_ptr[x+width*y] > 0 ? ((uint8_t) local_image_ptr[x+width*y]) * 0x000101010 : ((uint8_t) -1.0f*local_image_ptr[x+width*y]) * 0x000101010 ; //If its greater than 0, use its mag, otherwise use it's abs mag in 1 channel
                    //value = local_image_ptr[x+width*y] > 0 ? ((uint8_t) local_image_ptr[x+width*y]) * 0x000101010 : 0 ; //If its greater than 0, use its mag, otherwise use it's abs mag in 1 channel
                    mag = local_image_ptr[x+width*y] > floor ? (local_image_ptr[x+width*y] < ceiling ?  (local_image_ptr[x+width*y] - floor)*slope : 255) : 0;
                    QRgb value = mag * 0x00101010;
                    if(qrand() % 1000000 == 0  && DEBUG) //one in a 100000 chance
                    {
                        qDebug() << "@ x=" << x << " y=" << y << " pix val is d"<< local_image_ptr[x+width*y] << " mag is: " << mag << " float val is: " << local_image_ptr[x+width*y] << " value is: R=" << ((value & 0x00011000) >> 4) << "G=" << ((value & 0x00001100) >> 2) << "B=" << (value & 0x00000011) << " slope: " << slope ;
                    }
                    temp.setPixel(x,y,value);
                    //local_image_ptr++; //increment to get to next pixel, skip the little endian LSB
                }
            }
        }
        imageLabel->setPixmap(QPixmap::fromImage(temp));

    }
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
void frameview_widget::updateCeiling(int c)
{
    this->ceiling = (float)c;
    qDebug() << "ceiling updated";
}
void frameview_widget::updateFloor(int f)
{
    this->floor = (float)f;
    qDebug() << "floor updated";

}
