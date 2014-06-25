
#include "frameview_widget.h"
#include <QSize>
#include <QDebug>
#include <QtGlobal>
#include <QRect>
#include <QApplication>
#include <QMainWindow>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include "qcustomplot.h"
frameview_widget::frameview_widget(frameWorker *fw, image_t image_type, QWidget *parent) :
    QWidget(parent)
{


    this->fw = fw;
    this->image_type = image_type;
    this->ceiling = 10000;
    this->floor = 0;
    count=0;
    toggleGrayScaleButton.setText("Toggle grayscale output");
    outputGrayScale = true;
    qcp = NULL;

}
frameview_widget::~frameview_widget()
{
    if(qcp != NULL)
    {
        disconnect(this,SLOT(handleNewFrame()),fw,SIGNAL(newFrameAvailable()));
        delete colorScale;
        delete colorMapData;
        delete colorMap;
        delete qcp;
    }
}

void frameview_widget::initQCPStuff() //Needs to be in same thread as handleNewFrame?
{

    frHeight = fw->getHeight();
    frWidth = fw->getWidth();
    qcp = new QCustomPlot(this);

    QSizePolicy qsp(QSizePolicy::Preferred,QSizePolicy::Preferred);
         qsp.setHeightForWidth(true);
    qcp->setSizePolicy(qsp);
    qcp->heightForWidth(200);
    qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);
    qcp->axisRect()->setupFullAxesBox(true);
    qcp->xAxis->setLabel("x");
    qcp->yAxis->setLabel("y");
    //If this is uncommented, window size reflects focal plane size, otherwise it scales
    //qcp->setMaximumSize(fw->getWidth(),fw->getHeight());
    //qcp->setBackgroundScaled(Qt::AspectRatioMode);
    colorMap = new QCPColorMap(qcp->xAxis,qcp->yAxis);
    colorMapData = NULL;
    qcp->addPlottable(colorMap);


    colorScale = new QCPColorScale(qcp);
    qcp->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect

    colorScale->setType(QCPAxis::atRight);

    colorMap->setColorScale(colorScale);
    colorMap->data()->setValueRange(QCPRange(0,frHeight));
    colorMap->data()->setKeyRange(QCPRange(0,frWidth));
    //colorScale->axis()->setLabel("kabel");
    //QCPRange * drange = new QCPRange(0.0d,10000.0d);
    //colorScale->setDataRange(*drange);
    colorMap->setDataRange(QCPRange(floor,ceiling));

    colorMap->setGradient(QCPColorGradient::gpJet);
    colorMap->setInterpolate(false);

    QCPMarginGroup *marginGroup = new QCPMarginGroup(qcp);
    qcp->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);

    qcp->rescaleAxes();
    //   imageLabel->setGeometry(pictureRect);
    //   imageLabel->setPixmap(picturePixmap);



    layout.addWidget(qcp,8);
    fpsLabel.setText("FPS");
    layout.addWidget(&fpsLabel,1);
    layout.addWidget(&toggleGrayScaleButton,1);
    this->setLayout(&layout);


    qDebug() << "emitting capture signal, starting timer";
    fps = 0;
   // fpstimer = new QTimer(this);
    connect(&fpstimer, SIGNAL(timeout()), this, SLOT(updateFPS()));
    fpstimer.start(1000);

    connect(qcp->yAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledY(QCPRange)));
   connect(qcp->xAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledX(QCPRange)));
    connect(colorMap,SIGNAL(dataRangeChanged(QCPRange)),this,SLOT(colorMapDataRangeChanged(QCPRange)));
    //emit startCapturing(); //This sends a message to the frame worker to start capturing (in different thread)
    //fw->captureFrames();
}
void frameview_widget::handleNewFrame()
{
    if(qcp==NULL)
    {
        qDebug() << "chcking qcp";
        initQCPStuff();
        colorMapData = new QCPColorMapData(frWidth,frHeight,QCPRange(0,frWidth),QCPRange(0,frHeight));
        colorMap->setData(colorMapData);
    }


    if(fps%4 == 0 && !this->isHidden())
    {
        //qDebug() << image_type;
        if(image_type == BASE)
        {
            //qDebug() << "starting redraw";
            uint16_t * local_image_ptr = fw->getRawImagePtr();
            for(int col = 0; col < frWidth; col++)
            {
                for(int row = 0; row < frHeight; row++)
                {
                    colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row)*frWidth + col]);
                }
            }

        }
        else if(image_type == DSF)
        {
            boost::shared_array< float > local_image_ptr = fw->getDSF();
            for(int col = 0; col < frWidth; col++)
            {
                for(int row = 0; row < frHeight; row++)
                {
                    colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row)*frWidth + col]);
                }
            }

        }
        else if(image_type == STD_DEV)
        {
            boost::shared_array< float > local_image_ptr = fw->getStdDevData();
            for(int col = 0; col < frWidth; col++)
            {
                for(int row = 0; row < frHeight; row++)
                {
                    colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row)*frWidth + col]);
                }
            }
        }
        colorScale->setDataRange(QCPRange(floor,ceiling));
        qcp->replot();

    }
    fps++;
    count++;

}
void frameview_widget::updateFPS()
{
    fpsLabel.setText(QString("fps: %1").arg(fps));
    fps = 0;
}
void frameview_widget::toggleGrayScale()
{
    outputGrayScale = !outputGrayScale;
    qDebug() << outputGrayScale;

}
void frameview_widget::updateCeiling(int c)
{
    QMutexLocker ml(&mMutex);
    ceiling = (double)c;
    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));

    qDebug() << "ceiling updated" << this->ceiling;
}
void frameview_widget::updateFloor(int f)
{
    QMutexLocker ml(&mMutex);
    floor = (double)f;
    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));
    qDebug() << "floor updated" << this->floor;

}
void frameview_widget::colorMapScrolledY(const QCPRange &newRange)
{
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = frHeight;
    if (boundedRange.size() > upperRangeBound-lowerRangeBound)
    {
      boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    } else
    {
      double oldSize = boundedRange.size();
      if (boundedRange.lower < lowerRangeBound)
      {
        boundedRange.lower = lowerRangeBound;
        boundedRange.upper = lowerRangeBound+oldSize;
      }
      if (boundedRange.upper > upperRangeBound)
      {
        boundedRange.lower = upperRangeBound-oldSize;
        boundedRange.upper = upperRangeBound;
      }
    }
    qcp->yAxis->setRange(boundedRange);
}

void frameview_widget::colorMapScrolledX(const QCPRange &newRange)
{
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = frWidth;
    if (boundedRange.size() > upperRangeBound-lowerRangeBound)
    {
      boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    } else
    {
      double oldSize = boundedRange.size();
      if (boundedRange.lower < lowerRangeBound)
      {
        boundedRange.lower = lowerRangeBound;
        boundedRange.upper = lowerRangeBound+oldSize;
      }
      if (boundedRange.upper > upperRangeBound)
      {
        boundedRange.lower = upperRangeBound-oldSize;
        boundedRange.upper = upperRangeBound;
      }
    }
    qcp->xAxis->setRange(boundedRange);
}
void frameview_widget::colorMapDataRangeChanged(const QCPRange &newRange)
{
    qDebug() << "qcp new range data";
}
double frameview_widget::getCeiling()
{
    return ceiling;
}
double frameview_widget::getFloor()

{
    return floor;
}
