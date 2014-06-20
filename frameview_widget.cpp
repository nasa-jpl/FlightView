
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

    int start_width = 640; //Corresponds to "key"
    int start_height = 481; //Corresponds to "value"

    this->fw = fw;
    this->image_type = image_type;
    this->ceiling = (1<<16)-1;
    this->floor = -1.0f*ceiling;
    layout = new QVBoxLayout();
    toggleGrayScaleButton = new QPushButton("Toggle grayscale output");
    outputGrayScale = true;
    qcp = new QCustomPlot(this);

    qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);
    qcp->axisRect()->setupFullAxesBox(true);
    qcp->xAxis->setLabel("x");
    qcp->yAxis->setLabel("y");

    colorMap = new QCPColorMap(qcp->xAxis,qcp->yAxis);
    colorMapData = NULL;
    qcp->addPlottable(colorMap);


   // colorScale = new QCPColorScale(qcp);
    //qcp->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect

   // colorScale->setType(QCPAxis::atRight);

    //colorMap->setColorScale(colorScale);
    colorMap->data()->setValueRange(QCPRange(0,start_height));
    colorMap->data()->setKeyRange(QCPRange(0,start_width));
    //colorScale->axis()->setLabel("kabel");
    //colorScale->setDataRange(QCPRange(floor,ceiling));
    colorMap->setDataRange(QCPRange(0,10000));
    colorMap->setGradient(QCPColorGradient::gpJet);
    colorMap->setInterpolate(false);

    QCPMarginGroup *marginGroup = new QCPMarginGroup(qcp);
    qcp->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);
    //colorScale->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);

    qcp->rescaleAxes();
    //   imageLabel->setGeometry(pictureRect);
    //   imageLabel->setPixmap(picturePixmap);



    layout->addWidget(qcp);
    fpsLabel = new QLabel("FPS");
    layout->addWidget(fpsLabel);
    layout->addWidget(toggleGrayScaleButton);
    this->setLayout(layout);


    qDebug() << "emitting capture signal, starting timer";
    fps = 0;
    fpstimer = new QTimer(this);
    connect(fpstimer, SIGNAL(timeout()), this, SLOT(updateFPS()));
    fpstimer->start(1000);

    connect(qcp->yAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledY(QCPRange)));
    connect(qcp->xAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledX(QCPRange)));

    //emit startCapturing(); //This sends a message to the frame worker to start capturing (in different thread)
    //fw->captureFrames();

}


void frameview_widget::handleNewFrame()
{
    if(colorMapData == NULL) //This cannot be in the constructor because we do not know the height yet.
    {
        frHeight = fw->getHeight();
        frWidth = fw->getWidth();
        if(!fw->isChroma() && image_type == BASE)
        {
            frHeight += 1;
        }

       colorMapData = new QCPColorMapData(frWidth,frHeight,QCPRange(0,frWidth),QCPRange(0,frHeight));
       colorMap->setData(colorMapData);
    }
    if(fps%4 == 0 && !this->isHidden())
    {

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
            float slope = 255.0f/(ceiling - floor);
            boost::shared_array< float > local_image_ptr = fw->getDSF();

        }
        else if(image_type == STD_DEV)
        {
            float slope = 255.0f/(ceiling - floor);
            boost::shared_array< float > local_image_ptr = fw->getStdDevData();
        }

       colorMap->rescaleDataRange();
        //colorMap->rescaleAxes();
       // colorMap->setDataRange(QCPRange(floor,ceiling));
        qcp->replot();
        //colorMap->setDataRange(QCPRange(floor,ceiling));

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
    this->ceiling = (double)c;
    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));
    qDebug() << "ceiling updated" << ceiling;
}
void frameview_widget::updateFloor(int f)
{
    this->floor = (double)f;
    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));

    qDebug() << "floor updated" << floor;

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
