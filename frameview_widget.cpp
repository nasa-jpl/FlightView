
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
#include "settings.h"
frameview_widget::frameview_widget(frameWorker *fw, image_t image_type, QWidget *parent) : QWidget(parent)
{
    //CODE goes here...
    //code

    this->fw = fw;
    this->image_type = image_type;
    switch(image_type)
    {
    case BASE: ceiling = 10000; break;
    case DSF: ceiling = 20; break;
    case STD_DEV: ceiling = 20; break;
    }
    floor = 0;
    count=0;
    toggleGrayScaleButton.setText("Toggle grayscale output");
    outputGrayScale = true;
    frHeight = fw->getFrameHeight();
    qDebug() << "fw frame height " << fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);
    QSizePolicy qsp(QSizePolicy::Preferred,QSizePolicy::Preferred);
    qsp.setHeightForWidth(true);
    qcp->setSizePolicy(qsp);
    qcp->heightForWidth(200);
    qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);
    qcp->axisRect()->setupFullAxesBox(true);
    qcp->xAxis->setLabel("x");
    qcp->yAxis->setLabel("y");

    //If this is uncommented, window size reflects focal plane size, otherwise it scales

    //qcp->setBackgroundScaled(Qt::AspectRatioMode);
    colorMap = new QCPColorMap(qcp->xAxis,qcp->yAxis);
    //colorMap->valueAxis()->
    colorMapData = NULL;
    qcp->addPlottable(colorMap);


    colorScale = new QCPColorScale(qcp);
    qcp->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect

    colorScale->setType(QCPAxis::atRight);

    colorMap->setColorScale(colorScale);
    //colorMap->data()->setValueRange(QCPRange(0,frHeight));
    colorMap->data()->setValueRange(QCPRange(frHeight,0));

    colorMap->data()->setKeyRange(QCPRange(0,frWidth));
    //colorScale->axis()->setLabel("kabel");
    //QCPRange * drange = new QCPRange(0.0d,10000.0d);
    //colorScale->setDataRange(*drange);
    colorMap->setDataRange(QCPRange(floor,ceiling));

    colorMap->setGradient(QCPColorGradient::gpJet);
    colorMap->setInterpolate(false);
    colorMap->setAntialiased(false);
    QCPMarginGroup *marginGroup = new QCPMarginGroup(qcp);
    qcp->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);

    qcp->rescaleAxes();
    qcp->axisRect()->setBackgroundScaled(false);


    layout.addWidget(qcp,8);
    fpsLabel.setText("FPS");
    layout.addWidget(&fpsLabel,1);
    //layout.addWidget(&toggleGrayScaleButton,1);
    this->setLayout(&layout);

    qDebug() << "emitting capture signal, starting timer";
    fps = 0;
    // fpstimer = new QTimer(this);
    connect(&fpstimer, SIGNAL(timeout()), this, SLOT(updateFPS()));
    fpstimer.start(1000);
    connect(&rendertimer,SIGNAL(timeout()),this,SLOT(handleNewFrame()));
    rendertimer.start(33);
    connect(qcp->yAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledY(QCPRange)));
    connect(qcp->xAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledX(QCPRange)));
    connect(colorMap,SIGNAL(dataRangeChanged(QCPRange)),this,SLOT(colorMapDataRangeChanged(QCPRange)));

    colorMapData = new QCPColorMapData(frWidth,frHeight,QCPRange(0,frWidth),QCPRange(0,frHeight));
    colorMap->setData(colorMapData);
    //emit startCapturing(); //This sends a message to the frame worker to start capturing (in different thread)
    //fw->captureFrames();

}
frameview_widget::~frameview_widget()
{
    if(qcp != NULL)
    {
        delete colorScale;
        //delete colorMapData;
        delete colorMap;
        delete qcp;
    }
}

void frameview_widget::handleNewFrame()
{

    if(!this->isHidden() &&  fw->curFrame != NULL)
    {
        if(image_type == BASE)
        {


            uint16_t * local_image_ptr = fw->curFrame->image_data_ptr;
            for(int col = 0; col < frWidth; col++)
            {
                for(int row = 0; row < frHeight; row++)
                {
                    //colorMap->data()->setCell(col,row,row^col);

                    colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row-1)*frWidth + col]);
                }
            }
            qcp->replot();


        }
        if(image_type == DSF)
        {

            float * local_image_ptr = fw->curFrame->dark_subtracted_data;
            for(int col = 0; col < frWidth; col++)
            {
                for(int row = 0; row < frHeight; row++)
                {
                    colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row-1)*frWidth + col]);
                }
            }
            qcp->replot();

        }

        if(image_type == STD_DEV && fw->std_dev_frame != NULL)
        {
            float * local_image_ptr = fw->std_dev_frame->std_dev_data;
            for(int col = 0; col < frWidth; col++)
            {
                for(int row = 0; row < frHeight; row++)
                {
                    colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row-1)*frWidth + col]);

                }
            }
            //sum/=frWidth*frHeight;
            //printf("gui std. dev avg=%f\n",sum);

            qcp->replot();
        }
    }
    count++;

}



void frameview_widget::updateFPS()
{
    seconds_elapsed++;
    //fps = count/(double)seconds_elapsed;


    if(seconds_elapsed < 5)
    {
        fps=((seconds_elapsed-1)*fps + (count-old_count))/(seconds_elapsed);
    }
    else
    {
        fps = (fps*4 + (count-old_count))/5;
    }
    old_count = count;
    fpsLabel.setText(QString("avg framerate: %1").arg(fps));
}
void frameview_widget::toggleGrayScale()
{
    outputGrayScale = !outputGrayScale;
    qDebug() << outputGrayScale;

}
void frameview_widget::updateCeiling(int c)
{
    ceiling = (double)c;
    colorScale->setDataRange(QCPRange(floor,ceiling));

    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));
}
void frameview_widget::updateFloor(int f)
{
    floor = (double)f;
    colorScale->setDataRange(QCPRange(floor,ceiling));

    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));
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
void frameview_widget::rescaleRange()
{
    colorScale->setDataRange(QCPRange(floor,ceiling));
}

