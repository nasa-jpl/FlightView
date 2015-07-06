
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

frameview_widget::frameview_widget(frameWorker* fw, image_t image_type, QWidget* parent) : QWidget(parent)
{
    //CODE goes here...
    //code

    this->fw = fw;
    this->image_type = image_type;

    int base_ceiling;
    if( fw->to.cam_type == CL_6604A )
        base_ceiling = 16383;
    else
        base_ceiling = 65535;

    switch(image_type)
    {
    case BASE: ceiling = base_ceiling; break;
    case DSF: ceiling = 20; break;
    case STD_DEV: ceiling = 20; break;
    default: break; // to remove annoying warnings on compilation
    }
    floor=0;
    count=0;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();

    displayCrosshairCheck.setText( tr("Display Crosshairs on Frame") );
    displayCrosshairCheck.setChecked( true );
    if( image_type == STD_DEV )
    {
        displayCrosshairCheck.setEnabled( false );
        displayCrosshairCheck.setChecked( false );
    }
    connect(&displayCrosshairCheck,SIGNAL(toggled(bool)),fw,SLOT(updateCrossDiplay(bool)));



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
    colorMapData = NULL;
    qcp->addPlottable(colorMap);


    colorScale = new QCPColorScale(qcp);
    qcp->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect

    colorScale->setType(QCPAxis::atRight);

    colorMap->setColorScale(colorScale);
    colorMap->data()->setValueRange(QCPRange(frHeight,0));

    colorMap->data()->setKeyRange(QCPRange(0,frWidth));
    colorMap->setDataRange(QCPRange(floor,ceiling));

    colorMap->setGradient(QCPColorGradient::gpJet);
    colorMap->setInterpolate(false);
    colorMap->setAntialiased(false);
    QCPMarginGroup *marginGroup = new QCPMarginGroup(qcp);
    qcp->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop,marginGroup);

    qcp->rescaleAxes();
    qcp->axisRect()->setBackgroundScaled(false);

    layout.addWidget(qcp,0,0,8,8);
    fpsLabel.setText("FPS");
    layout.addWidget(&fpsLabel,8,0,1,2);
    layout.addWidget(&displayCrosshairCheck,8,2,1,2);
    this->setLayout(&layout);

    fps = 0;
    connect(&fpstimer, SIGNAL(timeout()), this, SLOT(updateFPS()));
    fpstimer.start(1000);
    connect(&rendertimer,SIGNAL(timeout()),this,SLOT(handleNewFrame()));
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);

    connect(qcp->yAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledY(QCPRange)));
    connect(qcp->xAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledX(QCPRange)));

    if(image_type==BASE || image_type==DSF)
    {
        this->setFocusPolicy(Qt::ClickFocus); //Focus accepted via clicking
        connect(qcp,SIGNAL(mouseDoubleClick(QMouseEvent*)),this,SLOT(setCrosshairs(QMouseEvent*)));
    }
    colorMapData = new QCPColorMapData(frWidth,frHeight,QCPRange(0,frWidth),QCPRange(0,frHeight));
    colorMap->setData(colorMapData);
}
frameview_widget::~frameview_widget()
{
    delete colorScale;
    delete colorMap;
    delete qcp;
}
void frameview_widget::keyPressEvent(QKeyEvent *event)
{
    if((image_type == BASE || image_type == DSF) && !this->isHidden() && event->key() == Qt::Key_Escape)
    {
        fw->crosshair_x = -1;
        fw->crosshair_y = -1;

        qDebug() << "x="<<fw->crosshair_x<< "y="<<fw->crosshair_y;
    }
}

void frameview_widget::handleNewFrame()
{
    if(!this->isHidden() && fw->curFrame)
    {
        if(image_type == BASE)
        {

            uint16_t * local_image_ptr = fw->curFrame->image_data_ptr;
            for(int col = 0; col < frWidth; col++ )
            {
                for(int row = 0; row < frHeight; row++ )
                {
                    if( (row == fw->crosshair_y || col == fw->crosshair_x || row == fw->crossStartRow || row == fw->crossHeight \
                         || col == fw->crossStartCol || col == fw->crossWidth) && fw->displayCross )
                    {
                        // this will blank out the part of the frame where the crosshair is pointing so that it is
                        // visible in the display
                        colorMap->data()->setCell(col,row,NAN);
                    }
                    else
                    {
                        // display normally
                        colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row-1)*frWidth + col]);
                    }
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
                    if( (row == fw->crosshair_y || col == fw->crosshair_x || row == fw->crossStartRow || row == fw->crossHeight \
                         || col == fw->crossStartCol || col == fw->crossWidth) && fw->displayCross )
                    {
                        // this will blank out the part of the frame where the crosshair is pointing so that it is
                        // visible in the display
                        colorMap->data()->setCell(col,row,NAN);
                    }
                    else
                    {
                        // display normally
                        colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row-1)*frWidth + col]);
                    }
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

            qcp->replot();
        }
    }
    count++;
}
void frameview_widget::updateFPS()
{
    seconds_elapsed++;

    if(seconds_elapsed < 5)
    {
        fps=((seconds_elapsed-1)*fps + (count-old_count))/(seconds_elapsed);
    }
    else
    {
        fps = (fps*4 + (count-old_count))/5;
    }
    old_count = count;
    fpsLabel.setText(QString("fps of display: %1").arg(fps));
}
void frameview_widget::updateCeiling(int c)
{
    ceiling = (double)c;
    colorScale->setDataRange(QCPRange(floor,ceiling));
}
void frameview_widget::updateFloor(int f)
{
    floor = (double)f;
    colorScale->setDataRange(QCPRange(floor,ceiling));
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
void frameview_widget::setCrosshairs(QMouseEvent * event)
{
    fw->crosshair_x = qcp->xAxis->pixelToCoord(event->pos().x());
    fw->crosshair_y = qcp->yAxis->pixelToCoord(event->pos().y());

    qDebug() << "x="<<fw->crosshair_x<< "y="<<fw->crosshair_y;
}
