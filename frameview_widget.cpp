/* Qt includes */
#include <QSize>
#include <QDebug>
#include <QtGlobal>
#include <QRect>
#include <QApplication>
#include <QMainWindow>

/* standard includes */
#include <iostream>
#include <fstream>
#include <stdint.h>

/* Live View includes */
#include "frameview_widget.h"
#include "qcustomplot.h"
#include "settings.h"

frameview_widget::frameview_widget(frameWorker* fw, image_t image_type, QWidget* parent) : QWidget(parent)
{
    this->fw = fw;
    this->image_type = image_type;

    switch(image_type)
    {
    case BASE: ceiling = fw->base_ceiling; break;
    case DSF: ceiling = 100; break;
    case STD_DEV: ceiling = 100; break;
    default: break; // to remove annoying warnings on compilation
    }
    floor=0;
    count=0;
    frHeight = fw->getFrameHeight();
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

    displayCrosshairCheck.setText( tr("Display Crosshairs on Frame") );
    displayCrosshairCheck.setChecked( true );
    if( image_type == STD_DEV )
    {
        displayCrosshairCheck.setEnabled( false );
        displayCrosshairCheck.setChecked( false );
    }
    fps = 0;

    connect(&fpstimer, SIGNAL(timeout()), this, SLOT(updateFPS()));
    connect(&rendertimer,SIGNAL(timeout()),this,SLOT(handleNewFrame()));
    connect(qcp->yAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledY(QCPRange)));
    connect(qcp->xAxis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(colorMapScrolledX(QCPRange)));
    connect(&displayCrosshairCheck,SIGNAL(toggled(bool)),fw,SLOT(updateCrossDiplay(bool)));

    if(image_type==BASE || image_type==DSF)
    {
        this->setFocusPolicy(Qt::ClickFocus); //Focus accepted via clicking
        connect(qcp,SIGNAL(mouseDoubleClick(QMouseEvent*)),this,SLOT(setCrosshairs(QMouseEvent*)));
    }
    colorMapData = new QCPColorMapData(frWidth,frHeight,QCPRange(0,frWidth),QCPRange(0,frHeight));
    colorMap->setData(colorMapData);

    fpstimer.start(1000);
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}
frameview_widget::~frameview_widget()
{
    /*! \brief Deallocate QCustomPlot elements
     * \bug JP: The deallocation process causes a segmentation violation on the end of QCustomPlot. */
    delete colorScale;
    delete colorMap;
    delete qcp;
}

// public functions
double frameview_widget::getCeiling()
{
    /*! \brief Return the value of the ceiling for this widget as a double */
    return ceiling;
}
double frameview_widget::getFloor()
{
    /*! \brief Return the value of the floor for this widget as a double */
    return floor;
}
void frameview_widget::toggleDisplayCrosshair()
{
    /*! \brief Turn on or off the rendering of the crosshair */
    displayCrosshairCheck.setChecked(!displayCrosshairCheck.isChecked());
}

// public slots
void frameview_widget::handleNewFrame()
{
    /*! \brief Rendering function for a frameview
     * \paragraph
     *
     * frameview_widget plots a color map using data from the curFrame in the backend conditionally selected using the image_t.
     * frameWorker contains a local copy of a frame from cuda_take with all processed data that can be read from directly.
     * \paragraph
     *
     * For BASE type images, image_data_ptr is used, which has the type uint16_t (2 bytes/pixel). BASE images may display crosshairs,
     * so we should check for the coordinates of these elements. Additionally, the image data is y-axis reversed, so our indexing should
     * start from the top row, work left to right across a row, then down.
     * \paragraph
     *
     * For DSF type images, dark_subtracted_data is used, which has the type float. DSF images may also display crosshairs. As with BASE
     * images, the y-axis is reversed.
     * \paragraph
     *
     * For STD_DEV type images, the current std_dev_frame in frameWorker is used rather than the curFrame. As calculating the standard deviation proceeds
     * on the GPU regardless of CPU timing, the device must send back a signal whenever it has completed the most recent image. STD_DEV images will
     * display at a lower framerate than other frameviews, especially on weaker graphics cards. They do not render crosshairs.
     * \author JP Ryan
     * \author Noah Levy
     */
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
                        colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row-1)*frWidth + col]); // y-axis reversed
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
                        colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row-1)*frWidth + col]); // y-axis reversed
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
                    colorMap->data()->setCell(col,row,local_image_ptr[(frHeight-row-1)*frWidth + col]); // y-axis reversed
                }
            }

            qcp->replot();
        }
    }
    count++;
}
void frameview_widget::updateFPS()
{
    /*! \brief Updates the FPS of Display label */
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
void frameview_widget::colorMapScrolledY(const QCPRange &newRange)
{
    /*! \brief Controls the behavior of zooming the plot.
     * \paragraph
     *
     * Color Maps must not allow the user to zoom past the dimensions of the frame.
     */
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
    /*! \brief Controls the behavior of zooming the plot.
     * \paragraph
     *
     * Color Maps must not allow the user to zoom past the dimensions of the frame.
     */
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
void frameview_widget::updateCeiling(int c)
{
    /*! \brief Change the value of the ceiling for this widget to the input parameter and replot the color scale. */
    ceiling = (double)c;
    colorScale->setDataRange(QCPRange(floor,ceiling));
}
void frameview_widget::updateFloor(int f)
{
    /*! \brief Change the value of the floor for this widget to the input parameter and replot the color scale. */
    floor = (double)f;
    colorScale->setDataRange(QCPRange(floor,ceiling));
}
void frameview_widget::rescaleRange()
{
    /*! \brief Set the color scale of the display to the last used values for this widget */
    colorScale->setDataRange(QCPRange(floor,ceiling));
}
void frameview_widget::setCrosshairs(QMouseEvent* event)
{
    /*! \brief Sets the value of the crosshair and lines to average selection for rendering
     * \author JP Ryan */
    // lol ;) ;) ;) This is a stupid bugfix
    int currentVDiff = fw->crossStartRow - fw->crossHeight;
    int currentHDiff = fw->crossStartCol - fw->crossWidth;

    /*! \bug Occasionally, the frame will render with ghost crosshair data or graphical errors.
     * I do not know why this occurs, but it could be due to a logical error or a QCustomPlot issue.
     * Both Noah and I have found this issue. -JP */

    fw->crosshair_x = qcp->xAxis->pixelToCoord(event->pos().x());
    fw->crosshair_y = qcp->yAxis->pixelToCoord(event->pos().y());
    if( currentHDiff )
    {
        if( (fw->crosshair_x + (currentHDiff/2)) > frWidth)
        {
            fw->crossStartCol = frWidth - currentHDiff;
            fw->crossWidth = frWidth;
        }
        else if( (fw->crosshair_x - (currentHDiff/2)) < 0)
        {
            fw->crossWidth = currentHDiff;
        }
        else
        {
            fw->crossStartCol = fw->crosshair_x - (currentHDiff/2);
            fw->crossWidth = fw->crosshair_x + (currentHDiff/2);
        }
    }
    if( currentVDiff )
    {
        if(fw->crosshair_y + (currentVDiff/2) > frHeight)
        {
            fw->crossStartRow = frHeight - currentVDiff;
            fw->crossHeight = frHeight;
        }
        else if(fw->crosshair_y - (currentVDiff/2) < 0)
        {
            fw->crossHeight = currentVDiff;
        }
        else
        {
            fw->crossStartRow = fw->crosshair_y - (currentVDiff/2);
            fw->crossHeight = fw->crosshair_y + (currentVDiff/2);
        }
    }

    qDebug() << "x="<<fw->crosshair_x<< "y="<<fw->crosshair_y;
}
