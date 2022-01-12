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

frameview_widget::frameview_widget(frameWorker *fw, image_t image_type, QWidget *parent)
    : QWidget(parent)
{
    this->fw = fw;
    this->image_type = image_type;
    floor = 0;
    useDSF = false;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();

    switch(image_type) {
    case BASE:
        ceiling = fw->base_ceiling;
        break;
    case DSF:
        ceiling = 100;
        break;
    case STD_DEV:
        ceiling = 100;
        break;
    case WATERFALL:
    {
        wflength = 1024;
        std::vector <float> blank;
        for(unsigned int c=0; c < (unsigned int)frWidth; c++)
        {
            blank.push_back(0.00);
        }
        for(unsigned int l=0; l < (unsigned int)wflength; l++)
        {
            wfimage.push_back(blank);
        }

        break;
    }
    default:
        break;
    }
    // Note, this code is only run once, at the initial execution of liveview.


    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);
    QSizePolicy qsp(QSizePolicy::Preferred, QSizePolicy::Preferred);
    qsp.setHeightForWidth(true);
    qcp->setSizePolicy(qsp);
    qcp->heightForWidth(200);
    qcp->setInteraction(QCP::iRangeDrag, true);
    qcp->setInteraction(QCP::iRangeZoom, true);
    // qcp->axisRect()->setRangeZoom(Qt::Horizontal);
    qcp->axisRect()->setupFullAxesBox(true);
    qcp->xAxis->setLabel("x");
    qcp->yAxis->setLabel("y");
    qcp->yAxis->setRangeReversed(true);
    //If this is uncommented, window size reflects focal plane size, otherwise it scales
    //qcp->setBackgroundScaled(Qt::AspectRatioMode);

    colorMap = new QCPColorMap(qcp->xAxis,qcp->yAxis);
    colorMapData = NULL;

    qcp->addPlottable(colorMap);

    colorScale = new QCPColorScale(qcp);
    qcp->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect

    colorScale->setType(QCPAxis::atRight);

    colorMap->setColorScale(colorScale);

    if(image_type == WATERFALL)
    {
        colorMap->data()->setValueRange(QCPRange(0, wflength-1));
    } else {
        colorMap->data()->setValueRange(QCPRange(0, frHeight-1));
    }

    colorMap->data()->setKeyRange(QCPRange(0, frWidth-1));

    colorMap->setDataRange(QCPRange(floor, ceiling));

    colorMap->setGradient(QCPColorGradient::gpJet); //gpJet for color, gpGrayscale for gray

    colorMap->setInterpolate(false);
    colorMap->setAntialiased(false);
    QCPMarginGroup * marginGroup = new QCPMarginGroup(qcp);
    qcp->axisRect()->setMarginGroup(QCP::msBottom | QCP::msTop,marginGroup);
    colorScale->setMarginGroup(QCP::msBottom | QCP::msTop,marginGroup);

    qcp->rescaleAxes();
    qcp->axisRect()->setBackgroundScaled(false);


    layout.addWidget(qcp, 0, 0, 8, 8);



    fpsLabel.setText("FPS");
    layout.addWidget(&fpsLabel, 8, 0, 1, 2);
    layout.addWidget(&displayCrosshairCheck, 8, 2, 1, 2);
    layout.addWidget(&zoomXCheck, 8, 4, 1, 2);
    layout.addWidget(&zoomYCheck, 8, 6, 1, 2);    
    this->setLayout(&layout);

    displayCrosshairCheck.setText(tr("Display Crosshairs on Frame"));
    displayCrosshairCheck.setChecked(true);
    if (image_type == STD_DEV) {
        displayCrosshairCheck.setEnabled(false);
        displayCrosshairCheck.setChecked(false);
    }

    zoomXCheck.setText("Zoom on X axis only");
    zoomYCheck.setText("Zoom on Y axis only");
    zoomXCheck.setChecked(false);
    zoomYCheck.setChecked(false);

    fps = 0;
    clock.start();

    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
    connect(qcp->yAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(colorMapScrolledY(QCPRange)));
    connect(qcp->xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(colorMapScrolledX(QCPRange)));
    connect(&displayCrosshairCheck, SIGNAL(toggled(bool)), fw, SLOT(updateCrossDiplay(bool)));
    connect(&zoomXCheck, SIGNAL(toggled(bool)), this, SLOT(setScrollX(bool)));
    connect(&zoomYCheck, SIGNAL(toggled(bool)), this, SLOT(setScrollY(bool)));
    connect(fw, SIGNAL(setColorScheme_signal(int)), this, SLOT(handleNewColorScheme(int)));

    if (image_type==BASE || image_type==DSF) {
        this->setFocusPolicy(Qt::ClickFocus); //Focus accepted via clicking
        connect(qcp, SIGNAL(mouseDoubleClick(QMouseEvent*)), this, SLOT(setCrosshairs(QMouseEvent*)));
    }
    if(image_type == WATERFALL)
    {
        colorMapData = new QCPColorMapData(frWidth, wflength, QCPRange(0, frWidth-1), QCPRange(0, wflength-1));
    } else {
        colorMapData = new QCPColorMapData(frWidth, frHeight, QCPRange(0, frWidth-1), QCPRange(0, frHeight-1));
    }
    colorMap->setData(colorMapData);
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}
frameview_widget::~frameview_widget()
{
    /*! \brief Deallocate QCustomPlot elements */
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

void frameview_widget::handleNewColorScheme(int scheme)
{
    switch(scheme)
    {
    case 0:
        colorMap->setGradient(QCPColorGradient::gpJet);
        break;
    case 1:
        colorMap->setGradient(QCPColorGradient::gpGrayscale);
        break;
    case 2:
        colorMap->setGradient(QCPColorGradient::gpThermal);
        break;
    case 3:
        colorMap->setGradient(QCPColorGradient::gpHues);
        break;
    case 4:
        colorMap->setGradient(QCPColorGradient::gpPolar);
        break;
    case 5:
        colorMap->setGradient(QCPColorGradient::gpHot);
        break;
    case 6:
        colorMap->setGradient(QCPColorGradient::gpCold);
        break;
    case 7:
        colorMap->setGradient(QCPColorGradient::gpNight);
        break;
    case 8:
        colorMap->setGradient(QCPColorGradient::gpIon);
        break;
    case 9:
        colorMap->setGradient(QCPColorGradient::gpCandy);
        break;
    case 10:
        colorMap->setGradient(QCPColorGradient::gpGeography);
        break;
    default:
        std::cerr << "color scheme not recognized, number: " << fw->color_scheme << std::endl;
        colorMap->setGradient(QCPColorGradient::gpJet);
        break;
    }


}


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
     * \author Jackie Ryan
     * \author Noah Levy
     */

    if((fw->curFrame->image_data_ptr != NULL) && image_type == WATERFALL)
    {
        // Copy waterfall data in, even if hidden:
        int row = fw->crosshair_y;
        if(row < 0)
            return;

        float *local_image_ptr = fw->curFrame->dark_subtracted_data;
        std::vector <float> line;
        for(int col = 0; col < frWidth; col++)
        {
            line.push_back(local_image_ptr[row * frWidth + col]);
        }
        // There's a better way, but for now this will be ok:
        wfimage.push_front(line); // Append to top
        // I have seen a crash here even when wflength was defined as 1024:
        // crash was when only items 0-420 existed
        // Other items were "not accessable"
        wfimage.resize(wflength); // Cut off anything too small.
    }

    if(!this->isHidden() && (fw->curFrame->image_data_ptr != NULL)) {
//        if(image_type == BASE) {
//            uint16_t *local_image_ptr = fw->curFrame->image_data_ptr;
//            for(int col = 0; col < frWidth; col++)
//                for(int row = 0; row < frHeight; row++ )
//                    // this will blank out the part of the frame where the crosshair is pointing so that it is
//                    // visible in the display
//                    if( (row == fw->crosshair_y || col == fw->crosshair_x || row == fw->crossStartRow || row == fw->crossHeight \
//                         || col == fw->crossStartCol || col == fw->crossWidth) && fw->displayCross )
//
//                        colorMap->data()->setCell(col, row, NAN);
//                    else
//                        // colorMap->data()->setCell(col, row, local_image_ptr[(frHeight - row - 1) * frWidth + col]); // y-axis reversed
//                        colorMap->data()->setCell(col, row, local_image_ptr[row * frWidth + col]); // y-axis NOT reversed
//            qcp->replot();
//        }
        if((image_type == DSF) || (image_type==BASE)) {
            uint16_t* local_image_ptr_uint = fw->curFrame->image_data_ptr;
            float* local_image_ptr_float = fw->curFrame->dark_subtracted_data;

            for(int col = 0; col < frWidth; col++)
                for(int row = 0; row < frHeight; row++)
                    // this will blank out the part of the frame where the crosshair is pointing so that it is
                    // visible in the display
                    if( (row == fw->crosshair_y || col == fw->crosshair_x || row == fw->crossStartRow || row == fw->crossHeight \
                         || col == fw->crossStartCol || col == fw->crossWidth) && fw->displayCross )
                    {
                        colorMap->data()->setCell(col, row, NAN);
                    } else {
                        // colorMap->data()->setCell(col, row, local_image_ptr[(frHeight - row - 1) * frWidth + col]); // y-axis reversed
                        if(useDSF && (image_type == DSF || image_type == BASE))
                        {
                            colorMap->data()->setCell(col, row, local_image_ptr_float[row * frWidth + col]); // y-axis NOT reversed
                        } else {
                            colorMap->data()->setCell(col, row, local_image_ptr_uint[row * frWidth + col]); // y-axis NOT reversed
                        }
                    }
            qcp->replot();
        }

        if(image_type == WATERFALL)
        {
            // Display time:
            std::vector <float> rowdata;

            for(unsigned int row=0; row < wfimage.size(); row++)
            {
                rowdata = wfimage.at(row);
                for(unsigned int col=0; col < (unsigned int)frWidth; col++)
                {
                    colorMap->data()->setCell(col, row, rowdata.at(col)); // y-axis NOT reversed
                }
            }
            qcp->replot();
        }

        if(image_type == STD_DEV && fw->std_dev_frame != NULL) {
            float * local_image_ptr = fw->std_dev_frame->std_dev_data;
            for (int col = 0; col < frWidth; col++)
                for (int row = 0; row < frHeight; row++)
                    // colorMap->data()->setCell(col, row, (double_t)local_image_ptr[(frHeight - row - 1) * frWidth + col]); // y-axis reversed
                    colorMap->data()->setCell(col, row, local_image_ptr[row * frWidth + col]); // y-axis NOT reversed
            qcp->replot();
        }
    }
    count++;
    if (count % 20 == 0 && count != 0) {
        fps = 20.0 / clock.restart() * 1000.0;
        fps_string = QString::number(fps, 'f', 1);
        fpsLabel.setText(QString("fps of display: %1").arg(fps_string));
    }
}
void frameview_widget::colorMapScrolledY(const QCPRange &newRange)
{
    /*! \brief Controls the behavior of zooming the plot.
     * \param newRange Mouse wheel scrolled range.
     * Color Maps must not allow the user to zoom past the dimensions of the frame.
     */
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = 0;
    if(image_type == WATERFALL)
    {
        upperRangeBound = wflength-1;
    } else {
        upperRangeBound = frHeight-1;
    }
    if (boundedRange.size() > upperRangeBound - lowerRangeBound) {
        boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    } else {
        double oldSize = boundedRange.size();
        if (boundedRange.lower < lowerRangeBound) {
            boundedRange.lower = lowerRangeBound;
            boundedRange.upper = lowerRangeBound+oldSize;
        } if (boundedRange.upper > upperRangeBound) {
            boundedRange.lower = upperRangeBound - oldSize;
            boundedRange.upper = upperRangeBound;
        }
    }
    qcp->yAxis->setRange(boundedRange);
}
void frameview_widget::colorMapScrolledX(const QCPRange &newRange)
{
    /*! \brief Controls the behavior of zooming the plot.
     * \param newRange Mouse wheel scrolled range.
     * Color Maps must not allow the user to zoom past the dimensions of the frame.
     */
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = frWidth-1;
    if (boundedRange.size() > upperRangeBound - lowerRangeBound) {
        boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    } else {
        double oldSize = boundedRange.size();
        if (boundedRange.lower < lowerRangeBound) {
            boundedRange.lower = lowerRangeBound;
            boundedRange.upper = lowerRangeBound + oldSize;
        }
        if (boundedRange.upper > upperRangeBound) {
            boundedRange.lower = upperRangeBound - oldSize;
            boundedRange.upper = upperRangeBound;
        }
    }
    qcp->xAxis->setRange(boundedRange);
}
void frameview_widget::setScrollX(bool Yenabled)
{
    scrollYenabled = !Yenabled;
    qcp->setInteraction(QCP::iRangeDrag, true);
    qcp->setInteraction(QCP::iRangeZoom, true);

    if (!scrollYenabled && scrollXenabled) {
        qcp->axisRect()->setRangeZoom(Qt::Horizontal);
        qcp->axisRect()->setRangeDrag(Qt::Horizontal);
    } else if (scrollXenabled && scrollYenabled) {
        qcp->axisRect()->setRangeZoom(Qt::Horizontal | Qt::Vertical);
        qcp->axisRect()->setRangeDrag(Qt::Horizontal | Qt::Vertical);
    } else if (!scrollXenabled && scrollYenabled) {
        qcp->axisRect()->setRangeZoom(Qt::Vertical);
        qcp->axisRect()->setRangeDrag(Qt::Vertical);
    } else {
        qcp->setInteraction(QCP::iRangeDrag, false);
        qcp->setInteraction(QCP::iRangeZoom, false);
    }

}
void frameview_widget::setScrollY(bool Xenabled)
{
    scrollXenabled = !Xenabled;
    qcp->setInteraction(QCP::iRangeDrag, true);
    qcp->setInteraction(QCP::iRangeZoom, true);
    if (!scrollXenabled && scrollYenabled) {
        qcp->axisRect()->setRangeZoom(Qt::Vertical);
        qcp->axisRect()->setRangeDrag(Qt::Vertical);
    } else if (scrollXenabled && scrollYenabled) {
        qcp->axisRect()->setRangeZoom(Qt::Horizontal | Qt::Vertical);
        qcp->axisRect()->setRangeDrag(Qt::Horizontal | Qt::Vertical);
    } else if (scrollXenabled && !scrollYenabled) {
        qcp->axisRect()->setRangeZoom(Qt::Horizontal);
        qcp->axisRect()->setRangeDrag(Qt::Horizontal);
    } else {
        qcp->setInteraction(QCP::iRangeDrag, false);
        qcp->setInteraction(QCP::iRangeZoom, false);

    }
}

void frameview_widget::updateCeiling(int c)
{
    /*! \brief Change the value of the ceiling for this widget to the input parameter and replot the color scale. */
    ceiling = (double)c;
    rescaleRange();
}

void frameview_widget::updateFloor(int f)
{
    /*! \brief Change the value  of the floor for this widget to the input parameter and replot the color scale. */
    floor = (double)f;
    rescaleRange();
}

void frameview_widget::setUseDSF(bool useDSF)
{
    this->useDSF = useDSF;
}

void frameview_widget::rescaleRange()
{
    /*! \brief Set the color scale of the display to the last used values for this widget */
    colorScale->setDataRange(QCPRange(floor,ceiling));
}

void frameview_widget::setCrosshairs(QMouseEvent *event)
{
    /*! \brief Sets the value of the crosshair and lines to average selection for rendering
     * \author Jackie Ryan */
    fw->displayCross = displayCrosshairCheck.isChecked();
    fw->setCrosshairBackend(qcp->xAxis->pixelToCoord(event->pos().x()), qcp->yAxis->pixelToCoord(event->pos().y()));
}
