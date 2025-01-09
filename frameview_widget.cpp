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
    this->setObjectName("lv:frameview");
    floor = 0;
    useDSF = false;
    havePrefs = false;
    prefs = NULL;
    options = fw->getStartupOptions();
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    bool ok = false;

    switch(image_type) {
    case BASE:
        //ceiling = fw->base_ceiling;
        //break;
    case DSF:
        //ceiling = 100;
        peakValueHolder = (float*)calloc(frWidth * frHeight, sizeof(float));
        peakHoldChk = new QCheckBox();
        clearPeaksBtn = new QPushButton();
        peakHoldChk->setText("Peak Hold");
        clearPeaksBtn->setText("Clear Peaks");
        connect(peakHoldChk, SIGNAL(toggled(bool)), this, SLOT(setPeakHoldMode(bool)));
        connect(clearPeaksBtn, SIGNAL(pressed()), this, SLOT(clearPeaks()));
        break;
    case STD_DEV:
        ceiling = 102;
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
    qcp->xAxis->setLabel("X (Spatial)");
    if(image_type == WATERFALL) {
        qcp->yAxis->setLabel("Y (Time)");
    } else {
        qcp->yAxis->setLabel("Y (Spectral)");
    }
    qcp->yAxis->setRangeReversed(true);
    //If this is uncommented, window size reflects focal plane size, otherwise it scales
    //qcp->setBackgroundScaled(Qt::AspectRatioMode);

    colorMap = new QCPColorMap(qcp->xAxis,qcp->yAxis);
    colorMapData = NULL;

    qcp->addPlottable(colorMap);

    colorScale = new QCPColorScale(qcp);
    ok = qcp->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    if(!ok) {
        sMessage("Error, could not add element to frameview plot.");
        qDebug() << "Error, could not add element to frameview plot.";
    }
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

    fpsLabel.setText("FPS of Display");


    wfSelectedRow.setText("ROW NOT SET");
    QRect fpsGeo = fpsLabel.geometry();
    fpsGeo.setWidth(25);
    fpsLabel.setGeometry(fpsGeo);
    layout.addWidget(&fpsLabel, 8, 0, 1, 2);
    if( (image_type == DSF) || (image_type==BASE)) {
        layout.addWidget(peakHoldChk, 9,0,1,1);
        layout.addWidget(clearPeaksBtn, 9,1,1,1);
    }

    if (!((image_type == STD_DEV) || (image_type == WATERFALL))) {
        layout.addWidget(&displayCrosshairCheck, 8, 2, 1, 2);
    } else if (image_type==WATERFALL) {
        layout.addWidget(&wfSelectedRow, 8,2,1,2);
    }

    layout.addWidget(&zoomXCheck, 8, 4, 1, 2);
    layout.addWidget(&zoomYCheck, 8, 6, 1, 2);
    this->setLayout(&layout);

    displayCrosshairCheck.setText(tr("Display Crosshairs on Frame"));
    displayCrosshairCheck.setChecked(true);

    if ( (image_type == STD_DEV) || (image_type == WATERFALL)) {
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

    connect(colorScale, SIGNAL(dataRangeChanged(QCPRange)), this, SLOT(colorScaleRangeChanged(QCPRange)));

    connect(&displayCrosshairCheck, SIGNAL(toggled(bool)), fw, SLOT(updateCrossDiplay(bool)));
    connect(&zoomXCheck, SIGNAL(toggled(bool)), this, SLOT(setScrollX(bool)));
    connect(&zoomYCheck, SIGNAL(toggled(bool)), this, SLOT(setScrollY(bool)));
    connect(fw, SIGNAL(setColorScheme_signal(int, bool)), this, SLOT(handleNewColorScheme(int, bool)));

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
    if(options.headless) {
        sMessage("Frameview display disabled due to headless mode.");
    } else {
        rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
    }
    sMessage(QString("%1: Finished constructor.").arg(QString(__PRETTY_FUNCTION__)));
    //qDebug() << __PRETTY_FUNCTION__ << ": Finished constructor";
}
frameview_widget::~frameview_widget()
{
    /*! \brief Deallocate QCustomPlot elements */
    delete qcp;
}

// public functions

void frameview_widget::setPrefsPtr(settingsT *prefsPtr)
{
    if(prefsPtr)
    {
        this->prefs = prefsPtr;
        this->havePrefs = true;
    } else {
        this->havePrefs = false;
    }
}

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

void frameview_widget::useDarkTheme(bool useDark)
{
    QCustomPlot *p = this->qcp;

    if(useDark)
    {
        p->setBackground(QBrush(Qt::black));
        p->xAxis->setBasePen(QPen(Qt::green)); // lower line of axis
        p->yAxis->setBasePen(QPen(Qt::green)); // left line of axis
        p->xAxis->setTickPen(QPen(Qt::green));
        p->yAxis->setTickPen(QPen(Qt::green));
        p->xAxis->setLabelColor(QColor(Qt::white));
        p->yAxis->setLabelColor(QColor(Qt::white));
        p->yAxis->setTickLabelColor(Qt::white);
        p->xAxis->setTickLabelColor(Qt::white);
        p->legend->setBrush(QBrush(Qt::black));
        p->legend->setTextColor(Qt::white);
        colorMap->setBrush(QBrush(Qt::black));
        colorMap->setPen(QPen(Qt::white));
        colorMap->valueAxis()->setLabelColor(Qt::white);
        colorMap->valueAxis()->setBasePen(QPen(Qt::white));
        colorMap->valueAxis()->setTickLabelColor(Qt::white);
        colorMap->keyAxis()->setLabelColor(Qt::white);
        colorMap->keyAxis()->setBasePen(QPen(Qt::white));
        colorMap->keyAxis()->setTickLabelColor(Qt::white);
        colorMap->keyAxis()->setTickPen(QPen(Qt::white));

        colorMap->colorScale()->axis()->setLabelColor(Qt::white);
        colorMap->colorScale()->axis()->setBasePen(QPen(Qt::black)); // line on RH side of scale
        colorMap->colorScale()->axis()->setTickLabelColor(Qt::white); // TEXT!!
    } else {
        p->setBackground(QBrush(Qt::white));
        p->xAxis->setBasePen(QPen(Qt::black)); // lower line of axis
        p->yAxis->setBasePen(QPen(Qt::black)); // left line of axis
        p->xAxis->setTickPen(QPen(Qt::black));
        p->yAxis->setTickPen(QPen(Qt::black));
        p->xAxis->setLabelColor(QColor(Qt::black));
        p->yAxis->setLabelColor(QColor(Qt::black));
        p->legend->setBrush(QBrush(Qt::white));
        p->legend->setTextColor(Qt::black);
        p->yAxis->setTickLabelColor(Qt::black);
        p->xAxis->setTickLabelColor(Qt::black);
        colorMap->setBrush(QBrush(Qt::white));
        colorMap->setPen(QPen(Qt::black));
        colorMap->valueAxis()->setLabelColor(Qt::black);
        colorMap->valueAxis()->setBasePen(QPen(Qt::black));
        colorMap->valueAxis()->setTickLabelColor(Qt::black);
        colorMap->keyAxis()->setLabelColor(Qt::black);
        colorMap->keyAxis()->setBasePen(QPen(Qt::black));
        colorMap->keyAxis()->setTickLabelColor(Qt::black);
        colorMap->keyAxis()->setTickPen(QPen(Qt::black));

        colorMap->colorScale()->axis()->setLabelColor(Qt::black);
        colorMap->colorScale()->axis()->setBasePen(QPen(Qt::black)); // line on RH side of scale
        colorMap->colorScale()->axis()->setTickLabelColor(Qt::black); // TEXT!!
    }
}


void frameview_widget::handleNewColorScheme(int scheme, bool useDarkThemeVal)
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
    case 11:
        colorMap->setGradient((QCPColorGradient::gpRedTop));
        break;
    default:
        sMessage(QString("color scheme [%1] not recognized.").arg(fw->color_scheme));
        colorMap->setGradient(QCPColorGradient::gpJet);
        break;
    }

    useDarkTheme(useDarkThemeVal);
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

    QMutexLocker locker(&this->drawMutex);

    if(fw->curFrame == NULL)
        return;

    if(fw->curFrame->image_data_ptr == NULL)
        return;

    if(image_type == WATERFALL)
    {
        // Copy waterfall data in, even if hidden:
        int row = fw->crosshair_y;
        if(row < 0)
            return;

        wfSelectedRow.setText(QString("Row: %1").arg(fw->crosshair_y));

        float *local_image_ptr = fw->curFrame->dark_subtracted_data;
        uint16_t* local_image_ptr_uint = fw->curFrame->image_data_ptr;

        std::vector <float> line;
        if(useDSF) {
            for(int col = 0; col < frWidth; col++)
            {
                line.push_back(local_image_ptr[row * frWidth + col]);
            }
        } else {
            for(int col = 0; col < frWidth; col++)
            {
                line.push_back(local_image_ptr_uint[row * frWidth + col]);
            }
        }
        // There's a better way, but for now this will be ok:
        wfimage.push_front(line); // Append to top
        // I have seen a crash here even when wflength was defined as 1024:
        // crash was when only items 0-420 existed
        // Other items were "not accessable"
        if(wfimage.size() > (unsigned long)wflength) {
            wfimage.resize(wflength); // Cut off anything too small.
            wfimage.shrink_to_fit();
        }
        if(this->isHidden())
            return;

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
        goto done_here;

    }

    if(!this->isHidden() && (fw->curFrame->image_data_ptr != NULL)) {


        if((image_type == DSF) || (image_type==BASE)) {
            uint16_t* local_image_ptr_uint = fw->curFrame->image_data_ptr;
            float* local_image_ptr_float = fw->curFrame->dark_subtracted_data;

            if(useDSF)
            {
                if(peakHoldMode) {
                    // DSF,
                    // Peak Hold Mode
                    for(int col = 0; col < frWidth; col++) {
                        for(int row = 0; row < frHeight; row++) {
                            // colorMap->data()->setCell(col, row, local_image_ptr[(frHeight - row - 1) * frWidth + col]); // y-axis reversed
                            // Check for higher peak value:
                            if(local_image_ptr_float[row * frWidth + col] > peakValueHolder[row * frWidth + col]) {
                                peakValueHolder[row * frWidth + col] = local_image_ptr_float[row * frWidth + col];
                            }
                            // Draw data from the peak hold array:
                            colorMap->data()->setCell(col, row, peakValueHolder[row * frWidth + col]); // y-axis NOT reversed
                        }
                    }
                } else {
                    // DSF
                    // NOT Peak Hold Mode:
                    for(int col = 0; col < frWidth; col++) {
                        for(int row = 0; row < frHeight; row++) {
                            // Draw the image:
                            // colorMap->data()->setCell(col, row, local_image_ptr[(frHeight - row - 1) * frWidth + col]); // y-axis reversed
                            colorMap->data()->setCell(col, row, local_image_ptr_float[row * frWidth + col]); // y-axis NOT reversed
                        }
                    }
                }
            } else {
                // Not DSF
                if(peakHoldMode) {
                    for(int col = 0; col < frWidth; col++) {
                        for(int row = 0; row < frHeight; row++) {
                            // Check for a potential new peak value:
                            if(local_image_ptr_uint[row * frWidth + col] > peakValueHolder[row * frWidth + col]) {
                                peakValueHolder[row * frWidth + col] = local_image_ptr_uint[row * frWidth + col];
                            }
                            // Set the data into the image, from the peakValueHolder array:
                            colorMap->data()->setCell(col, row, peakValueHolder[row * frWidth + col]);
                        }
                    }
                } else {
                    // Not in peak hold mode:
                    for(int col = 0; col < frWidth; col++) {
                        for(int row = 0; row < frHeight; row++) {
                            // colorMap->data()->setCell(col, row, local_image_ptr_uint[(frHeight - row - 1) * frWidth + col]); // y-axis reversed
                            colorMap->data()->setCell(col, row, local_image_ptr_uint[row * frWidth + col]); // y-axis NOT reversed
                        }
                    }
                }
            }

            if(fw->displayCross) {
                int row=fw->crosshair_y;
                if(row < frHeight) {
                    for(int col = 0; col < frWidth; col++) {
                        colorMap->data()->setCell(col, row, NAN);
                    }
                }
                int col = fw->crosshair_x;
                if(col < frWidth) {
                    for(int row = 0; row < frHeight; row++) {
                        colorMap->data()->setCell(col, row, NAN);
                    }
                }
                if(isOverlayImage) {
                    // draw the other ones as well:
                    row = fw->crossStartRow;
                    for(int col = 0; col < frWidth; col++) {
                        colorMap->data()->setCell(col, row, NAN);
                    }
                    row = fw->crossHeight;
                    for(int col = 0; col < frWidth; col++) {
                        colorMap->data()->setCell(col, row, NAN);
                    }
                    col = fw->crossStartCol;
                    for(int row = 0; row < frHeight; row++) {
                        colorMap->data()->setCell(col, row, NAN);
                    }
                    col = fw->crossWidth;
                    for(int row = 0; row < frHeight; row++) {
                        colorMap->data()->setCell(col, row, NAN);
                    }
                }
            }
            if(drawrgbRow) {
                int ifloor = (int)this->floor;
                int iceiling = (int)this->ceiling;
                int imid = (iceiling - ifloor)/2;

                if( (redRow < frHeight) && (greenRow < frHeight) && (blueRow < frHeight) )
                {
                    for(int col=0; col < frWidth; col++)
                    {
                        colorMap->data()->setCell(col, redRow, iceiling);
                        colorMap->data()->setCell(col, ((redRow+1)<frHeight)?redRow+1:redRow-1, iceiling);

                        colorMap->data()->setCell(col, greenRow, imid);
                        colorMap->data()->setCell(col, ((greenRow+1)<frHeight)?greenRow+1:greenRow-1, imid);

                        colorMap->data()->setCell(col, blueRow, ifloor);
                        colorMap->data()->setCell(col, ((blueRow+1)<frHeight)?blueRow+1:blueRow-1, ifloor);
                    }

                    for(int col=0; col < frWidth/10; col++)
                    {
                        colorMap->data()->setCell(col, redRow, ((col%2)==0)?iceiling:ifloor);
                        colorMap->data()->setCell(col, ((redRow+1)<frHeight)?redRow+1:redRow-1, ((col%2)==0)?iceiling:ifloor);

                        colorMap->data()->setCell(col, greenRow, ((col%2)==0)?iceiling:ifloor);
                        colorMap->data()->setCell(col, ((greenRow+1)<frHeight)?greenRow+1:greenRow-1, ((col%2)==0)?iceiling:ifloor);

                        colorMap->data()->setCell(col, blueRow, ((col%2)==0)?iceiling:ifloor);
                        colorMap->data()->setCell(col, ((blueRow+1)<frHeight)?blueRow+1:blueRow-1, ((col%2)==0)?iceiling:ifloor);
                    }


                }
            }
            qcp->replot();
            goto done_here;

        } // end if DSF or BASE

        if(image_type == STD_DEV && fw->std_dev_frame != NULL) {
            float * local_image_ptr = fw->std_dev_frame->std_dev_data;
            for (int col = 0; col < frWidth; col++)
                for (int row = 0; row < frHeight; row++)
                    // colorMap->data()->setCell(col, row, (double_t)local_image_ptr[(frHeight - row - 1) * frWidth + col]); // y-axis reversed
                    colorMap->data()->setCell(col, row, local_image_ptr[row * frWidth + col]); // y-axis NOT reversed
            qcp->replot(); // crash??
            goto done_here;
        }
    }

    return;


done_here:
    count++;
    if (count % 20 == 0 && count != 0) {
        fps = 20.0 / clock.restart() * 1000.0;
        fps_string = QString::number(fps, 'f', 1);
        fpsLabel.setText(QString("FPS of Display: %1").arg(fps_string));
    }
}

void frameview_widget::colorMapScrolledY(const QCPRange &newRange)
{
    /*! \brief Controls the behavior of zooming or panning the frame image.
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
    /*! \brief Controls the behavior of zooming or panning the frame image.
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

void frameview_widget::colorScaleRangeChanged(const QCPRange &newRange) {
    // This is called when the color scale itself is scrolled or dragged.
    // Interaction within QCP is already happening,
    // so the only thing we need to do is update the local
    // ceiling and floor values, UI elements, and preferences.
    //emit statusMessage("Color Scale Range Changed.");
    emit haveFloorCeilingValuesFromColorScaleChange(newRange.lower, newRange.upper);
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
    //    if(prefs)
    //    {
    //        switch(image_type)
    //        {
    //        case(DSF):
    //            prefs->dsfCeiling = c;
    //            break;
    //        case(BASE):
    //            if(useDSF)
    //            {
    //                prefs->dsfCeiling = c;
    //            } else {
    //                prefs->frameViewCeiling = c;
    //            }
    //            break;

    //        case(STD_DEV):
    //            prefs->stddevCeiling = c;
    //            break;

    //        default:
    //            break;
    //        }
    //    }

    ceiling = (double)c;
    rescaleRange();
}

void frameview_widget::updateFloor(int f)
{
    /*! \brief Change the value  of the floor for this widget to the input parameter and replot the color scale. */
    //    if(prefs)
    //    {
    //        switch(image_type)
    //        {
    //        case(DSF):
    //            prefs->dsfFloor = f;
    //            break;

    //        case(BASE):
    //            if(useDSF)
    //            {
    //                prefs->dsfFloor = f;
    //            } else {
    //                prefs->frameViewFloor = f;
    //            }
    //            break;

    //        case(STD_DEV):
    //            prefs->stddevFloor = f;
    //            break;

    //        default:
    //            break;
    //        }
    //    }
    floor = (double)f;
    rescaleRange();
}

void frameview_widget::clearPeaks() {

    for(int col = 0; col < frWidth; col++)
    {
        for(int row = 0; row < frHeight; row++) {
            peakValueHolder[row * frWidth + col] = 0;
        }
    }
}

void frameview_widget::setPeakHoldMode(bool hold) {
    this->peakHoldMode = hold;
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

void frameview_widget::toggleDrawRGBRow(bool draw)
{
    this->drawrgbRow = draw;
    colorMap->setInterpolate(draw);
    colorMap->setAntialiased(draw);
}

void frameview_widget::setIsOverlayImage(bool isOverlay) {
    this->isOverlayImage = isOverlay;
    if(fw->frWidth > 1280) {
        rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS*1.30); // lower FPS due to more things being drawn
    } else {
        rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
    }
}

void frameview_widget::showRGB(int r, int g, int b)
{
    this->redRow = r;
    this->greenRow = g;
    this->blueRow = b;
    this->toggleDrawRGBRow(true);
    //sMessage(QString("Showing RGB for r: %1, g: %2, b: %3").arg(r).arg(g).arg(b));
}

void frameview_widget::sMessage(QString statusMessageText)
{
    statusMessageText.prepend("[frameview_widget]: ");
    emit statusMessage(statusMessageText);
}
