#include "profile_widget.h"
#include "settings.h"
/* #define QDEBUG */

profile_widget::profile_widget(frameWorker *fw, image_t image_type, QWidget *parent) :
    QWidget(parent)
{
    /*! \brief Establishes a plot for a specified image type.
     * \param image_type Determines the type of graph that will be output by profile_widget
     * \author JP Ryan
     * \author Noah Levy */
    itype = image_type;
    qcp = NULL;
    this->fw = fw;
    ceiling = fw->base_ceiling;
    floor = 0;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    x_coord = -1;
    y_coord = -1;

    qcp = new QCustomPlot(this);
    qcp->addLayer("Plot Layer");
    qcp->setCurrentLayer("Plot Layer");
    qcp->setNotAntialiasedElement(QCP::aeAll);

    qcp->plotLayout()->insertRow(0);
    plotTitle = new QCPPlotTitle(qcp);
    qcp->plotLayout()->addElement(0, 0, plotTitle);
    qcp->addGraph();

    if (itype == VERTICAL_MEAN || itype == VERTICAL_CROSS) {
        xAxisMax = frHeight;
        qcp->xAxis->setLabel("Y index");
    } else if (itype == HORIZONTAL_MEAN || itype == HORIZONTAL_CROSS) {
        xAxisMax = frWidth;
        qcp->xAxis->setLabel("X index");
    }
    x = QVector<double>(xAxisMax);
    for (int i = 0; i < xAxisMax; i++)
        x[i] = double(i);
    y = QVector<double>(xAxisMax);
    qcp->xAxis->setRange(QCPRange(0, xAxisMax));

    qcp->addLayer("Box Layer", qcp->currentLayer());
    qcp->setCurrentLayer("Box Layer");
    callout = new QCPItemText(qcp);
    qcp->addItem(callout);
    callout->position->setCoords(xAxisMax / 2, ceiling - 1000);
    callout->setFont(QFont(font().family(), 16));
    callout->setPen(QPen(Qt::black));
    callout->setBrush(Qt::white);
    qcp->setSelectionTolerance(100);
    callout->setSelectedBrush(Qt::white);
    callout->setSelectedFont(QFont(font().family(), 16));
    callout->setSelectedPen(QPen(Qt::black));
    callout->setSelectedColor(Qt::black);
    callout->setVisible(false);
    qcp->addLayer("Arrow Layer", qcp->currentLayer(), QCustomPlot::limBelow);
    qcp->setCurrentLayer("Arrow Layer");
    arrow = new QCPItemLine(qcp);
    qcp->addItem(arrow);
    arrow->start->setParentAnchor(callout->bottom);
    arrow->setHead(QCPLineEnding::esSpikeArrow);
    arrow->setSelectable(false);
    arrow->setVisible(false);
    qcp->setInteractions(QCP::iRangeZoom | QCP::iSelectItems);

    qcp->yAxis->setLabel("Pixel Magnitude [DN]");
    qcp->yAxis->setRange(QCPRange(0, fw->base_ceiling)); //From 0 to 2^16

    qcp->graph(0)->setData(x, y);

    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);

    connect(qcp, SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(moveCallout(QMouseEvent*)));
    connect(qcp, SIGNAL(mouseDoubleClick(QMouseEvent*)), this, SLOT(setCallout(QMouseEvent*)));
    connect(qcp->xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(profileScrolledX(QCPRange)));
    connect(qcp->yAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(profileScrolledY(QCPRange)));
    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));

    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}

// public functions
double profile_widget::getFloor()
{
    /*! \brief Return the value of the floor for this widget as a double */
    return floor;
}
double profile_widget::getCeiling()
{
    /*! \brief Return the value of the ceiling for this widget as a double */
    return ceiling;
}
void profile_widget::hide_callout()
{
    if (callout->visible() || fw->crosshair_x == -1) {
        callout->setVisible(false);
        arrow->setVisible(false);
    } else {

        callout->setVisible(true);
        arrow->setVisible(true);
    }
}

// public slots
void profile_widget::handleNewFrame()
{
    /*! \brief Plots a specific dimension profile.
     * \paragraph
     * The switch statement is a bit silly here, I only use it to differentiate the plot title and the type of profile array to use.
     * The y-axis data is reversed in these images.
     * \author JP Ryan
     */
    float *local_image_ptr;
    bool whichDim = itype == VERTICAL_CROSS || itype == VERTICAL_MEAN; // vertical profiles are true, horizontal profiles are false
    bool isMeanProfile = itype == VERTICAL_MEAN || itype == HORIZONTAL_MEAN;
    if (!this->isHidden() &&  fw->curFrame != NULL && ((fw->crosshair_x != -1 && fw->crosshair_y) || isMeanProfile)) {
        allow_callouts = true;
        local_image_ptr = whichDim ? fw->curFrame->vertical_mean_profile : fw->curFrame->horizontal_mean_profile;
        if (whichDim)
            // vertical profiles

            for (int r = 1; r < frHeight; r++)
                y[r] = double(local_image_ptr[frHeight - r]);
        else
            // horizontal profiles
            for (int c = 0; c < frWidth; c++)
                y[c] = double(local_image_ptr[c]);
        qcp->graph(0)->setData(x, y);
        qcp->replot();
        if (callout->visible())
            updateCalloutValue();
        switch (itype) {
        case HORIZONTAL_MEAN: plotTitle->setText(QString("Horizontal Mean Profile")); break;
        case HORIZONTAL_CROSS: plotTitle->setText(QString("Horizontal Profile centered @ y = %1").arg(fw->crosshair_y)); break;
        case VERTICAL_MEAN: plotTitle->setText(QString("Vertical Mean Profile")); break;
        case VERTICAL_CROSS: plotTitle->setText(QString("Vertical Profile centered @ x = %1").arg(fw->crosshair_x)); break;
        default: break;
        }
    } else {
        plotTitle->setText("No Crosshair designated");
        allow_callouts = false;
        qcp->graph(0)->clearData();
        qcp->replot();
    }
}
void profile_widget::updateCeiling(int c)
{
    /*! \brief Change the value of the ceiling for this widget to the input parameter and replot the color scale. */
    ceiling = (double)c;
    rescaleRange();
}
void profile_widget::updateFloor(int f)
{
    /*! \brief Change the value of the floor for this widget to the input parameter and replot the color scale. */
    floor = (double)f;
    rescaleRange();
}
void profile_widget::rescaleRange()
{
    /*! \brief Set the color scale of the display to the last used values for this widget */
    qcp->yAxis->setRange(QCPRange(floor, ceiling));
}
void profile_widget::profileScrolledX(const QCPRange &newRange)
{
    /*! \brief Controls the behavior of zooming the plot.
     * \param newRange Unused.
     * Profiles must not allow the user to zoom in the x direction.
     */
    Q_UNUSED(newRange);
    qcp->xAxis->setRange(0, xAxisMax);
}
void profile_widget::profileScrolledY(const QCPRange &newRange)
{
    /*! \brief Controls the behavior of zooming the plot.
     * \param newRange Mouse wheel scrolled range.
     * Profiles must not allow the user to zoom past the dimensions of the frame.
     */
    QCPRange boundedRange = newRange;
    double lowerRangeBound = slider_low_inc ? LIL_MIN : BIG_MIN;
    double upperRangeBound = slider_low_inc ? LIL_MAX : BIG_MAX;
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
    floor = boundedRange.lower;
    ceiling = boundedRange.upper;
    qcp->yAxis->setRange(boundedRange);
}
void profile_widget::setCallout(QMouseEvent *e)
{
    x_coord = qcp->xAxis->pixelToCoord(e->pos().x());
    x_coord = x_coord < 0 ? 0 : x_coord;
    x_coord = x_coord > xAxisMax ? xAxisMax : x_coord;
    y_coord = y[x_coord];
    if (callout->position->coords().y() > ceiling || callout->position->coords().y() < floor)
        callout->position->setCoords(callout->position->coords().x(), (ceiling - floor) * 0.9 + floor);
    arrow->end->setCoords(x_coord, y_coord);
    callout->setText(QString(" x: %1 \n y: %2 ").arg(x_coord).arg(y_coord));
    if(allow_callouts) {
        callout->setVisible(true);
        arrow->setVisible(true);
    }
}
void profile_widget::moveCallout(QMouseEvent *e)
{
    if ((callout->selectTest(e->posF(), true) < (0.99 * qcp->selectionTolerance())) && (e->buttons() & Qt::LeftButton)) {
        callout->position->setPixelPoint(e->posF());
    } else {
        return;
    }
}

// private slots
void profile_widget::updateCalloutValue()
{
    y_coord = y[x_coord];
    arrow->end->setCoords(x_coord, y_coord);
    callout->setText(QString(" x: %1 \n y: %2 ").arg(x_coord).arg(y_coord));
}

