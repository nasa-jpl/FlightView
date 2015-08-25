#include "histogram_widget.h"


histogram_widget::histogram_widget(frameWorker *fw, QWidget *parent) :
    QWidget(parent)
{
    ceiling = (1<<16) - 1;
    floor = 0;
    qcp = NULL;
    this->fw = fw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);

    histogram = new QCPBars(qcp->xAxis, qcp->yAxis);
    qcp->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    qcp->addPlottable(histogram);

    double penWidth = .064; //This value was derived to make the bars look the best in the 2-4 range
    histogram->setWidth(penWidth);
    histogram->setName("Histogram of Standard Deviation per pixel");
    const uint16_t sigma = 0x03C3;
    qcp->xAxis->setLabel(QString::fromUtf16(&sigma, 1));
    qcp->yAxis->setLabel("Spatial Frequency");

    std::array<float,NUMBER_OF_BINS> histbinvals = getHistogramBinValues();
    histo_bins = QVector<double>(NUMBER_OF_BINS);

    for(unsigned int i = 0; i < NUMBER_OF_BINS; i++)
    {
        histo_bins[i]  = histbinvals[i];
    }
    histo_data_vec = QVector<double>(NUMBER_OF_BINS);

    histogram->keyAxis()->setRangeUpper(histo_bins[histo_bins.size()-1]);
    histogram->keyAxis()->setRangeLower(1);
    histogram->keyAxis()->setScaleType(QCPAxis::stLogarithmic);
    histogram->keyAxis()->setScaleLogBase(2);
    histogram->valueAxis()->setRange(QCPRange(0, ceiling));
    connect(histogram->keyAxis(), SIGNAL(rangeChanged(QCPRange)), this, SLOT(histogramScrolledX(QCPRange)));
    connect(histogram->valueAxis(), SIGNAL(rangeChanged(QCPRange)), this, SLOT(histogramScrolledY(QCPRange)));

    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);

    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}

// public functions
double histogram_widget::getCeiling()
{
    /*! \brief Return the value of the ceiling for this widget as a double */
    return ceiling;
}
double histogram_widget::getFloor()
{
    /*! \brief Return the value of the floor for this widget as a double */
    return floor;
}

// public slots
void histogram_widget::handleNewFrame()
{
    /*! \brief Render the bars of histogram data
     * \paragraph
     *
     * As the histogram relies on standard deviation data, it must use the std_dev_frame rather than the curFrame. */
    if(!this->isHidden() && fw->std_dev_frame != NULL)
    {
        uint32_t *histogram_data_ptr = fw->std_dev_frame->std_dev_histogram;
        for(unsigned int b = 0; b < NUMBER_OF_BINS;b++)
        {
            histo_data_vec[b] = histogram_data_ptr[b];
        }

        histogram->setData(histo_bins,histo_data_vec);

        qcp->replot();
    }
    count++;
}
void histogram_widget::histogramScrolledY(const QCPRange &newRange)
{
    /*! \brief Defines behavior for zooming the Y Axis of the histogram in and out.
     * \param newRange Unused.
     * There is no special behavior for scrolling the Y Axis of the histogram. This function may
     * be deprecated in future versions (can be replaced with qcp->replot()).
     */
    Q_UNUSED(newRange);
    rescaleRange();
}
void histogram_widget::histogramScrolledX(const QCPRange &newRange)
{
    /*! \brief Defines behavior for zooming the Y Axis of the histogram in and out.
     * \param newRange Passed in as the new scroll position on the X axis.
     * The X Axis is log scale, but it must stay within the bounded range from 0 to 8192. Additionally, the bars must be
     * rescaled with the axis.
     */
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    double upperRangeBound = histo_bins[histo_bins.size()-1];

    if (boundedRange.size() > upperRangeBound - lowerRangeBound) {
        boundedRange = QCPRange(lowerRangeBound, upperRangeBound);
    } else {
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
    histogram->keyAxis()->setRange(boundedRange);
}
void histogram_widget::updateCeiling(int c)
{
    /*! \brief Change the value of the ceiling for this widget to the input parameter and replot the color scale. */
    ceiling = (double)c;
    rescaleRange();
}
void histogram_widget::updateFloor(int f)
{
    /*! \brief Change the value of the floor for this widget to the input parameter and replot the color scale. */
    floor = (double)f;
    rescaleRange();
}
void histogram_widget::rescaleRange()
{
    /*! \brief Set the color scale of the display to the last used values for this widget */
    qcp->yAxis->setRange(QCPRange(floor, ceiling));
}
void histogram_widget::resetRange()
{
    /*! \brief Reset the range of the xAxis of the histogram to the initial parameters - 1 to 8192. */
    qcp->xAxis->setRange(QCPRange(1, histo_bins[histo_bins.size() - 1]));
}
