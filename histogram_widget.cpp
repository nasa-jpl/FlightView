#include "histogram_widget.h"


histogram_widget::histogram_widget(frameWorker *fw, QWidget *parent) :
    QWidget(parent)
{
    ceiling = 10000;
    floor = 0;
    qcp = NULL;
    this->fw = fw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);

    histogram = new QCPBars(qcp->xAxis, qcp->yAxis);
    qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);
    qcp->addPlottable(histogram);

    double penWidth = .064; //This value was derived to make the bars look the best in the 2-4 range
    histogram->setWidth(penWidth);
    histogram->setName("Histogram of Standard Deviation per pixel");
    const uint16_t sigma = 0x03C3;
    qcp->xAxis->setLabel(QString::fromUtf16(&sigma,1));
    qcp->yAxis->setLabel("Spatial Frequency");

    std::array<float,NUMBER_OF_BINS> histbinvals = getHistogramBinValues();
    histo_bins = QVector<double>(NUMBER_OF_BINS);

    for(unsigned int i = 0; i < NUMBER_OF_BINS; i++)
    {
        histo_bins[i]  = histbinvals[i];
    }
    histo_data_vec = QVector<double>(NUMBER_OF_BINS);

    histogram->keyAxis()->setRangeUpper(histo_bins[histo_bins.size()-1]);
    histogram->keyAxis()->setScaleType(QCPAxis::stLogarithmic);
    histogram->keyAxis()->setScaleLogBase(2);
    histogram->valueAxis()->setRange(QCPRange(0,ceiling));
    connect(histogram->keyAxis(),SIGNAL(rangeChanged(QCPRange)),this,SLOT(histogramScrolledX(QCPRange)));
    connect(histogram->valueAxis(),SIGNAL(rangeChanged(QCPRange)),this,SLOT(histogramScrolledY(QCPRange)));

    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);

    connect(&rendertimer,SIGNAL(timeout()),this,SLOT(handleNewFrame()));
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}
void histogram_widget::handleNewFrame()
{
    if(!this->isHidden())
    {
        uint32_t * histogram_data_ptr = fw->std_dev_frame->std_dev_histogram;
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
    qcp->yAxis->setRange(QCPRange(floor,ceiling));
}
void histogram_widget::histogramScrolledX(const QCPRange &newRange)
{
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;


    double upperRangeBound = histo_bins[histo_bins.size()-1];

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
    histogram->keyAxis()->setRange(boundedRange);
}
void histogram_widget::updateCeiling(int c)
{
    ceiling = (double)c;
    qcp->yAxis->setRange(QCPRange(floor,ceiling));
}
void histogram_widget::updateFloor(int f)
{
    floor = (double)f;
    qcp->yAxis->setRange(QCPRange(floor,ceiling));
}
double histogram_widget::getCeiling()
{
    return ceiling;
}
double histogram_widget::getFloor()
{
    return floor;
}
void histogram_widget::rescaleRange()
{
    qcp->yAxis->setRange(QCPRange(floor,ceiling));
}
