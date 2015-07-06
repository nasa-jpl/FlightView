#include "histogram_widget.h"


histogram_widget::histogram_widget(frameWorker *fw, image_t image_type, QWidget *parent) :
    QWidget(parent)
{
    ceiling = 8000;
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


    //double penWidth = (0xFFFF/(double)NUMBER_OF_BINS);
    double penWidth = .064; //This value was derived to make the bars look the best in the 2-4 range
    histogram->setWidth(penWidth);
    histogram->setName("Histogram of Standard Deviation per pixel");
    const uint16_t sigma = 0x03C3;
    qcp->xAxis->setLabel(QString::fromUtf16(&sigma,1));
    qcp->yAxis->setLabel("Spatial Frequency");
    //histogram->setPen(pen);

    std::array<float,NUMBER_OF_BINS> histbinvals = getHistogramBinValues();
    histo_bins = QVector<double>(NUMBER_OF_BINS);

    for(unsigned int i = 0; i < NUMBER_OF_BINS; i++)
    {
        histo_bins[i]  = histbinvals[i];
    }
    histo_data_vec = QVector<double>(NUMBER_OF_BINS);
    double bar_width = histo_bins[3]-histo_bins[2]; //Probably not the best way to get the bar width, but w/e



    histogram->keyAxis()->setRangeUpper(histo_bins[histo_bins.size()-1]);
    histogram->keyAxis()->setScaleType(QCPAxis::stLogarithmic);
    //histogram->keyAxis()->setNumberFormat("b");
    histogram->keyAxis()->setScaleLogBase(2);
    //histogram->keyAxis()->setAutoTicks(true);
    //histogram->keyAxis()->setTickLabelType(QCPAxis::12);
    //histogram->keyAxis()->setAutoTickCount(10);
    //histogram->keyAxis()->setRangeUpper(10);

    histogram->valueAxis()->setRange(QCPRange(0,ceiling));

    //qcp->xAxis->setTickVector(histo_bins);
    //histogram->keyAxis()->setSca

    //    /qcp->xAxis->setRange(QCPRange(0.0d,100.0d));
    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);


    connect(histogram->keyAxis(),SIGNAL(rangeChanged(QCPRange)),this,SLOT(histogramScrolledX(QCPRange)));
    connect(histogram->valueAxis(),SIGNAL(rangeChanged(QCPRange)),this,SLOT(histogramScrolledY(QCPRange)));
    connect(&rendertimer,SIGNAL(timeout()),this,SLOT(handleNewFrame()));
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}

histogram_widget::~histogram_widget()
{

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
    //do Nothing!
    /*
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;

    double upperRangeBound = fw->histoDataMax*1.3;

    boundedRange = QCPRange(lowerRangeBound, upperRangeBound);

    histogram->valueAxis()->setRange(boundedRange);
    */
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
    //histogram->keyAxis()->setAutoTickCount(10);
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
