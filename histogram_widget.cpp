#include "histogram_widget.h"

histogram_widget::histogram_widget(frameWorker *fw, image_t image_type, QWidget *parent) :
    QWidget(parent)
{
    qcp = NULL;
    this->fw = fw;
    fps=0;
}

histogram_widget::~histogram_widget()
{

}
void histogram_widget::initQCPStuff()
{
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    qcp = new QCustomPlot(this);
    histogram = new QCPBars(qcp->xAxis, qcp->yAxis);
    qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);
    qcp->addPlottable(histogram);
    QPen pen;
    pen.setWidthF(1.2);
    histogram->setName("Histogram of Standard Deviation per pixel");
    //histogram->setPen(pen);

    std::vector<float> copy_bins(*(fw->getHistogramBins()));
    histo_bins = QVector<double>(copy_bins.size());

    for(int i = 0; i < histo_bins.size(); i++)
    {
        histo_bins[i]  = copy_bins[i];
    }
    double bar_width = copy_bins[3]-copy_bins[2]; //Probably not the best way to get the bar width, but w/e



    histogram->keyAxis()->setRangeUpper(histo_bins[histo_bins.size()-1]);
    histogram->keyAxis()->setScaleType(QCPAxis::stLogarithmic);
    histogram->keyAxis()->setNumberFormat("b");
    histogram->keyAxis()->setScaleLogBase(2);
    histogram->keyAxis()->setAutoTicks(true);
    //histogram->keyAxis()->setTickLabelType(QCPAxis::12);
    histogram->keyAxis()->setAutoTickCount(10);
    //histogram->keyAxis()->setRangeUpper(10);

    histogram->valueAxis()->setRange(QCPRange(0,fw->histoDataMax));

    //qcp->xAxis->setTickVector(histo_bins);
    //histogram->keyAxis()->setSca

    histogram->setWidth(.034); //This value was derived experimentally...
    //    /qcp->xAxis->setRange(QCPRange(0.0d,100.0d));
    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);


    connect(histogram->keyAxis(),SIGNAL(rangeChanged(QCPRange)),this,SLOT(histogramScrolledX(QCPRange)));
    connect(histogram->valueAxis(),SIGNAL(rangeChanged(QCPRange)),this,SLOT(histogramScrolledY(QCPRange)));

}


void histogram_widget::handleNewFrame(QSharedPointer<QVector<double>> histo_data_vec)
{
    if(qcp == NULL)
    {
        initQCPStuff();
    }
    if(!this->isHidden())
    {


        histogram->setData(histo_bins,*histo_data_vec);

        qcp->replot();
    }
    fps++;
}

void histogram_widget::histogramScrolledY(const QCPRange &newRange)
{
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;

    double upperRangeBound = fw->histoDataMax*1.3;

    boundedRange = QCPRange(lowerRangeBound, upperRangeBound);

    histogram->valueAxis()->setRange(boundedRange);
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
