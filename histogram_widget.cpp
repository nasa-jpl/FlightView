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
    frHeight = fw->getHeight();
    frWidth = fw->getWidth();
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
    histo_data = QVector<double>(histo_bins.size());



    histogram->keyAxis()->setRangeUpper(histo_bins[histo_bins.size()-1]);
    histogram->keyAxis()->setScaleType(QCPAxis::stLogarithmic);
    histogram->keyAxis()->setScaleLogBase(2);
    histogram->keyAxis()->setAutoTicks(true);
    histogram->keyAxis()->setAutoTickCount(10);
    histogram->keyAxis()->setRangeUpper(10);

    update_histo_data();
    histogram->valueAxis()->setRange(QCPRange(0,dataMax));

    //qcp->xAxis->setTickVector(histo_bins);
    //histogram->keyAxis()->setSca

    histogram->setWidth(.033);
    //    /qcp->xAxis->setRange(QCPRange(0.0d,100.0d));
    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);


    connect(histogram->keyAxis(),SIGNAL(rangeChanged(QCPRange)),this,SLOT(histogramScrolledX(QCPRange)));
    connect(histogram->valueAxis(),SIGNAL(rangeChanged(QCPRange)),this,SLOT(histogramScrolledY(QCPRange)));

}
void histogram_widget::update_histo_data()
{
    QMutexLocker ml(&maxMux);
    dataMax = 0;
    for(int i = 0; i < histo_data.size();i++)
    {
        histo_data[i] = (double)fw->getHistogramData()[i];
        if(dataMax < histo_data[i])
        {
            dataMax = histo_data[i];
        }
    }
    ml.unlock();
}

void histogram_widget::handleNewFrame()
{
    if(qcp == NULL)
    {
        initQCPStuff();
    }
    if(fps%10==0 && !this->isHidden())
    {
        update_histo_data();
        //boost::shared_array <uint32_t> hist_shad = fw->getHistogramData();

        //qDebug() << "sum " << sum << "targt " << frWidth*frHeight;
        //qDebug() << histo_data;
        //qDebug() << histo_bins;
        // histogram->
        histogram->setData(histo_bins,histo_data);
        //histogram->keyAxis()->setRangeUpper(100.0d);
        //histogram->keyAxis()->setRangeLower(0.0d);
        //histogram->valueAxis()->setRangeLower(0.0d);
        //histogram->valueAxis()->setRangeUpper(frWidth*frHeight);
        //histogram->valueAxis()->rescale(true);


        qcp->replot();
    }
    fps++;
}

void histogram_widget::histogramScrolledY(const QCPRange &newRange)
{
    QCPRange boundedRange = newRange;
    double lowerRangeBound = 0;
    QMutexLocker ml(&maxMux);

    double upperRangeBound = dataMax*1.3;
    ml.unlock();

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
}
