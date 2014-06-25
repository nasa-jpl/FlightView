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
    histo_data = QVector<double>(histo_bins.size());

    qcp->xAxis->setAutoTicks(false);
    qcp->xAxis->setTickStep(1);
    qcp->xAxis->setTickVector(histo_bins);
    histogram->valueAxis()->setRangeUpper(100000000.0d);
    histogram->keyAxis()->setRangeUpper(100.0d);
//    /qcp->xAxis->setRange(QCPRange(0.0d,100.0d));
    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);
}
void histogram_widget::handleNewFrame()
{
    if(qcp == NULL)
    {
        initQCPStuff();
    }
    if(fps%10==0 && !this->isHidden())
    {
        //boost::shared_array <uint32_t> hist_shad = fw->getHistogramData();
        for(int i = 0; i < histo_data.size();i++)
        {
            histo_data[i] = (double)fw->getHistogramData()[i];
        }
        //qDebug() << histo_data;
        //qDebug() << histo_bins;
        // histogram->
        histogram->setData(histo_bins,histo_data);
        histogram->keyAxis()->setRangeUpper(100.0d);
        histogram->keyAxis()->setRange(0.0d,100.0d);
        histogram->valueAxis()->setRangeLower(0.0d);


        qcp->replot();
    }
    fps++;
}
