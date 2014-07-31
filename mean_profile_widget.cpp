#include "mean_profile_widget.h"
#include "settings.h"

mean_profile_widget::mean_profile_widget(frameWorker *fw, image_t image_type, QWidget *parent) : QWidget(parent)
{
    itype = image_type;
    qcp = NULL;
    this->fw = fw;
    ceiling = 1000;
    floor = 0;
}

mean_profile_widget::~mean_profile_widget()
{

}

void mean_profile_widget::initQCPStuff()
{
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);

    //qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);

    qcp->addGraph();


    if(itype == VERTICAL_MEAN)
    {
        qcp->graph(0)->setName("Vertical Mean Profile");
        x = QVector<double>(frHeight);
        for(int r=0;r<frHeight;r++)
        {
            x[r] = (double) r;
        }
        y = QVector<double>(frHeight);
        qcp->xAxis->setLabel("Y index");
        qcp->xAxis->setRange(QCPRange(0,frHeight)); //From 0 to 2^16


    }
    else //image_type == HORIZONTAL_MEAN
    {
        qcp->graph(0)->setName("Horizontal Mean Profile");
        x = QVector<double>(frWidth);
        for(int c=0;c<frWidth;c++)
        {
            x[c] = (double) c;
        }
        y = QVector<double>(frWidth);
        qcp->xAxis->setLabel("X index");
        qcp->xAxis->setRange(QCPRange(0,frWidth)); //From 0 to 2^16

    }
    qcp->yAxis->setLabel("Average Magnitude");
    //qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);

    qcp->graph(0)->setData(x,y);

    qcp->yAxis->setRange(QCPRange(0,(1<<16))); //From 0 to 2^16
    qvbl.addWidget(qcp);

    this->setLayout(&qvbl);
}


void mean_profile_widget::handleNewFrame(frame_c * frame)
{
    if(qcp==NULL)
    {
        initQCPStuff();
    }

    if(count%FRAME_SKIP_FACTOR == 0 && !this->isHidden())
    {
        if(itype == VERTICAL_MEAN)
        {
            float * local_image_ptr = frame->vertical_mean_profile;
            for(int r=0;r<frHeight;r++) //Y Axis is reversed
            {
                y[frHeight-r-1] = (double) local_image_ptr[r];
            }
        }
        if(itype == HORIZONTAL_MEAN)
        {
            float * local_image_ptr = frame->horizontal_mean_profile;
            for(int c=0;c<frWidth;c++)
            {
                y[c] = (double) local_image_ptr[c];
            }

        }
        //qcp->graph(0)->rescaleValueAxis();
        qcp->yAxis->setRangeLower(0);
        qcp->yAxis->setRangeLower(floor);
        qcp->yAxis->setRangeUpper(ceiling);
       qcp->graph(0)->setData(x,y);
        //qcp->graph(0)->rescaleAxes();
        qcp->replot();
    }

    count++;
}

void mean_profile_widget::updateCeiling(int c)
{
    ceiling = (double)c;
    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));
}
void mean_profile_widget::updateFloor(int f)
{
    floor = (double)f;
    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));
}
double mean_profile_widget::getFloor()
{
return floor;
}
double mean_profile_widget::getCeiling()
{
return ceiling;
}
