#include "profile_widget.h"
#include "settings.h"

profile_widget::profile_widget(frameWorker *fw, image_t image_type, QWidget *parent) : QWidget(parent)
{
    itype = image_type;
    qcp = NULL;
    this->fw = fw;
    ceiling = (1<<16);
    floor = 0;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);

    //qcp->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);
    qcp->plotLayout()->insertRow(0);
    plotTitle = new QCPPlotTitle(qcp);
    qcp->plotLayout()->addElement(0, 0,plotTitle);
    qcp->addGraph();


    if(itype == VERTICAL_MEAN || itype== VERTICAL_CROSS)
    {
        plotTitle->setText("Vertical Mean Profile");
        x = QVector<double>(frHeight);
        for(int r=0;r<frHeight;r++)
        {
            x[r] = (double) r;
        }
        y = QVector<double>(frHeight);
        qcp->xAxis->setLabel("Y index");
        qcp->xAxis->setRange(QCPRange(0,frHeight)); //From 0 to 2^16


    }
    else if (itype == HORIZONTAL_MEAN || itype== HORIZONTAL_CROSS)
    {
        plotTitle->setText("Horizontal Mean Profile");
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

    connect(&rendertimer,SIGNAL(timeout()),this,SLOT(handleNewFrame()));
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}

profile_widget::~profile_widget()
{

}




void profile_widget::handleNewFrame()
{

    if(!this->isHidden() &&  fw->curFrame != NULL)
    {
        if(itype == VERTICAL_MEAN)
        {
            float * local_image_ptr = fw->curFrame->vertical_mean_profile;
            for(int r=0;r<frHeight;r++) //Y Axis is reversed
            {
                y[frHeight-r-1] = (double) local_image_ptr[r];
            }
        }
        else if(itype == HORIZONTAL_MEAN)
        {
            float * local_image_ptr = fw->curFrame->horizontal_mean_profile;
            for(int c=0;c<frWidth;c++)
            {
                y[c] = (double) local_image_ptr[c];
            }

        }
        else if(fw->crosshair_x != -1 && fw->crosshair_y != -1)
        {
            this->setEnabled(true);
            if(itype == VERTICAL_CROSS)
            {

                //qcp->graph(0)->set1
                plotTitle->setText(QString("Vertical Profile @ x=%1").arg(fw->crosshair_x));

                if(!fw->crosshair_useDSF)
                {
                    uint16_t * local_image_ptr = fw->curFrame->image_data_ptr;
                    for(int r=0;r<frHeight;r++) //Y Axis is reversed
                    {
                        y[r] = (double) local_image_ptr[(frHeight - r - 1)*frWidth + fw->crosshair_x];
                    }
                }
                else
                {
                    float * local_image_ptr = fw->curFrame->dark_subtracted_data;
                    for(int r=0;r<frHeight;r++) //Y Axis is reversed
                    {
                        y[r] = (double) local_image_ptr[(frHeight - r - 1)*frWidth + fw->crosshair_x];
                    }
                }
            }
            else if(itype == HORIZONTAL_CROSS)
            {
                plotTitle->setText(QString("Horizontal Profile @ y=%1").arg(fw->crosshair_y));


                if(!fw->crosshair_useDSF)
                {
                    uint16_t * local_image_ptr = fw->curFrame->image_data_ptr;
                    for(int c=0;c<frWidth;c++)
                    {
                        y[c] = (double) local_image_ptr[(frHeight - fw->crosshair_y - 1)*frWidth + c];
                    }
                }
                else
                {
                    float * local_image_ptr = fw->curFrame->dark_subtracted_data;
                    for(int c=0;c<frWidth;c++)
                    {
                        y[c] = (double) local_image_ptr[(frHeight - fw->crosshair_y - 1)*frWidth + c];
                    }
                }

            }
        }
        else
        {
            this->setEnabled(false);
        }
        //qcp->graph(0)->rescaleValueAxis();

        qcp->graph(0)->setData(x,y);
        //qcp->graph(0)->rescaleAxes();
        qcp->replot();
    }

    count++;
}

void profile_widget::updateCeiling(int c)
{
    ceiling = (double)c;
    qcp->yAxis->setRangeUpper(ceiling);
    //colorMap->setDataRange(QCPRange((double)floor,(double)ceiling));
}
void profile_widget::updateFloor(int f)
{
    floor = (double)f;
    qcp->yAxis->setRangeLower(floor);
}
double profile_widget::getFloor()
{
    return floor;
}
double profile_widget::getCeiling()
{
    return ceiling;
}
void profile_widget::rescaleRange()
{
    qcp->yAxis->setRange(QCPRange(ceiling,floor));
}
