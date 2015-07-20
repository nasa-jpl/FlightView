#include "profile_widget.h"
#include "settings.h"
/* #define QDEBUG */

profile_widget::profile_widget(frameWorker *fw, image_t image_type, QWidget *parent) : QWidget(parent)
{
    itype = image_type;
    qcp = NULL;
    this->fw = fw;
    ceiling = fw->base_ceiling;
    floor = 0;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    startRow = 0;
    endRow = frHeight;
    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);

    qcp->plotLayout()->insertRow(0);
    plotTitle = new QCPPlotTitle(qcp);
    qcp->plotLayout()->addElement(0, 0,plotTitle);
    qcp->addGraph();

    if(itype == VERTICAL_MEAN || itype== VERTICAL_CROSS)
    {
        plotTitle->setText("Vertical Mean Profile");
        x = QVector<double>(frHeight);
        for(int r=startRow;r<endRow;r++)
        {
            x[r] = (double) r;
        }
        y = QVector<double>(frHeight);
        qcp->xAxis->setLabel("Y index");
        qcp->xAxis->setRange(QCPRange(startRow,endRow)); //From 0 to 2^16
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
    qcp->yAxis->setRange(QCPRange(0,(1<<16))); //From 0 to 2^16

    qcp->graph(0)->setData(x,y);

    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);

    connect(&rendertimer,SIGNAL(timeout()),this,SLOT(handleNewFrame()));
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}
void profile_widget::handleNewFrame()
{
    if(fw->crosshair_x != -1 && fw->crosshair_y != -1)
    {
        if(!this->isHidden() &&  fw->curFrame != NULL)
        {
            float* local_image_ptr;
            switch(itype)
            {
            case VERTICAL_MEAN:
                plotTitle->setText(QString("Vertical Mean Profile @ x=%1").arg(fw->crosshair_x));
                local_image_ptr = fw->curFrame->vertical_mean_profile;
                for(int r=endRow;r>startRow;r--) //Y Axis is reversed
                {
                    y[r] = (double) local_image_ptr[frHeight - r];
                }
                break;
            case HORIZONTAL_MEAN:
                plotTitle->setText(QString("Horizontal Mean Profile @ y=%1").arg(fw->crosshair_y));
                local_image_ptr = fw->curFrame->horizontal_mean_profile;
                for(int c=0;c<frWidth;c++)
                {
                    y[c] = (double) local_image_ptr[c];
                }
                break;
            case VERTICAL_CROSS:
                plotTitle->setText(QString("Vertical Profile Centered @ x=%1").arg(fw->crosshair_x));

                local_image_ptr = fw->curFrame->vertical_mean_profile;
                for(int r=endRow;r>startRow;r--)
                {
                    y[r] = (double) local_image_ptr[frHeight - r];
                }
                break;
            case HORIZONTAL_CROSS:
                plotTitle->setText(QString("Horizontal Profile Centered @ y=%1").arg(fw->crosshair_y));

                local_image_ptr = fw->curFrame->horizontal_mean_profile;
                for(int c=0;c<frWidth;c++)
                {
                    y[c] = (double) local_image_ptr[c];
                }
                break;
            default:
                break;
            }
            qcp->graph(0)->setData(x,y);
            qcp->replot();
        }
        else
        {
            if(!this->isHidden())
            {
                plotTitle->setText("No Crosshair designated");
                qcp->graph(0)->clearData();
                qcp->replot();
            }
        }
        count++;
    }
}
void profile_widget::updateCeiling(int c)
{
    ceiling = (double)c;
    qcp->yAxis->setRangeUpper(ceiling);
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
void profile_widget::updateStartRow( int sr )
{
    startRow = sr;
    if( itype == VERTICAL_MEAN || itype == VERTICAL_CROSS)
        qcp->xAxis->setRange(startRow, endRow);
    else if( itype == HORIZONTAL_MEAN )
        fw->to.update_start_row( startRow );
}
void profile_widget::updateEndRow( int er )
{
    endRow = er;
    if( itype == VERTICAL_MEAN || itype == VERTICAL_CROSS )
        qcp->xAxis->setRange(startRow,endRow);
    else if( itype == HORIZONTAL_MEAN )
        fw->to.update_end_row( endRow );
}
void profile_widget::updateCrossRange( int linesToAverage )
{
    int startCol = 0;
    int startRow = 0;
    int endCol = frWidth;
    int endRow = frHeight;
    //std::cout << "Image type @ cross update: " << itype << std::endl;
    if( itype == VERTICAL_CROSS )
    {
        fw->horizLinesAvgd = linesToAverage;
        if( (fw->crosshair_x + (linesToAverage/2)) > frWidth)
        {
            startCol = frWidth - linesToAverage;
            endCol = frWidth;
        }
        else if( (fw->crosshair_x - (linesToAverage/2)) < 0)
        {
            endCol = linesToAverage;
        }
        else
        {
            startCol = fw->crosshair_x - (linesToAverage/2);
            endCol = fw->crosshair_x + (linesToAverage/2);
        }
        fw->crossStartCol = startCol;
        fw->crossWidth = endCol;
    }
    else if( itype == HORIZONTAL_CROSS )
    {
        vertLinesAvgd = linesToAverage;
        if(fw->crosshair_y + (linesToAverage/2) > frHeight)
        {
            startRow = frHeight - linesToAverage;
            endRow = frHeight;
        }
        else if(fw->crosshair_y - (linesToAverage/2) < 0)
        {
            endRow = linesToAverage;
        }
        else
        {
            startRow = fw->crosshair_y - (linesToAverage/2);
            endRow = fw->crosshair_y + (linesToAverage/2);
        }
        fw->crossStartRow = startRow;
        fw->crossHeight = endRow;
    }
    //std::cout << "Vertical Range: " << startRow << " to " << endRow << std::endl;
    //std::cout << "Horizontal Range: " << startCol << " to " << endCol << std::endl;
    fw->to.updateVertRange( startRow, endRow );
    fw->to.updateHorizRange( startCol, endCol );
}
