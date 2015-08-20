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
    startRow = 0;
    endRow = frHeight;
    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);

    qcp->plotLayout()->insertRow(0);
    plotTitle = new QCPPlotTitle(qcp);
    plotTitle->setText("No crosshair designated");
    qcp->plotLayout()->addElement(0, 0, plotTitle);
    qcp->addGraph();

    if (itype == VERTICAL_MEAN || itype == VERTICAL_CROSS) {
        x = QVector<double>(frHeight);
        for(int r = startRow; r < endRow; r++)
            x[r] = (double)r;
        y = QVector<double>(frHeight);
        qcp->xAxis->setLabel("Y index");
        qcp->xAxis->setRange(QCPRange(startRow,endRow)); //From 0 to 2^16
    } else if (itype == HORIZONTAL_MEAN || itype == HORIZONTAL_CROSS) {
        x = QVector<double>(frWidth);
        for(int c = 0; c < frWidth; c++)
            x[c] = (double)c;
        y = QVector<double>(frWidth);
        qcp->xAxis->setLabel("X index");
        qcp->xAxis->setRange(QCPRange(0, frWidth)); //From 0 to 2^16
    }

    qcp->yAxis->setLabel("Average Magnitude");
    qcp->yAxis->setRange(QCPRange(0, fw->base_ceiling)); //From 0 to 2^16

    qcp->graph(0)->setData(x, y);

    qvbl.addWidget(qcp);
    this->setLayout(&qvbl);

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

// public slots
void profile_widget::handleNewFrame()
{
    /*! \brief Plots a specific dimension profile.
     * \paragraph
     * The switch statement is a bit silly here, I only use it to differentiate the plot title and the type of profile array to use.
     * The y-axis data is reversed in these images.
     * \author JP Ryan
     */
    if (fw->crosshair_x != -1 && fw->crosshair_y != -1) {
        if (!this->isHidden() &&  fw->curFrame != NULL) {
            float *local_image_ptr;
            switch (itype) {
            case VERTICAL_MEAN:
                plotTitle->setText(QString("Vertical Mean Profile"));
                local_image_ptr = fw->curFrame->vertical_mean_profile;
                for(int r = startRow;r < endRow; r++) //Y Axis is reversed
                    y[r] = (double) local_image_ptr[frHeight - r];
                break;
            case HORIZONTAL_MEAN:
                plotTitle->setText(QString("Horizontal Mean Profile"));
                local_image_ptr = fw->curFrame->horizontal_mean_profile;
                for(int c = 0;c < frWidth; c++)
                    y[c] = (double) local_image_ptr[c];
                break;
            case VERTICAL_CROSS:
                plotTitle->setText(QString("Vertical Profile Centered @ x=%1").arg(fw->crosshair_x));

                local_image_ptr = fw->curFrame->vertical_mean_profile;
                for(int r = startRow; r < endRow; r++)
                {
                    y[r] = (double) local_image_ptr[frHeight - r];
                }
                break;
            case HORIZONTAL_CROSS:
                plotTitle->setText(QString("Horizontal Profile Centered @ y=%1").arg(fw->crosshair_y));
                local_image_ptr = fw->curFrame->horizontal_mean_profile;
                for(int c = 0; c < frWidth; c++)
                    y[c] = (double) local_image_ptr[c];
                break;
            default:
                break;
            }
            qcp->graph(0)->setData(x, y);
            qcp->replot();
        } else {
            plotTitle->setText("No Crosshair designated");
            qcp->graph(0)->clearData();
            qcp->replot();
        }
        count++;
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
