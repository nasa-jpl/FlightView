#ifndef MEAN_PROFILE_WIDGET_H
#define MEAN_PROFILE_WIDGET_H

/* Standard includes */
#include <atomic>

/* Qt includes */
#include <QCheckBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

/* Live View includes */
#include "qcustomplot.h"
#include "frame_worker.h"
#include "image_type.h"

/*! \file
 * \brief Widget which displays a line plot of two dimensions of image data.
 * \paragraph
 *
 * profile_widget accepts processed mean data with adjustable parameters. That is, the user may select to view the mean of all rows across a
 * specific column (vertical mean profile), or of all columns across a specific row (horizontal mean profile). Additionally, there is the option
 * to have Vertical or Horizontal Crosshair Profiles which offer flexibility in the number of rows or columns to average. For example, a vertical
 * crosshair profile centered at x = 300 would contain the image data for column 300 averaged with the data for any number of other columns up to
 * the width of the frame.
 * \author JP Ryan
 * \author Noah Levy
 */

class profile_widget : public QWidget
{
    Q_OBJECT

    frameWorker *fw;
    QTimer rendertimer;

    /* GUI elements */
    QVBoxLayout qvbl;
    QCheckBox *showCalloutCheck;
    QCPItemText *callout;
    QCPItemLine *arrow;

    /* Plot elements */
    QCustomPlot *qcp;
    QCPPlotTitle *plotTitle;

    /* Frame rendering elements */
    int frWidth, frHeight;
    int xAxisMax = 0;

    volatile double ceiling;
    volatile double floor;

    QVector<double> x;
    QVector<double> y;

    int x_coord;
    volatile int y_coord;
    bool allow_callouts = true;

public:
    explicit profile_widget(frameWorker *fw, image_t image_type , QWidget *parent = 0);
    virtual ~profile_widget();

    /*! \addtogroup getters
     * @{ */
    double getCeiling();
    double getFloor();
    /*! @} */

    const unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

    image_t itype;

public slots:
    /*! \addtogroup renderfunc
     * @{ */
    void handleNewFrame();
    /*! @} */

    /*! \addtogroup plotfunc
     * @{ */
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();
    void profileScrolledX(const QCPRange &newRange);
    void profileScrolledY(const QCPRange &newRange);
    /*! @} */

    void setCallout(QMouseEvent *e);
    void moveCallout(QMouseEvent *e);
    void hideCallout();

private:
    void updateCalloutValue();

};

#endif // MEAN_PROFILE_WIDGET_H
