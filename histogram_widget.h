#ifndef HISTOGRAM_WIDGET_H
#define HISTOGRAM_WIDGET_H

/* Qt includes */
#include <QWidget>
#include <QVBoxLayout>
#include <QTimer>

/* Live View includes */
#include "qcustomplot.h"
#include "frame_worker.h"
#include "image_type.h"
#include "std_dev_filter.hpp"
#include "settings.h"
#include "constants.h"

/*! \file
 * \brief Plots a histogram of the spatial frequency of pixel standard deviations.
 * \paragraph
 *
 * The histogram displays the spacial frequency of the standard deviations calculated for the standard deviation frame.
 * Each bar represents a range of sigma (in DN) that the individual pixel sigmas are binned within. These data are plotted
 * with a logarithmic x-axis. Scrolling on the mouse wheel will zoom in and out. The standard viewing scale omits dead
 * pixel bars, but these can be viewed by zooming out.
 * \author Noah Levy */

class histogram_widget : public QWidget
{
    Q_OBJECT

    frameWorker *fw;
    QTimer rendertimer;

    /*! GUI elements */
    QVBoxLayout qvbl;

    /*! Plot elements */
    QCustomPlot *qcp;
    QCPBars *histogram;

    /*! Plot rendering elements */
    int frHeight, frWidth;

    volatile double ceiling;
    volatile double floor;

    QVector<double> histo_bins;
    QVector<double> histo_data_vec;
    unsigned int count = 0;

public:
    explicit histogram_widget(frameWorker *fw, QWidget *parent = 0);

    /*! \addtogroup getters
     * @{ */
    double getCeiling();
    double getFloor();
    /*! @} */

    bool slider_low_inc = false;
    const unsigned int slider_max = 100000;

public slots:
    /*! \addtogroup renderfunc
     * @{ */
    void handleNewFrame();
    /*! @} */

    /*! \addtogroup plotfunc
     * @{ */
    void histogramScrolledY(const QCPRange &newRange);
    void histogramScrolledX(const QCPRange &newRange);
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();
    void resetRange();
    /*! @} */
};

#endif // HISTOGRAM_WIDGET_H
