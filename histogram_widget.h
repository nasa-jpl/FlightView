#ifndef HISTOGRAM_WIDGET_H
#define HISTOGRAM_WIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QTimer>
#include "qcustomplot.h"
#include "frame_worker.h"
#include "image_type.h"
#include "std_dev_filter.hpp"
#include "settings.h"
#include "constants.h"

class histogram_widget : public QWidget
{
    Q_OBJECT

    frameWorker * fw;
    QTimer rendertimer;

    // GUI elements
    QVBoxLayout qvbl;

    // Plot elements
    QCustomPlot* qcp;
    QCPBars* histogram;

    // Plot rendering elements
    int frHeight, frWidth;

    volatile double ceiling;
    volatile double floor;

    QVector<double> histo_bins;
    QVector<double> histo_data_vec;
    unsigned int count = 0;

public:
    explicit histogram_widget(frameWorker * fw, QWidget *parent = 0);

    double getCeiling();
    double getFloor();

    unsigned int slider_max = 300000;
    bool slider_low_inc = false;

public slots:
    void handleNewFrame();

    // Plot controls
    void histogramScrolledY(const QCPRange &newRange);
    void histogramScrolledX(const QCPRange &newRange);
    void updateCeiling(int);
    void updateFloor(int);
    void rescaleRange();
    void resetRange();
};

#endif // HISTOGRAM_WIDGET_H
