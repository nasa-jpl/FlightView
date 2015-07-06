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
    QVBoxLayout qvbl;
    QCustomPlot * qcp;
    QCPBars *histogram;
    int frHeight;
    int frWidth;
    QVector<double> histo_bins;
    unsigned int count = 0;
    frameWorker * fw;
    volatile double ceiling;
    volatile double floor;

    QTimer rendertimer;
    QVector<double> histo_data_vec;
public:
    explicit histogram_widget(frameWorker * fw,image_t image_type ,QWidget *parent = 0);
    ~histogram_widget();
    double getCeiling();
    double getFloor();

    unsigned int slider_max = 300000;
    bool slider_low_inc = false;


signals:

public slots:
    //void handleNewFrame(QSharedPointer<QVector<double>> histo_data_vec);
    void handleNewFrame();
    void histogramScrolledX(const QCPRange &newRange);
    void histogramScrolledY(const QCPRange &newRange);
    void updateCeiling(int);
    void updateFloor(int);
    void rescaleRange();
};

#endif // HISTOGRAM_WIDGET_H
