#ifndef MEAN_PROFILE_WIDGET_H
#define MEAN_PROFILE_WIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <atomic>
#include <QCheckBox>
#include <QTimer>

#include "qcustomplot.h"
#include "frame_worker.h"
#include "image_type.h"
class profile_widget : public QWidget//, public view_widget_interface
{
    Q_OBJECT

    frameWorker * fw;
    QTimer rendertimer;

    // GUI elements
    QVBoxLayout qvbl;

    // Plot elements
    QCustomPlot * qcp;
    QCPPlotTitle * plotTitle;

    // Frame rendering elements
    int frWidth, frHeight;

    volatile double ceiling;
    volatile double floor;

    int startRow,endRow;
    QVector<double> x;
    QVector<double> y;

    unsigned int count = 0;

public:
    explicit profile_widget(frameWorker * fw, image_t image_type , QWidget *parent = 0);

    double getCeiling();
    double getFloor();

    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

    image_t itype;

public slots:
    void handleNewFrame();

    // plot controls
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();
    void updateStartRow(int sr);
    void updateEndRow(int er);
    void updateCrossRange(int linesToAverage);
};

#endif // MEAN_PROFILE_WIDGET_H
