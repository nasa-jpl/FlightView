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
    QVBoxLayout qvbl;
    QCustomPlot * qcp;
    QCPPlotTitle * plotTitle;
    int frWidth;
    int startRow;
    int endRow;
    int beginCol;
    int beginRow;
    unsigned int count = 0;
    QVector<double> x;
    QVector<double> y;
    volatile double ceiling;
    volatile double floor;
    QTimer rendertimer;
    frameWorker * fw;
public:
    explicit profile_widget(frameWorker * fw, image_t image_type , QWidget *parent = 0);
    double getCeiling();
    double getFloor();
    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;
    int frHeight;
    //int horizLinesAvgd = 1;
    int vertLinesAvgd = 1;
    image_t itype;
public slots:
    void handleNewFrame();
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();
    void updateStartRow(int sr);
    void updateEndRow(int er);
    void updateCrossRange(int linesToAverage);
};

#endif // MEAN_PROFILE_WIDGET_H
