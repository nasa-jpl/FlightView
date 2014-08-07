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
//#include "view_widget_interface.h"
class profile_widget : public QWidget//, public view_widget_interface
{
    Q_OBJECT
   // Q_INTERFACES(view_widget_interface)
    QVBoxLayout qvbl;
    QCustomPlot * qcp;
    QCPPlotTitle * plotTitle;
    int frHeight;
    int frWidth;
    unsigned int count = 0;
    image_t itype;
    QVector<double> x;
    volatile double ceiling;
    volatile double floor;
    QVector<double> y;
    QTimer rendertimer;
    frameWorker * fw;
public:
    explicit profile_widget(frameWorker * fw, image_t image_type , QWidget *parent = 0);
    ~profile_widget();
    double getCeiling();
    double getFloor();
    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

signals:
    
public slots:
    void handleNewFrame();
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();

};

#endif // MEAN_PROFILE_WIDGET_H
