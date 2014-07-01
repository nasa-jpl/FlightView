#ifndef MEAN_PROFILE_WIDGET_H
#define MEAN_PROFILE_WIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include "qcustomplot.h"
#include "frame_worker.h"
#include "image_type.h"
class mean_profile_widget : public QWidget
{
    Q_OBJECT
    QVBoxLayout qvbl;
    QCustomPlot * qcp;

    int frHeight;
    int frWidth;
    int fps;
    image_t itype;
    QVector<double> x;

    QVector<double> y;

    frameWorker * fw;
public:
    explicit mean_profile_widget(frameWorker * fw,image_t image_type ,QWidget *parent = 0);
    ~mean_profile_widget();

private:
    void initQCPStuff();
signals:
    
public slots:
    void handleNewFrame();
};

#endif // MEAN_PROFILE_WIDGET_H
