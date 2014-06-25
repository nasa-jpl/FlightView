#ifndef FRAMEVIEW_WIDGET_H
#define FRAMEVIEW_WIDGET_H

#include <QWidget>
#include <QThread>
#include <QImage>
#include <QVBoxLayout>
#include <QLabel>
#include <QTimer>
#include <QPushButton>
#include <QMutex>
#include "frame_worker.h"
#include "image_type.h"
#include "qcustomplot.h"
class frameview_widget : public QWidget
{
    Q_OBJECT
    QMutex mMutex;
    QVBoxLayout layout ;
    QCustomPlot * qcp;
    QCPColorMap * colorMap;
    QCPColorMapData * colorMapData;
    QCPColorScale * colorScale;
    QLabel fpsLabel;
    unsigned int fps;
    QTimer fpstimer;
    bool outputGrayScale;
    frameWorker * fw;
    image_t image_type;

    volatile double ceiling;
    volatile double floor;
    int frHeight;
    int frWidth;
    int count;
public:
    explicit frameview_widget(frameWorker * fw,image_t image_type ,QWidget *parent = 0);
    ~frameview_widget();
    QPushButton toggleGrayScaleButton;
    double getCeiling();
    double getFloor();
private:
    void initQCPStuff(); //Needs to be in same thread as handleNewFrame?
signals:
    void startCapturing();
public slots:
    void handleNewFrame();
    void updateFPS();
    void toggleGrayScale();
    void updateCeiling(int c);
    void updateFloor(int f);
    void colorMapScrolledY(const QCPRange &newRange);
    void colorMapScrolledX(const QCPRange &newRange);
    void colorMapDataRangeChanged(const QCPRange &newRange);
};

#endif // FRAMEVIEW_WIDGET_H
