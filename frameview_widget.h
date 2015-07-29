#ifndef FRAMEVIEW_WIDGET_H
#define FRAMEVIEW_WIDGET_H

#include <QWidget>
#include <QThread>
#include <QImage>
#include <QGridLayout>
#include <QCheckBox>
#include <QLabel>
#include <QTimer>
#include <QPushButton>
#include <QMutex>
#include <atomic>
#include "frame_c_meta.h"

#include "frame_worker.h"
#include "image_type.h"
#include "qcustomplot.h"

class frameview_widget : public QWidget
{
    Q_OBJECT

    frameWorker* fw;
    QTimer rendertimer;

    // Plot elements
    QCustomPlot* qcp;
    QCPColorMap* colorMap;
    QCPColorMapData* colorMapData;
    QCPColorScale* colorScale;

    // GUI elements
    QGridLayout layout;
    QLabel fpsLabel;
    QCheckBox displayCrosshairCheck;

    // Plot rendering elements
    int frHeight, frWidth;

    volatile double ceiling;
    volatile double floor;

    // Frame timing elements
    QTimer fpstimer;
    unsigned int count = 0;
    unsigned int old_count = 0;
    double fps;
    unsigned long seconds_elapsed = 0;

public:
    explicit frameview_widget(frameWorker* fw, image_t image_type , QWidget* parent = 0);
    ~frameview_widget();

    double getCeiling();
    double getFloor();
    image_t image_type;
    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

protected:
    void keyPressEvent(QKeyEvent* event);

signals:
    void startCapturing();

public slots:
    void handleNewFrame();
    void updateFPS();

    // plot controls
    void colorMapScrolledY(const QCPRange &newRange);
    void colorMapScrolledX(const QCPRange &newRange);
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();
    void setCrosshairs(QMouseEvent* event);
};

#endif // FRAMEVIEW_WIDGET_H
