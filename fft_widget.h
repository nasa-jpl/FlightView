#ifndef FFT_WIDGET_H
#define FFT_WIDGET_H

#include <QSharedPointer>
#include <QWidget>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QTimer>
#include "qcustomplot.h"
#include "frame_worker.h"
#include "image_type.h"

#include "camera_types.h"

class fft_widget : public QWidget
{
    Q_OBJECT
    QVBoxLayout qvbl;
    QCustomPlot * qcp;
    QCPBars *fft_bars;
    QVector<double> freq_bins;
    QVector<double> rfft_data_vec;
    QCheckBox zero_const_box;
    frameWorker * fw;
    unsigned int count = 0;
    volatile double ceiling;
    volatile double floor;
    QTimer rendertimer;

public:
    explicit fft_widget(frameWorker *fw, image_t image_type,QWidget *parent = 0);
    ~fft_widget();

    double getCeiling();
    double getFloor();
    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;
signals:
    
public slots:
    void handleNewFrame();
    void updateCeiling(int);
    void updateFloor(int);
    void rescaleRange();
};

#endif // FFT_WIDGET_H
