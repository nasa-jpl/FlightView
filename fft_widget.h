#ifndef FFT_WIDGET_H
#define FFT_WIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QCheckBox>
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
    QCheckBox zero_const_box;
    frameWorker * fw;
    unsigned int fps;
public:
    explicit fft_widget(frameWorker *fw, image_t image_type,QWidget *parent = 0);
    ~fft_widget();
private:
    void updateFFTData();
    void initQCPStuff();

signals:
    
public slots:
    void handleNewFrame();

};

#endif // FFT_WIDGET_H
