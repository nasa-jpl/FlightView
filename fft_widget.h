#ifndef FFT_WIDGET_H
#define FFT_WIDGET_H

#include <QSharedPointer>
#include <QWidget>
#include <QGridLayout>
#include <QCheckBox>
#include <QRadioButton>
#include <QSpinBox>
#include <QTimer>
#include "qcustomplot.h"
#include "frame_worker.h"

class fft_widget : public QWidget
{
    Q_OBJECT

    QGridLayout qgl;
    QCustomPlot* qcp;
    QCPBars* fft_bars;
    QVector<double> freq_bins;
    QVector<double> rfft_data_vec;
    QCheckBox zero_const_box;
    QSpinBox tapToProfile;
    unsigned int count = 0;
    volatile double ceiling;
    volatile double floor;
    QTimer rendertimer;
public:
    explicit fft_widget(frameWorker *fw, QWidget *parent = 0);

    frameWorker* fw;
    double getCeiling();
    double getFloor();
    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;
    QRadioButton* plMeanButton;
    QRadioButton* vCrossButton;
    QRadioButton* tapPrfButton;

public slots:
    void handleNewFrame();
    void updateCeiling(int);
    void updateFloor(int);
    void rescaleRange();
    void updateCrossRange(int);
    void updateFFT();

private slots:
    void tapPrfChanged(int);
};

#endif // FFT_WIDGET_H
