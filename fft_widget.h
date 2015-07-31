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

/*! \file */

class fft_widget : public QWidget
{
    Q_OBJECT

    QTimer rendertimer;

    // GUI elements
    QGridLayout qgl;
    QCheckBox zero_const_box;
    QSpinBox tapToProfile;

    // Plot elements
    QCustomPlot* qcp;
    QCPBars* fft_bars;
    QVector<double> freq_bins;
    QVector<double> rfft_data_vec;

    // Plot rendering elements
    volatile double ceiling;
    volatile double floor;

    unsigned int count = 0;

public:
    explicit fft_widget(frameWorker *fw, QWidget *parent = 0);

    frameWorker* fw;

    double getCeiling();
    double getFloor();

    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

    // Public GUI elements
    QRadioButton* plMeanButton;
    QRadioButton* vCrossButton;
    QRadioButton* tapPrfButton;

public slots:
    void handleNewFrame();

    // plot controls
    void updateCeiling(int);
    void updateFloor(int);
    void rescaleRange();
    void updateCrossRange(int);
    void updateFFT();

private slots:
    void tapPrfChanged(int);
};

#endif // FFT_WIDGET_H
