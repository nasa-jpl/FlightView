#ifndef FFT_WIDGET_H
#define FFT_WIDGET_H

/* Qt includes */
#include <QSharedPointer>
#include <QWidget>
#include <QGridLayout>
#include <QCheckBox>
#include <QRadioButton>
#include <QSpinBox>
#include <QTimer>

/* Live View includes */
#include "qcustomplot.h"
#include "frame_worker.h"

/*! \file
 * \brief Plots a bar graph of the FFT of a time series.
 * \paragraph
 *
 *The FFT Widget displays the Fourier Transform of three different types of data in a bar graph format.
 * There are three sampling types which may be selected using the buttons underneath the plot. From the
 * cuda_take documentation, here are the three different types of time series input by cuda_take:
 * \paragraph
 *
 * \list
 *      1. Frame Mean (Frame Rate sampling) - the average of all pixels in one frame is determined and
 *         used as a single value in the series. Thus, all the frame means combine to make a series with
 *         a length of 1500 frames in time.
 *      2. Vertical Crosshair Profile (Row Clock sampling) - The vertical profile contains the data for
 *         all pixels in one column. This vertical profile is used as the time series input and the FFT
 *         is calculated each frame. Therefore, the image data is being sampled at the rate that rows of
 *         data are sent along the data bus, which is approximately 48 kHz (Pixel Clock / Number of Rows)
 *      3. Tap Profile (Pixel Clock sampling) - All the pixels in a tap with dimensions 160cols x 480 rows
 *         is concatenated into a time series which is input to the FFT. This enables detection of periodic
 *         signals at the pixel level. The sampling rate is equal to the pixel clock, which usually runs at
 *         10MHz. If this value is changed at the hardware level, the value of the pixel clock must be
 *         re-entered, and the program recompiled. (jk - I think this depends on resolution)
 * \paragraph
 *
 * In vertical crosshair FFTs, the number of columns to be binned may be selected using the slider in the
 * controls box. In the tap profile, the tap that is being sampled may be selected using the numbered display
 * next to the button.
 * \author JP Ryan
 * \author Noah Levy
 */

class fft_widget : public QWidget
{
    Q_OBJECT

    QTimer rendertimer;

    /*! GUI elements */
    QGridLayout qgl;
    QCheckBox zero_const_box;
    QSpinBox tapToProfile;

    /*! Plot elements */
    QCustomPlot* qcp;
    QCPBars* fft_bars;
    QVector<double> freq_bins;
    QVector<double> rfft_data_vec;

    /*! Plot rendering elements */
    volatile double ceiling;
    volatile double floor;

    unsigned int count = 0;

public:
    explicit fft_widget(frameWorker *fw, QWidget *parent = 0);

    frameWorker* fw;

    /*! \addtogroup getters
     * @{ */
    double getCeiling();
    double getFloor();
    /*! @} */

    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

    /*! Public GUI elements - ControlsBox needs access
     * @{ */
    QRadioButton* plMeanButton;
    QRadioButton* vCrossButton;
    QRadioButton* tapPrfButton;
    /*! @} */

public slots:
    /*! \addtogroup renderfunc
     * @{ */
    void handleNewFrame();
    /*! @} */

    /*! \addtogroup plotfunc
     * @{ */
    void updateCeiling(int);
    void updateFloor(int);
    void rescaleRange();
    void updateCrossRange(int);
    void updateFFT();
    /*! @} */

private slots:
    void tapPrfChanged(int);
};

#endif // FFT_WIDGET_H
