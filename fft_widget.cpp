#include "fft_widget.h"
#include "mean_filter.hpp"
#include "settings.h"

fft_widget::fft_widget(frameWorker *fw, QWidget *parent) :
    QWidget(parent)
{
    /*!
     * We need to set up the GUI elements which control the FFT type, and the checkbox to display the constant term.
     * \author JP Ryan
     * \author Noah Levy
     */
    qcp = NULL;
    this->fw = fw;

    zero_const_box.setText("Set constant FFT term to zero");
    zero_const_box.setChecked(true);
    plMeanButton = new QRadioButton("Plane Mean", this);
    plMeanButton->setChecked(true);
    vCrossButton = new QRadioButton("Vertical Crosshair", this);
    vCrossButton->setChecked(false);
    tapPrfButton = new QRadioButton("Tap Profile", this);
    tapPrfButton->setChecked(false);

    tapToProfile.setMinimum(0);
    tapToProfile.setMaximum(number_of_taps[fw->camera_type()]);
    tapToProfile.setSingleStep(1);
    tapToProfile.setEnabled(false);
    connect(tapPrfButton, SIGNAL(toggled(bool)), &tapToProfile, SLOT(setEnabled(bool)));
    connect(&tapToProfile, SIGNAL(valueChanged(int)), fw, SLOT(tapPrfChanged(int)));
    connect(plMeanButton, SIGNAL(clicked()), this, SLOT(updateFFT()));
    connect(vCrossButton, SIGNAL(clicked()), this, SLOT(updateFFT()));
    connect(tapPrfButton, SIGNAL(clicked()), this, SLOT(updateFFT()));

    ceiling = 100;
    floor = 0;
    qcp = new QCustomPlot(this);
    qcp->setNotAntialiasedElement(QCP::aeAll);
    qcp->xAxis->setLabel("Frequency (Hz)");
    qcp->yAxis->setLabel("Magnitude");
    fft_bars = new QCPBars(qcp->xAxis, qcp->yAxis);
    qcp->addPlottable(fft_bars);
    fft_bars->setName("Magnitude of FFT average pixel value");

    freq_bins = QVector<double>(FFT_INPUT_LENGTH / 2);
    rfft_data_vec = QVector<double>(FFT_INPUT_LENGTH / 2);

    qgl.addWidget(qcp, 0, 0, 8, 8);
    qgl.addWidget(&zero_const_box, 8, 0, 1, 2);
    qgl.addWidget(plMeanButton, 8, 2, 1, 1);
    qgl.addWidget(vCrossButton, 8, 3, 1, 1);
    qgl.addWidget(tapPrfButton, 8, 4, 1, 1);
    qgl.addWidget(&tapToProfile, 8, 5, 1, 1);
    this->setLayout(&qgl);

    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
    rendertimer.start(FRAME_DISPLAY_PERIOD_MSECS);
}

// public functions
double fft_widget::getCeiling()
{
    /*! \brief Return the value of the ceiling for this widget as a double */
    return ceiling;
}
double fft_widget::getFloor()
{
    /*! \brief Return the value of the floor for this widget as a double */
    return floor;
}

// public slots
void fft_widget::handleNewFrame()
{
    /*! \brief Render a new frame of data from the backend
     * \paragraph
     *
     * The nyquist frequency is determined based on the FFT type. The sampling rate is dependent on
     * the pixels being sampled, hence the dimensions of the frame and the frame rate are used.
     * Next, the bar graph bars are spaced according to the input vector length.
     * Then the bars are plotted based on the fftMagnitudes inside the frame struct for the current
     * frame in the buffer.
     * \author Noah Levy
     * \author JP Ryan
     */
    if (!this->isHidden() && fw->curFrame->fftMagnitude != NULL) {
        double nyquist_freq = 50.0;
        switch (fw->to.getFFTtype()) {
        case PLANE_MEAN:
            nyquist_freq = fw->delta / 2.0;
            break;
        case VERT_CROSS:
            nyquist_freq = fw->getFrameHeight() * fw->delta / 2.0;
            break;
        case TAP_PROFIL:
            nyquist_freq = TAP_WIDTH * fw->getFrameHeight() * fw->delta / 2.0;
            break;
        }
        double increment = nyquist_freq / (FFT_INPUT_LENGTH / 2);
        fft_bars->setWidth(increment);
        for(unsigned int i = 0; i < FFT_INPUT_LENGTH / 2; i++)
            freq_bins[i] = increment * i;

        float *fft_data_ptr = fw->curFrame->fftMagnitude;
        for(unsigned int b = 0; b < FFT_INPUT_LENGTH / 2; b++)
            rfft_data_vec[b] = fft_data_ptr[b];
        if(zero_const_box.isChecked())
            rfft_data_vec[0]=0;
        fft_bars->setData(freq_bins, rfft_data_vec);
        qcp->xAxis->setRange(QCPRange(0, nyquist_freq));
        qcp->replot();
    }
    count++;
}
void fft_widget::updateCeiling(int c)
{
    /*! \brief Change the value of the ceiling for this widget to the input parameter and replot the color scale. */
    ceiling = (double)c;
    rescaleRange();
}
void fft_widget::updateFloor(int f)
{
    /*! \brief Change the value of the floor for this widget to the input parameter and replot the color scale. */
    floor = (double)f;
    rescaleRange();
}
void fft_widget::rescaleRange()
{
    /*! \brief Set the color scale of the display to the last used values for this widget */
    qcp->yAxis->setRange(QCPRange(floor, ceiling));
}
void fft_widget::updateFFT()
{
    /*! \brief Update the type of FFT being input at the backend.
     * \paragraph
     *
     * We also need to change the mean filter range due to the nature
     * of how general my mean calculation is.
     * \author JP Ryan
     */
    if (plMeanButton->isChecked())
        fw->update_FFT_range(PLANE_MEAN);
    else if(vCrossButton->isChecked())
        fw->update_FFT_range(VERT_CROSS);
    else if (tapPrfButton->isChecked())
        fw->update_FFT_range(TAP_PROFIL, tapToProfile.value());
}
