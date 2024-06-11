#ifndef FRAMEVIEW_WIDGET_H
#define FRAMEVIEW_WIDGET_H

/* Qt includes */
#include <QWidget>
#include <QThread>
#include <QImage>
#include <QGridLayout>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QPushButton>
#include <QMutex>
#include <vector>
#include <deque>


/* standard includes */
#include <atomic>

/* Live View includes */
#include "frame_c_meta.h"
#include "frame_worker.h"
#include "image_type.h"
#include "qcustomplot.h"
#include "preferences.h"
#include "startupOptions.h"


/*! \file
 * \brief Widget which plots live color maps of image data.
 * \paragraph
 *
 * The frameview_widget is the first view which is displayed on opening Live View. There are three main image types which are categorized
 * as frameviews. The "Live View" is considered a BASE image. This is the image_data_ptr array of the frame_c data structure.
 * The DSF image type is contains dark subtracted data if a mask has been collected. Otherwise it displays zero.
 * Both BASE and DSF image types accept mouse events as a method for selecting the crosshair.
 * The STD_DEV image type displays the standard deviation calculation from cuda_take.
 * \paragraph
 *
 * When constructing a copy of this widget, you must select one of the above three members of the image_t enum. */

class frameview_widget : public QWidget
{
    Q_OBJECT

    frameWorker *fw;
    QTimer rendertimer;

    float *peakValueHolder;
    bool peakHoldMode = false;

    /* QCustomPlot elements
     * Contains the necessary QCustomPlot components to create the plots for the widget. */
    QCustomPlot *qcp;
    QCPColorMap *colorMap;
    QCPColorMapData *colorMapData;
    QCPColorScale *colorScale;

    std::deque <std::vector <float>> wfimage;

    /* GUI elements
     * Contains elements of the GUI specific to a widget */
    QGridLayout layout;
    QLabel fpsLabel;
    QLabel wfSelectedRow;
    QCheckBox displayCrosshairCheck;
    QCheckBox zoomXCheck;
    QCheckBox zoomYCheck;

    QCheckBox *peakHoldChk = NULL;
    QPushButton *clearPeaksBtn = NULL;

    /* Plot Rendering elements
     * Contains local copies of the frame geometry and color map range. */
    int frHeight, frWidth;

    int wflength;

    volatile double ceiling;
    volatile double floor;


    int redRow = 0;
    int greenRow = 0;
    int blueRow = 0;

    bool drawrgbRow = false;

    bool scrollXenabled = true;
    bool scrollYenabled = true;

    /* Frame Timing elements
     * Contains the variables used to keep track of framerate and fps, as well the program time elapsed, which is used
     * to calculate the render FPS. */
    QTime clock;
    unsigned int count;
    double fps;
    QString fps_string;
    bool useDSF;
    bool havePrefs = false;
    settingsT *prefs;
    startupOptionsType options;
    void sMessage(QString statusMessageText);

public:
    explicit frameview_widget(frameWorker *fw, image_t image_type , QWidget *parent = 0);
    ~frameview_widget();

    /*! \addtogroup getters Getter Functions
     * Functions which act as getters for private information.
     * @{ */
    double getCeiling();
    double getFloor();
    /*! @} */

    void toggleDisplayCrosshair();
    void toggleDrawRGBRow(bool draw);
    void showRGB(int r, int g, int b);

    image_t image_type;
    const unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;
    void setPrefsPtr(settingsT *prefsPtr);

public slots:
    /*! \addtogroup renderfunc Rendering Functions
     * Functions which are responsible for the rendering of plots in a widget.
     * @{ */
    void handleNewColorScheme(int scheme, bool useDarkTheme);
    void handleNewFrame();
    /*! @} */

    /*! \addtogroup plotfunc Plotting Controls
     * Contains slots which adjust the display of the plots.
     *  @{ */
    void colorMapScrolledY(const QCPRange &newRange);
    void colorMapScrolledX(const QCPRange &newRange);
    void colorScaleRangeChanged(const QCPRange &newRange);
    void setScrollX(bool Yenabled);
    void setScrollY(bool Xenabled);
    void updateCeiling(int c);
    void updateFloor(int f);
    void setUseDSF(bool useDSF);
    void useDarkTheme(bool useDark);
    void rescaleRange();
    void setCrosshairs(QMouseEvent *event);
    /*! @} */

private slots:
    void clearPeaks();
    void setPeakHoldMode(bool hold);

signals:
    void statusMessage(QString message);
    void haveFloorCeilingValuesFromColorScaleChange(
            double floor, double ceiling);
};

#endif // FRAMEVIEW_WIDGET_H
