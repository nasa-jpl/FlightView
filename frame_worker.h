#ifndef FRAME_WORKER_H
#define FRAME_WORKER_H

/* Qt includes */
#include <QObject>
#include <QMutex>
#include <QSharedPointer>
#include <QThread>
#include <QTime>
#include <QVector>

/* standard include */
#include <memory>
// include <atomic>

/* cuda_take includes */
#include "take_object.hpp"
#include "frame_c_meta.h"
#include "image_type.h"
#include "startupOptions.h"

/*! \file
 * \brief Communicates with the backend and connects public information between widgets.
 * \paragraph
 *
 * The framWorker class contains backend and analysis information that must be shared between classes in the GUI.
 * Structurally, frameWorker is a worker object tied to a QThread started in main. The main event loop in this object
 * may therefore be considered the function handleNewFrame() which periodically grabs a frame from the backend
 * and checks for asynchronous signals from the filters about the status of the data processing. A frame may still
 * be displayed even if analysis functions associated with it time out for the loop.
 * In general, the frameWorker is the only object with a copy of the take_object and should exclusively handle
 * communication with cuda_take.
 *
 * \author Noah Levy
 * \author Jackie Ryan
 */

/* regular slider range */
const static int BIG_MAX = (1<<16) - 1;
const static int BIG_MIN = -1 * BIG_MAX / 4;
const static int BIG_TICK = 400;

/* slider low increment range */
const static int LIL_MAX = 2000;
const static int LIL_MIN = -2000;
const static int LIL_TICK = 1;

class frameWorker : public QObject
{
    Q_OBJECT

    frame_c *std_dev_processing_frame = NULL;

    unsigned int dataHeight;
    unsigned int frHeight;
    unsigned int frWidth;

    int lastTime;

    float *histogram_bins;
    bool doRun = true;

    startupOptionsType options;

public:
    explicit frameWorker(startupOptionsType options, QObject *parent = 0);
    virtual ~frameWorker();

    take_object to;

    frame_c *curFrame  = NULL;
    frame_c *std_dev_frame = NULL;

    float delta;
    quint16 navgs = 1;

    /* Used for frameview widgets */
    bool displayCross = true;

    /* Used for profile widgets */
    int horizLinesAvgd = -1;
    int vertLinesAvgd = -1;
    int crosshair_x = -1;
    int crosshair_y = -1;
    int crossStartRow = -1;
    int crossHeight = -1;
    int crossStartCol = -1;
    int crossWidth = -1;
    bool isSkippingFirst = false;
    bool isSkippingLast = false;

    int lh_start;
    int lh_end;
    int cent_start;
    int cent_end;
    int rh_start;
    int rh_end;

    bool use_gray = false;
    int color_scheme = 0;

    /*! Determines the default ceiling for all raw data based widgets based on the camera_t */
    int base_ceiling;

    camera_t camera_type();
    unsigned int getFrameHeight();
    unsigned int getDataHeight();
    unsigned int getFrameWidth();
    bool dsfMaskCollected();
    bool usingDSF();

signals:
    /*! \brief Calls to update the value of the backend FPS label */
    void updateFPS();

    /*! \brief Calls to render a new frame at the frontend.
     * \deprecated This signal may be deprecated in future versions. It is connected to the rendering slot in frameview_widget, but not emitted. */
    void newFrameAvailable();

    /*! \brief Calls to update the value of the Frames to Save label
     * \param n New number of frames left to save */
    void savingFrameNumChanged(unsigned int n);

    /*! \brief Calls to skip the first row of profile data.
     * \param skip Whether or not to skip the row. */

    /*! \brief Calls to skip the last row of profile data.
     * \param skip Whether or not to skip the last row. */

    /*! \brief Closes the class event loop and calls to deallocate the workerThread. */
    void finished();

    void setColorScheme_signal(int scheme, bool useDarkTheme);

    void sendStatusMessage(QString statusMessage);

public slots:
    /*! \addtogroup renderfunc
     * @{ */
    void captureFrames();
    /*! @} */

    /*! \addtogroup maskfunc
     * @{ */
    void startCapturingDSFMask();
    void finishCapturingDSFMask();
    void toggleUseDSF(bool t);
    /*! @} */

    /*! \addtogroup savingfunc
     * @{ */
    void startSavingRawData(unsigned int framenum, QString verifiedName, unsigned int numavgsave);
    void stopSavingRawData();
    /*! @} */

    void skipFirstRow(bool skip);
    void skipLastRow(bool skip);
    void updateMeanRange(int linesToAverage, image_t profile);
    void updateOverlayParams(int lh_start, int lh_end, int cent_start, int cent_end, int rh_start, int rh_end);
    void update_FFT_range(FFT_t type, int tapNum = 0);
    void tapPrfChanged(int tapNum);
    void setCrosshairBackend(int pos_x, int pos_y);
    void updateCrossDiplay(bool checked);
    void setStdDev_N(int newN);
    void stop();
    void setColorScheme(int scheme, bool useDarkTheme);
    void sMessage(QString message);
};


#endif // FRAME_WORKER_H
