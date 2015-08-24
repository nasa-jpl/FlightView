#ifndef PLAYBACK_WIDGET_H
#define PLAYBACK_WIDGET_H

/* Qt includes */
#include <QGridLayout>
#include <QIcon>
#include <QLabel>
#include <QMutex>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QWidget>

/* Live View / cuda_take includes */
#include "dark_subtraction_filter.hpp"
#include "frame_worker.h"
#include "qcustomplot.h"

enum err_code {SUCCESS, NO_LOAD, NO_DATA, NO_FILE, READ_FAIL, NO_MASK};

/*! \file
 * \brief Enables the playback of image data in a video player environment.
 * \paragraph
 *
 * The Playback Widget offers self-contained playback of data. It is assumed that the data being viewed has the same geometry
 * as the currently loaded camera hardware. Playback is handled using VCR-like controls. Users may drag and drop data onto the
 * viewing window to load a file, or use the folder icon on the left side of the window.
 * \paragraph
 *
 * The widget is split into two main classes: the buffer_handler and the playback_widget. The buffer class is a worker type class which runs
 * in a parallel thread to the main GUI. It controls the file read operations for the Just-In-Time (JIT) buffer which handles frame data for
 * the playback. Whenever a new frame needs to be loaded, it is read from the file several microseconds before being plotted. A mutex handles
 * serialization of the image data between the two threads. In the playback widget, all the GUI components, frame movement, and plotting are
 * handled.
 * \author All classes and their members - JP Ryan
 */

class buffer_handler : public QObject
{
    Q_OBJECT

    FILE *fp;

    int fr_height, fr_width;
    unsigned int pixel_size = sizeof(uint16_t);
    unsigned int fr_size;

    bool running;

public:
    buffer_handler(int height, int width, QObject *parent = 0);
    virtual ~buffer_handler();

    bool hasFP();

    QMutex buf_access;

    int current_frame;
    int old_frame = 1;
    int num_frames;
    uint16_t *frame; // Array of raw data
    float *dark_data; // Array of dark subtracted data

public slots:
    void loadFile(QString file_name);
    void loadDSF(QString file_name, unsigned int elements_to_read, long offset);

    void getFrame();
    uint16_t *tapPixelRemap();
    void stop();

    void debug();

signals:
    /*! \brief This signal is emitted when a new data file is loaded in. */
    void loaded(err_code e);
    /*! \brief This signal is emitted when a new dark mask has been generated and is ready to be loaded into the
     * dark subtraction filter. */
    void loadMask(float *mask);
    /*! \brief This signal calls to stop the class event loop and deallocate the thread later. */
    void finished();

};

class playback_widget : public QWidget
{
    Q_OBJECT

    frameWorker *fw;
    dark_subtraction_filter *dark;
    QTimer render_timer; // Enables us to have time between handling frames for manipulating GUI elements
    QThread *buffer_thread;

    /* GUI elements */
    QGridLayout qgl;

    QIcon playIcon;
    QIcon pauseIcon;
    /* These buttons all have a dual purpose and change their function simultaneously.
     * When the playback is paused, the forward and backward buttons function as frameskip keys
     * When it is playing, they function as fast forward and rewind. */
    QPushButton *playPauseButton;
    QPushButton *forwardButton;
    QPushButton *backwardButton;
    QPushButton *openFileButton;
    QSpinBox *frame_value;
    QSlider *progressBar;

    /* This label displays errors, shows the current progress through the file (current frame / total frames),
     * and gives intermediate status messages (e.g, "Loading file...") */
    QLabel *statusLabel;

    bool play = false;
    bool playBackward = false;
    int interval = 1;

    /* Plot elements */
    QCustomPlot *qcp;
    QCPColorMap *colorMap;
    QCPColorMapData *colorMapData;
    QCPColorScale *colorScale;

    /* Plot rendering elements */
    const unsigned int pixel_size = sizeof(uint16_t);
    unsigned int frame_size;
    int frHeight, frWidth;

    bool useDSF = false;

    int nFrames;

    volatile double floor;
    volatile double ceiling;

public:
    explicit playback_widget(frameWorker *fw, QWidget *parent = 0);
    ~playback_widget();

    /*! \addtogroup getters
     * @{ */
    bool isPlaying();
    double getCeiling();
    double getFloor();
    /*! @} */
    bool usingDSF();

    buffer_handler* bh; // public copy of playback_widget's backend

    const unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

public slots:
    void loadDSF(QString f, unsigned int e, long o); // for some reason we need this middleman function between the controlsbox and buffer_handler
    void toggleUseDSF(bool t);

    /*! \addtogroup plotfunc
     * @{ */
    void colorMapScrolledY(const QCPRange &newRange);
    void colorMapScrolledX(const QCPRange &newRange);
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();
    /*! @} */

    /*! \addtogroup playback Playback functions
     * Functions which control the frame sequencing of data requests to the buffer_handler.
     * @{ */
    void playPause();
    void stop();
    void moveForward();
    void moveBackward();
    void fastForward();
    void fastRewind();
    /*! @} */

protected:
    void dragEnterEvent(QDragEnterEvent *event);
    void dropEvent(QDropEvent *event);

signals:
    /*! \brief This signal is emitted when a frame has finished rendering. */
    void frameDone(int);

private slots:
    void loadFile();
    void finishLoading(err_code e);
    void loadMaskIn(float *mask_arr);
    void updateStatus(int frameNum);
    void handleFrame(int frameNum);

};
#endif // PLAYBACK_WIDGET_H
