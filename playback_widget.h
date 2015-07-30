#ifndef PLAYBACK_WIDGET_H
#define PLAYBACK_WIDGET_H

#include <QGridLayout>
#include <QIcon>
#include <QLabel>
#include <QMutex>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QWidget>

#include "dark_subtraction_filter.hpp"
#include "frame_worker.h"
#include "qcustomplot.h"

enum err_code { SUCCESS, NO_LOAD, NO_DATA, NO_FILE, READ_FAIL, NO_MASK };

class buffer_handler : public QObject
{
    Q_OBJECT

    FILE* fp;

    int fr_height, fr_width;
    unsigned int pixel_size = sizeof(uint16_t);
    unsigned int fr_size;

    bool running;

public:
    buffer_handler(int height, int width, QObject* parent = 0);
    virtual ~buffer_handler();

    QMutex buf_access;

    int current_frame;
    int old_frame = 1;
    int num_frames;
    uint16_t* frame;
    float* dark_data;

public slots:
    void loadFile(QString file_name);
    void loadDSF(QString file_name, unsigned int elements_to_read, long offset);

    void getFrame();
    void stop();

    void debug();

signals:
    void loaded(err_code e);
    void loadMask(float*);
    void finished();

};

class playback_widget : public QWidget
{
    Q_OBJECT

    frameWorker* fw;
    dark_subtraction_filter* dark;
    QTimer render_timer; // Enables us to have time between handling frames for manipulating GUI elements
    QThread* buffer_thread;

    // GUI elements
    QGridLayout qgl;

    QIcon playIcon;
    QIcon pauseIcon;
    // These buttons all have a dual purpose and change their function simultaneously.
    // When the playback is paused, the forward and backward buttons function as frameskip keys
    // When it is playing, they function as fast forward and rewind.
    QPushButton* playPauseButton;
    QPushButton* forwardButton;
    QPushButton* backwardButton;
    QPushButton* openFileButton;
    QSpinBox* frame_value;
    QSlider* progressBar;
    // This label displays errors, shows the current progress through the file (current frame / total frames),
    // and gives intermediate status messages (e.g, "Loading file...")
    QLabel* statusLabel;
    bool play = false;
    bool playBackward = false;
    int interval = 1;

    // Plot elements
    QCustomPlot* qcp;
    QCPColorMap* colorMap;
    QCPColorMapData* colorMapData;
    QCPColorScale* colorScale;

    // Plot rendering elements
    unsigned int pixel_size = sizeof(uint16_t);
    unsigned int frame_size;
    int frHeight, frWidth;

    bool useDSF = false;

    int nFrames;

    volatile double floor;
    volatile double ceiling;

public:
    explicit playback_widget(frameWorker* fw, QWidget *parent = 0);
    ~playback_widget();

    bool isPlaying();
    double getCeiling();
    double getFloor();

    buffer_handler* bh;

    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

public slots:
    void loadDSF(QString f, unsigned int e, long o); // for some reason we need this middleman function between the controlsbox and buffer_handler
    void toggleUseDSF(bool t);
    void stop();

    // plot controls
    void colorMapScrolledY(const QCPRange &newRange);
    void colorMapScrolledX(const QCPRange &newRange);
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();

    // playback controls
    void playPause();
    void moveForward();
    void moveBackward();
    void fastForward();
    void fastRewind();

protected:
    void dragEnterEvent(QDragEnterEvent* event);
    void dropEvent(QDropEvent* event);

signals:
    void frameDone(int);

private slots:
    void loadFile();
    void finishLoading(err_code e);
    void loadMaskIn(float*);
    void updateStatus(int);
    void handleFrame(int); 
};
#endif // PLAYBACK_WIDGET_H
