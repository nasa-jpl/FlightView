#ifndef PLAYBACK_WIDGET_H
#define PLAYBACK_WIDGET_H

#include <QGridLayout>
#include <QIcon>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QWidget>

#include "dark_subtraction_filter.hpp"
#include "frame_worker.h"
#include "qcustomplot.h"

class playback_widget : public QWidget
{
    Q_OBJECT

    frameWorker* fw;
    dark_subtraction_filter* dsf;
    QTimer renderTimer; // Enables us to have time between handling frames for manipulating GUI elements

    // We need these objects to plot individual frames
    QCustomPlot* qcp;
    QCPColorMap* colorMap;
    QCPColorMapData* colorMapData;
    QCPColorScale* colorScale;

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

    QSlider* progressBar;

    // This label displays errors, shows the current progress through the file (current frame / total frames),
    // and gives intermediate status messages (e.g, "Loading file...")
    QLabel* statusLabel;

    bool play = false;
    bool playBackward = false;
    int interval = 1;

    // Some constants we will need later for processing the frames
    unsigned int pixel_size = sizeof(uint16_t);
    unsigned int frame_size;
    int frHeight, frWidth;

    bool useDSF = false;
    bool maskLoaded = false;

    int current_frame;
    int nFrames;
    uint16_t* frames;
    unsigned int* input_array;

    volatile double floor;
    volatile double ceiling;

public:
    explicit playback_widget(frameWorker* fw, QWidget *parent = 0);
    ~playback_widget();

    bool isPlaying();
    double getCeiling();
    double getFloor();

    unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;

public slots:
    void toggleUseDSF(bool t);
    void loadDSF(QString, unsigned int bytes_to_read);

    // plot controls
    void colorMapScrolledY(const QCPRange &newRange);
    void colorMapScrolledX(const QCPRange &newRange);
    void updateCeiling(int c);
    void updateFloor(int f);
    void rescaleRange();

protected:
    void keyPressEvent(QKeyEvent* c);

signals:
    void frameDone(int);

private slots:
    void loadFile();
    void updateStatus(int);
    void handleFrame(int);

    // playback controls
    void playPause();
    void moveForward();
    void moveBackward();
    void fastForward();
    void fastRewind();
    
};

#endif // PLAYBACK_WIDGET_H
