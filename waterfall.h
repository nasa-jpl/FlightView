#ifndef WATERFALL_H
#define WATERFALL_H
// This is the RGB Waterfall

// Define this for FPS data logged every second on debug builds
#define WF_DEBUG_FPS

// Define this for dynamic FPS
#define dynamicFPS

#include <deque>
#include <memory>
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <bits/shared_ptr.h>

#include <QObject>
#include <QWidget>
#include <QDebug>
#include <QMutex>
#include <QMutexLocker>
#include <QPainter>
#include <QTimer>
#include <QElapsedTimer>
#include <QDateTime>

#include "settings.h"
#include "startupOptions.h"
#include "frame_worker.h"
#include "rgbline.h"
#include "wfshared.h"
#include "imagetagger.h"


#define MAX8(x) ((x>255)?255:x)
#define MAXWF(x,y) ((x>y)?x:y)
#define MINWF(x,y) ((x<y)?x:y)
#define TOP(x,top) ((x>top)?top:x)
#define BOT(x,bot) ((x<bot)?bot:x)

class waterfall : public QWidget
{
    Q_OBJECT

    frameWorker *fw;
    int frHeight;
    int frWidth;
    startupOptionsType options;

    unsigned int TARGET_WF_FRAMERATE = 33; // FPS
    int WF_DISPLAY_PERIOD_MSECS = 1000 / TARGET_WF_FRAMERATE;

    unsigned int TARGET_WF_FRAMERATE_SECONDARY = 24; // FPS
    int WF_DISPLAY_PERIOD_MSECS_SECONDARY = 1000 / TARGET_WF_FRAMERATE_SECONDARY;

    unsigned int initialFPSSetting = TARGET_WF_FRAMERATE;
    unsigned int minimumFPS = 19; // minimum allowed dynamic FPS
    unsigned int metFPS = 0;
    unsigned int maximumFPS = 35;
    bool justMovedUpFPS = false;
    bool justMovedDownFPS = false;
    unsigned int flipFlopFPSCounter = 0;
    QTimer rendertimer;

    QTimer FPSTimer;
    QElapsedTimer FPSElapsedTimer;
    unsigned int framesDelivered = 0;
    float fps = 0;
    int fpsUnderEvents = 0;
    int fpsUEThreshold = 10; // If FPS not met for this many seconds in a row, decrease FPS by one.

    int maxWFlength = 1024;
    void statusMessage(QString);
    void debugMessage(QString);

    int wflength; // length of graphics drawn
    int ceiling;
    int floor;

    // Row numbers:
    int r_row;
    int g_row;
    int b_row;

    // Strength multipliers:
    double redLevel = 1.0;
    double greenLevel = 1.0;
    double blueLevel = 1.0;

    double gammaLevel = 1.0;

    bool useGamma = true;

    rgbLine* wflines[1024];
    int currentWFLine = 0;
    unsigned int recordingStartLineNumber = 0;
    bool justStartedRecording = false;
    bool justStoppedRecording = false;
    QMutex wfInUse;


    std::mutex addingFrame;
    void processLineToRGB(rgbLine* line); // float data to scaled RGB values
    void processLineToRGB_MP(rgbLine* line); // multi-processor version

    std::mutex scalingValues;

    int vSize;
    int hSize;
    int vEdge;
    int hEdge;
    unsigned char opacity;
    QImage *specImage = NULL;
    QImage *priorSpecImage = NULL;
    specImageBuff_t* buffer = NULL;
    gpsMessage startGPSMessage;
    gpsMessage destGPSMessage;

    void redraw();
    bool useDSF;
    bool recordToJPG = false;
    int jpgQuality = 75; // TODO: parameter
    unsigned int frameCount = 0;
    bool saveImageReady = false;
    bool isSecondary = false;
    bool followingExternalSpecImage = false;

public:
    explicit waterfall(frameWorker *fw, int vSize, int hSize, startupOptionsType options, QWidget *parent = nullptr);
    explicit waterfall(QWidget *parent = nullptr);
    ~waterfall();
    void setup(frameWorker *fw, int vSize, int hSize, bool isSecondary, startupOptionsType options);
    void process();
    QImage* getImage();
    specImageBuff_t* getImageBuffer();
    struct wfInfo_t {
        int wflength = 100;
        int ceiling = 255;
        int floor = 0;
        bool useDSF = false;
        int r_row = 100;
        int g_row = 102;
        int b_row = 104;
        double redLevel = 1.0;
        double greenLevel = 1.0;
        double blueLevel = 1.0;
        double gammaLevel = 1.0;
        bool recordToJPG = false;
        int jpgQuality = 75;
    };
    wfInfo_t getSettings();

public slots:
    void paintEvent(QPaintEvent *event);
    void changeWFLength(int length);
    void setSpecImage(QImage *specImage);
    void setSpecImageBuffer(specImageBuff_t*);
    void setSecondaryWF(bool isSecondary);
    void resetFPS(int desiredFPS);
    void debugThis();


private slots:
    void computeFPS();
    void cheapRedraw(); // follows other waterfall, no calculation


signals:
    void statusMessageOut(QString);
    void wfReady();

};

#endif // WATERFALL_H
