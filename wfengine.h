#ifndef WFENGINE_H
#define WFENGINE_H
// This is the RGB Waterfall

// Define this for FPS data logged every second on debug builds
// define WF_DEBUG_FPS

// Define this for dynamic FPS
// define dynamicFPS_engine

// Define this for image GPS tagging
#define WF_GPS_TAGGING

#include <deque>
#include <memory>
#include <atomic>
#include <mutex>
#include <unistd.h>
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

#if (QT_VERSION < QT_VERSION_CHECK(5,10,0))
qsrand(QDateTime::currentMSecsSinceEpoch()%UINT_MAX); // Remove this line if it causes trouble
#else
#include <QRandomGenerator>
#endif

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

inline uint64_t getRand64() {
#if (QT_VERSION < QT_VERSION_CHECK(5,10,0))
        return (uint64_t)qrand();
#else
        return (uint64_t)QRandomGenerator::global()->generate64();
#endif
}

class wfengine : public QObject
{
    Q_OBJECT

    frameWorker *fw;
    int frHeight;
    int frWidth;
    startupOptionsType options;

    unsigned int TARGET_WF_FRAMERATE = 35; // FPS
    int WF_DISPLAY_PERIOD_MSECS = 1000 / TARGET_WF_FRAMERATE;

    unsigned int initialFPSSetting = TARGET_WF_FRAMERATE;
    unsigned int minimumFPS_dynamic = 19; // minimum allowed dynamic FPS
    unsigned int minimumFPS_setpoint = 5;
    unsigned int metFPS = 0;
    unsigned int maximumFPS = 75;
    bool justMovedUpFPS = false;
    bool justMovedDownFPS = false;
    unsigned int flipFlopFPSCounter = 0;
    QTimer *rendertimer = NULL;
    bool timeToStop = false;

    QTimer *FPSTimer = NULL;
    QElapsedTimer FPSElapsedTimer;
    unsigned int framesDelivered = 0;
    float fps = 0;
    int fpsUnderEvents = 0;
    int fpsUEThreshold = 10; // If FPS not met for this many seconds in a row, decrease FPS by one.
    long nproc = 0;
    long nprocToUse = 0;

    void allocateBlankWF();
    void copyPixToLine(float* image, float* dst, int pixPosition);
    void copyPixToLine(uint16_t* image, float* dst, int pixPosition);

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
    uint16_t collectionID = 0;

    unsigned char scaleDataPoint(float dataPt); // to ceiling and floor

    void addNewFrame();
    std::mutex addingFrame;
    void processLineToRGB(rgbLine* line); // float data to scaled RGB values
    void processLineToRGB_MP(rgbLine* line); // multi-processor version

    void rescaleWF();
    bool waitingToReprocess = false;
    std::mutex scalingValues;

    int vSize;
    int hSize;
    int vEdge;
    int hEdge;
    unsigned char opacity;
    QImage *specImage = NULL;
    QImage *priorSpecImage = NULL;
    specImageBuff_t *buffer = NULL;
    gpsMessage *liveGPSMessagePointer = NULL;
    int startingRow = 0;
    int endingRow = 0;
    int nRowsRecorded = 0;
    bool justReachedEndWF = false;
    gpsMessage startRecordingGPSMessage;
    gpsMessage stopRecordingGPSMessage;
    gpsMessage topOfFileGPSMessage;
    gpsMessage botOfFileGPSMessage;

    void redraw();
    bool useDSF;
    bool recordToJPG = false;
    int jpgQuality = 75; // TODO: parameter
    unsigned int frameCount = 0;
    void prepareWfImage();
    void saveImage(int top, int bot);
    bool saveImageReady = false;
    bool isSecondary = false;
    bool followingExternalSpecImage = false;

public:
    explicit wfengine(frameWorker *fw, int vSize, int hSize, startupOptionsType options, QWidget *parent = nullptr);
    explicit wfengine(QWidget *parent = nullptr);
    ~wfengine();
    void setParameters(frameWorker *fw, int vSize, int hSize, startupOptionsType options);
    void process();
    QImage* getImage();
    specImageBuff_t* getImageBuffer();
    uint16_t getCollectionID();
    void setGPSStart(gpsMessage m);
    void setGPSEnd(gpsMessage m);
    void setGPSPointer(gpsMessage *m);
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
    void setup();
    void stop();
    void requestImage();
    void handleNewFrame();
    void changeRGB(int r, int g, int b);
    void setRGBLevels(double r, double g, double b, double gamma, bool reprocess);
    void setRGBLevelsAndReprocess(double r, double g, double b, double gamma);

    void changeWFLength(int length);
    void setSpecOpacity(unsigned char opacity);
    void updateCeiling(int c);
    void updateFloor(int f);
    void setUseDSF(bool useDSF);
    void setRecordWFImage(bool recordImageOn);
    void immediatelySaveImage(); // save image right now, no questions asked.
    void setSecondaryWF(bool isSecondary);
    void resetFPS(int desiredFPS);
    void debugThis();


private slots:
    void computeFPS();


signals:
    void statusMessageOut(QString);
    void  hereIsTheImage(QImage*);
    void  hereIsTheImageBuffer(specImageBuff_t*);
    void wfReady();

};

#endif // WFENGINE_H
