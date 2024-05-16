#ifndef WATERFALL_H
#define WATERFALL_H

// This is the RGB Waterfall

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

    QTimer rendertimer;

    QTimer FPSTimer;
    QElapsedTimer FPSElapsedTimer;
    unsigned int framesDelivered = 0;

    void allocateBlankWF();
    void copyPixToLine(float* image, float* dst, int pixPosition);
    void copyPixToLine(uint16_t* image, float* dst, int pixPosition);

    int maxWFlength;
    void statusMessage(QString);

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

    std::deque<std::shared_ptr<rgbLine>> wf;
    QMutex wfInUse;
    //std::atomic_bool wfInUse;

    unsigned char scaleDataPoint(float dataPt); // to ceiling and floor

    void addNewFrame();
    std::mutex addingFrame;
    void processLineToRGB(rgbLine* line); // float data to scaled RGB values
    void processLineToRGB_MP(rgbLine* line); // multi-processor version

    void rescaleWF();
    std::mutex scalingValues;

    int vSize;
    int hSize;
    int vEdge;
    int hEdge;
    unsigned char opacity;
    QImage specImage;
    void redraw();
    bool useDSF;
    bool recordToJPG = false;
    int jpgQuality = 75;
    unsigned int frameCount = 0;
    void prepareWfImage();
    void saveImage();
    bool saveImageReady = false;
    bool isSecondary = false;

public:
    explicit waterfall(frameWorker *fw, int vSize, int hSize, startupOptionsType options, QWidget *parent = nullptr);
    explicit waterfall(QWidget *parent = nullptr);
    void setup(frameWorker *fw, int vSize, int hSize, bool isSecondary, startupOptionsType options);
    void process();
    QImage* getImage();
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
    void setSecondaryWF(bool isSecondary);
    void debugThis();

private slots:
    void computeFPS();


signals:
    void statusMessageOut(QString);
    void wfReady();

};

#endif // WATERFALL_H
