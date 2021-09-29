#ifndef WATERFALL_H
#define WATERFALL_H

// This is the RGB Waterfall

#include <deque>
#include <memory>
#include <atomic>
#include <mutex>
#include <stdio.h>

#include <QObject>
#include <QWidget>
#include <QPainter>
#include <QTimer>

#include "settings.h"
#include "frame_worker.h"
#include "rgbline.h"

class waterfall : public QWidget
{
    Q_OBJECT

    frameWorker *fw;
    int frHeight;
    int frWidth;

    QTimer rendertimer;

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

    std::deque<std::shared_ptr<rgbLine>> wf;
    std::atomic_bool wfInUse;

    unsigned char scaleDataPoint(float dataPt); // to ceiling and floor

    void addNewFrame();
    std::mutex addingFrame;
    void processLineToRGB(rgbLine* line); // float data to scaled RGB values
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

public:
    explicit waterfall(frameWorker *fw, int vSize, int hSize, QWidget *parent = nullptr);

public slots:
    void paintEvent(QPaintEvent *event);
    void handleNewFrame();
    void changeRGB(int r, int g, int b);
    void changeWFLength(int length);
    void setSpecOpacity(unsigned char opacity);
    void updateCeiling(int c);
    void updateFloor(int f);
    void setUseDSF(bool useDSF);
    void debugThis();

signals:
    void statusMessageOut(QString);
    void wfReady();

};

#endif // WATERFALL_H
