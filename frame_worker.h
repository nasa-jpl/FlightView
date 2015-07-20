#ifndef FRAME_WORKER_H
#define FRAME_WORKER_H

// Qt includes
#include <QElapsedTimer>
#include <QObject>
#include <QMutex>
#include <QSharedPointer>
#include <QThread>
#include <QVector>

// standard include
#include <memory>

// cuda_take includes
#include "take_object.hpp"
#include "frame_c_meta.h"

class frameWorker : public QObject
{
    Q_OBJECT

    frame_c * std_dev_processing_frame = NULL;

    unsigned int dataHeight;
    unsigned int frHeight;
    unsigned int frWidth;

    unsigned long c = 0;
    unsigned long framecount_window = 50; //we measure elapsed time for the backend fps every 50 frames

    float * histogram_bins;

    bool crosshair_useDSF = false;
    bool doRun = true;

    QElapsedTimer deltaTimer;

public:
    explicit frameWorker(QObject *parent = 0);
    virtual ~frameWorker();

    take_object to;

    int base_ceiling;

    frame_c * curFrame  = NULL;
    frame_c * std_dev_frame = NULL;

    float delta;

    bool displayCross = true;

    int horizLinesAvgd = 1;
    int crosshair_x = -1;
    int crosshair_y = -1;
    int crossStartRow = -1;
    int crossHeight = -1;
    int crossStartCol = -1;
    int crossWidth = -1;

    camera_t camera_type();
    unsigned int getFrameHeight();
    unsigned int getDataHeight();
    unsigned int getFrameWidth();
    bool dsfMaskCollected();

signals:
    void newFrameAvailable();
    void stdDevFrameCompleted(frame_c *);
    void updateFPS();

    void savingFrameNumChanged(unsigned int n);
    void finished();

public slots:
    void captureFrames();
    void startCapturingDSFMask();
    void finishCapturingDSFMask();
    void toggleUseDSF(bool t);
    void startSavingRawData(unsigned int framenum,QString name);
    void stopSavingRawData();
    void updateCrossDiplay(bool checked);
    void setStdDev_N(int newN);
    void updateDelta();
    void stop();

};


#endif // FRAME_WORKER_H
