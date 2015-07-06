#ifndef FRAME_WORKER_H
#define FRAME_WORKER_H

#include <QObject>
#include <QSharedPointer>
#include <QThread>
#include <QMutex>
#include <QElapsedTimer>
#include <QVector>
#include "take_object.hpp"
#include "frame_c_meta.h"
#include <memory>

class frameWorker : public QObject
{
    Q_OBJECT
    QElapsedTimer deltaTimer;
public:
    take_object to;

    explicit frameWorker(QObject *parent = 0);
    virtual ~frameWorker();

    std::vector<float> *getHistogramBins();
    unsigned int getFrameHeight();
    unsigned int getDataHeight();
    unsigned int getFrameWidth();

    bool dsfMaskCollected();
    bool doRun = true;
    camera_t camera_type();

    double histoDataMax;

    QMutex vector_mutex;
    //QVector<double> rfft_data_vec;

    unsigned int old_save_framenum;
    frame_c * curFrame  = NULL;
    frame_c * std_dev_frame = NULL;
    frame_c * std_dev_processing_frame = NULL;
    unsigned long c = 0;
    unsigned long framecount_window = 50; //we measure elapsed time for the backend fps every 50 frames
    float delta;
    unsigned long old_c = 0;

    unsigned int frHeight;
    unsigned int frWidth;

    int crosshair_x = -1;
    int crosshair_y = -1;
    bool crosshair_useDSF= false;
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
    void loadDSFMask(QString);
    void toggleUseDSF(bool t);
    void startSavingRawData(unsigned int framenum,QString name);
    void stopSavingRawData();
    void stop();

    void setStdDev_N(int newN);
    void updateDelta();

private:
    unsigned int dataHeight;


    float * histogram_bins;
    //std::shared_ptr<frame_c> curFrame;


};


#endif // FRAME_WORKER_H
