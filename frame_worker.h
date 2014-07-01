#ifndef FRAME_WORKER_H
#define FRAME_WORKER_H

#include <QObject>
#include <QThread>
#include "take_object.hpp"
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/shared_ptr.hpp>
class frameWorker : public QObject
{
    Q_OBJECT
public:
    explicit frameWorker(QObject *parent = 0);
    virtual ~frameWorker();

    uint16_t * getImagePtr();
    uint16_t * getRawPtr();

    float * getDSF();
    float * getStdDevData();

    float * getVerticalMean();
    float * getHorizontalMean();
    uint32_t * getHistogramData();
    std::vector<float> *getHistogramBins();
    unsigned int getHeight();
    unsigned int getWidth();
    bool dsfMaskCollected();

    bool isChroma();
signals:
    void newFrameAvailable();
    void std_dev_ready();
public slots:
    void captureFrames();
    void startCapturingDSFMask();
    void finishCapturingDSFMask();
    void loadDSFMask(const char *);

    void startSavingRawData(const char *name);
    void stopSavingRawData();

    void startSavingDSFData(const char *name);
    void stopSavingDSFData();

    void startSavingStd_DevData(const char *name);
    void stopSavingStd_DevData();

    void setStdDev_N(int newN);

private:
    take_object to;




};

#endif // FRAME_WORKER_H
