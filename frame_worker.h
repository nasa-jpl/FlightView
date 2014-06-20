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
    boost::shared_ptr< frame > getFrame();
    uint16_t * getFrameImagePtr();
    uint16_t * getRawImagePtr();

    boost::shared_array< float > getDSF();
    boost::shared_array< float > getStdDevData();

    unsigned int getHeight();
    unsigned int getWidth();

    bool isChroma();
signals:
    void newFrameAvailable();

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

private:
    take_object to;
    boost::shared_ptr< frame > fr;




};

#endif // FRAME_WORKER_H
