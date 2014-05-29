#ifndef FRAME_WORKER_H
#define FRAME_WORKER_H

#include <QObject>
#include <QThread>
#include "take_object.hpp"
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/thread/condition_variable.hpp>
class frameWorker : public QObject
{
    Q_OBJECT
public:
    explicit frameWorker(QObject *parent = 0);
    virtual ~frameWorker();
    frame * getFrame();
    u_char * getFrameImagePtr();
    unsigned int getHeight();
    unsigned int getWidth();
signals:
    void newFrameAvailable();

public slots:
    void captureFrames();
private:
    take_object to;



};

#endif // FRAME_WORKER_H
