#include "frame_worker.h"
#include <QDebug>
frameWorker::frameWorker(QObject *parent) :
    QObject(parent)
{


}
frameWorker::~frameWorker()
{
    //delete to;
}

void frameWorker::captureFrames()
{
    //to = new take_object(); //allocate inside slot so that it is owned by 2nd thread
    to.start();
    qDebug("starting capture");
    boost::unique_lock<boost::mutex> lock(to.framebuffer_mutex);
    qDebug( "created lock");

    to.newFrameAvailable.wait(lock);
    while(1)
    {
        to.newFrameAvailable.wait(lock);
        //qDebug("new frame availblable fc: " + to.getFrontFrame()->framecount + " timestamp: " + to.getFrontFrame()->cmTime + "\n");
        emit newFrameAvailable();
    }
}

frame * frameWorker::getFrame()
{
    return to.getFrontFrame().get();
}

u_char * frameWorker::getFrameImagePtr()
{
    //return NULL;
    return to.getFrontFrame().get()->image_data_ptr;
}

unsigned int frameWorker::getHeight()
{
   return to.height;
}

unsigned int frameWorker::getWidth()
{
   return to.width;
}
