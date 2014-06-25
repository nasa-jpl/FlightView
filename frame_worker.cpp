#include "frame_worker.h"
#include <QDebug>
#include <QCoreApplication>
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


    while(1)
    {
        QCoreApplication::processEvents();
        fr = to.getFrontFrame(); //This now blocks
        emit newFrameAvailable(); //This onyl emits when there is a new frame
    }
}
void frameWorker::startCapturingDSFMask()
{
    qDebug() << "calling to start DSF cap";
    to.startCapturingDSFMask();
}
void frameWorker::finishCapturingDSFMask()
{
    qDebug() << "calling to stop DSF cap";
    to.finishCapturingDSFMask();
}

boost::shared_ptr < frame > frameWorker::getFrame()
{
    return fr;
}

uint16_t * frameWorker::getFrameImagePtr()
{
    //return NULL;
    return fr->image_data_ptr;
}
uint16_t * frameWorker::getRawImagePtr()
{
    //return NULL;
    return fr->raw_data;
}

unsigned int frameWorker::getHeight()
{
    return to.height;
}

unsigned int frameWorker::getWidth()
{
    return to.width;
}
boost::shared_array < float > frameWorker::getDSF()
{
    return to.getDarkSubtractedData();
}
boost::shared_array < float > frameWorker::getStdDevData()
{
    return to.getStdDevData();
}
boost::shared_array < uint32_t > frameWorker::getHistogramData()
{
    return to.getHistogramData();
}
std::vector<float> *frameWorker::getHistogramBins()
{
    return to.getHistogramBins();
}

void frameWorker::loadDSFMask(const char * file_name)
{
    to.loadDSFMask(file_name);
}


void frameWorker::startSavingRawData(const char * name)
{
    qDebug() << "inside frame worker ssr!";
    to.startSavingRaws(name);
}

void frameWorker::stopSavingRawData()
{
    to.stopSavingRaws();
}
void frameWorker::startSavingDSFData(const char * name)
{

}
void frameWorker::stopSavingDSFData()
{

}
void frameWorker::startSavingStd_DevData(const char * name)
{

}
void frameWorker::stopSavingStd_DevData()
{

}
bool frameWorker::isChroma()
{
    return to.isChroma;
}
