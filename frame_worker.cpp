#include "frame_worker.h"
#include <QDebug>
#include <QCoreApplication>
#include <QMutexLocker>
#include <QSharedPointer>
#include <QDebug>
frameWorker::frameWorker(QObject *parent) :
    QObject(parent)
{
    to.start();

#ifdef VERBOSE
    qDebug("starting capture");
#endif

    frHeight = to.getFrameHeight();
    frWidth = to.getFrameWidth();
    dataHeight = to.getDataHeight();

    if(camera_type() == CL_6604A)
        base_ceiling = (1<<14) - 1;
    else
        base_ceiling = (1<<16) - 1;

    deltaTimer.start();
}
frameWorker::~frameWorker()
{
#ifdef VERBOSE
    qDebug() << "end frameWorker";
#endif
    doRun = false;
}

// public functions
camera_t frameWorker::camera_type()
{
    return to.cam_type;
}
unsigned int frameWorker::getFrameHeight()
{
    return frHeight;
}
unsigned int frameWorker::getDataHeight()
{
    return dataHeight;
}
unsigned int frameWorker::getFrameWidth()
{
    return frWidth;
}
bool frameWorker::dsfMaskCollected()
{
    return to.dsfMaskCollected;
}

// public slots
void frameWorker::captureFrames()
{
    unsigned int last_savenum = 0;
    frame_c* workingFrame;

    while(doRun)
    {
        QCoreApplication::processEvents();
        usleep(100); //So that CPU utilization is not 100%
        workingFrame = &to.frame_ring_buffer[c%CPU_FRAME_BUFFER_SIZE];

        if(std_dev_processing_frame != NULL)
        {
            if(std_dev_processing_frame->has_valid_std_dev == 2)
            {
                std_dev_frame = std_dev_processing_frame;
            }
        }
        if(workingFrame->async_filtering_done != 0)
        {
            curFrame = workingFrame;
            if(curFrame->has_valid_std_dev==1)
            {
                std_dev_processing_frame = curFrame;
            }
            unsigned int save_num = to.save_framenum.load(std::memory_order_relaxed);
            if(!to.saving_list.empty())
            {
                save_num = 1;
            }
            if(save_num != last_savenum)
                emit savingFrameNumChanged(to.save_framenum);
            last_savenum = save_num;
            c++;
        }
        if( c%framecount_window == 0 && c != 0 )
        {
            updateDelta();
        }
    }
#ifdef VERBOSE
    qDebug() << "emitting finished";
#endif
    emit finished();
}
void frameWorker::startCapturingDSFMask()
{
    qDebug() << "Starting to record Dark Frames";
    to.startCapturingDSFMask();
}
void frameWorker::finishCapturingDSFMask()
{
    qDebug() << "Stop recording Dark Frames";
    to.finishCapturingDSFMask();
}
void frameWorker::toggleUseDSF(bool t)
{
    to.useDSF = t;
    crosshair_useDSF = t;
}
void frameWorker::startSavingRawData(unsigned int framenum, QString name)
{
    qDebug() << "Start Saving Frames @ " << name;

    to.startSavingRaws(name.toUtf8().constData(), framenum);
}
void frameWorker::stopSavingRawData()
{
    to.stopSavingRaws();
}
void frameWorker::updateCrossDiplay( bool checked )
{
    displayCross = checked;
}
void frameWorker::setStdDev_N(int newN)
{
    to.setStdDev_N(newN);
}

void frameWorker::updateDelta()
{
    delta = (float)(c * 1000) / (float)(deltaTimer.elapsed());
    emit updateFPS();
}
void frameWorker::stop()
{
#ifdef VERBOSE
    qDebug() << "stop frameWorker";
#endif
    doRun = false;
}
