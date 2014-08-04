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
    qDebug("starting capture");
    frHeight = to.getFrameHeight();
    frWidth = to.getFrameWidth();
    dataHeight = to.getDataHeight();
}
frameWorker::~frameWorker()
{
    qDebug() << "end frameWorker";
    doRun = false;
}
void frameWorker::stop()
{
    qDebug() << "stop frameWorker";
    doRun = false;
}

void frameWorker::captureFrames()
{
    unsigned int last_savenum;

    frame_c * workingFrame;
    while(doRun)
    {
        QCoreApplication::processEvents();
        //fr = to.getRawData(); //This now blocks
        usleep(50); //So that CPU utilization is not 100%
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

    }

    qDebug() << "emitting finished";
    emit finished();
}
unsigned int frameWorker::getFrameHeight()
{
    return frHeight;
}

unsigned int frameWorker::getFrameWidth()
{
    return frWidth;
}

unsigned int frameWorker::getDataHeight()
{
    return dataHeight;
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


void frameWorker::loadDSFMask(QString file_name)
{
    to.loadDSFMask(file_name.toUtf8().constData());
}


void frameWorker::startSavingRawData(unsigned int framenum, QString name)
{
    qDebug() << "Start Saving! @" << name;

    to.startSavingRaws(name.toUtf8().constData(),framenum);
}

void frameWorker::stopSavingRawData()
{
    to.stopSavingRaws();
}

bool frameWorker::dsfMaskCollected()
{
    return to.dsfMaskCollected;
}

camera_t frameWorker::camera_type()
{
    return to.cam_type;
}
void frameWorker::setStdDev_N(int newN)
{
    to.setStdDev_N(newN);
}
void frameWorker::toggleUseDSF(bool t)
{
    to.useDSF = t;
}
