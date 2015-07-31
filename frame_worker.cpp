#include "frame_worker.h"
#include <QDebug>
#include <QCoreApplication>
#include <QMutexLocker>
#include <QSharedPointer>
#include <QDebug>
frameWorker::frameWorker(QObject *parent) :
    QObject(parent)
{
    /*! \brief Communicates with the backend and connects public information between widgets.
     * The framWorker class contains backend and analysis information that must be shared between classes in the GUI.
     * Structurally, frameWorker is a worker object tied to a QThread started in main. The main event loop in this object
     * may therefore be considered the function handleNewFrame() which periodically grabs a frame from the backend
     * and checks for asynchronous signals from the filters about the status of the data processing. A frame may still
     * be displayed even if analysis functions associated with it time out for the loop.
     * In general, the frameWorker is the only object with a copy of the take_object and should exclusively handle
     * communication with cuda_take.
     *
     * \author Noah Levy
     * \author JP Ryan
     */

    to.start(); // begin cuda_take

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

    deltaTimer.start(); // this is the backend timer which is tied to the collection of new frames from take_object
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
    /*! \brief Returns the value of the camera_type enum for the current hardware. */
    return to.cam_type;
}
unsigned int frameWorker::getFrameHeight()
{
    /*! \brief Returns the value of the frame height dimension as an unsigned int (480 for most geometries) */
    return frHeight;
}
unsigned int frameWorker::getDataHeight()
{
    /*! \brief Reutrns the value of the number of rows of raw data as an unsigned int (including any metadata) */
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
    /*!
     * \brief The backend communication with take object is handled for each frame in this loop.
     *
     *
     */
    unsigned int last_savenum = 0;
    frame_c* workingFrame;

    while(doRun)
    {
        QCoreApplication::processEvents();
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
