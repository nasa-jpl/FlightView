#include "frame_worker.h"
#include <QDebug>
#include <QCoreApplication>
#include <QMutexLocker>
#include <QSharedPointer>
#include <QDebug>
frameWorker::frameWorker(QObject *parent) :
    QObject(parent)
{
    /*! \brief Launches cuda_take using the take_object.
     * \paragraph
     * Also gathers the frame geometry from the backend. These values are used by all other members of Live View.
     * Determines the default ceiling to use based on the camera type (14-bit or 16-bit systems)
     * \author JP Ryan
     * \author Noah Levy
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
    /*! \brief Returns the number of columns as an unsigned int */
    return frWidth;
}
bool frameWorker::dsfMaskCollected()
{
    /*! \brief Returns whether or not a dark mask is loaded into cuda_take. */
    return to.dsfMaskCollected;
}

// public slots
void frameWorker::captureFrames()
{
    /*!
     * \brief The backend communication with take object is handled for each frame in this loop.
     * \paragraph
     * This event loop determines which processing elements have been completed for a frame in cuda_take.
     * First, all other events in the thread are completed, then the process sleeps for 50 microseconds to add wait time to the loop.
     * The backend frame, workingFrame is incremented based on the framecount. The framecount indexes the cuda_take ring buffer data
     * structure which contains 1500 arriving images from the camera link at a time.
     * \paragraph
     * Standard Deviation processing and Asynchronous processing are sent as signals from cuda_take. As cuda_take is a non-Qt project,
     * the signals are handled as status ints. If the asynchronous processing takes longer than a single loop through the backend, it
     * is skipped at the frontend. This prevents access to bad data by the plots. curFrames are therefore the frames which are used by
     * the frontend. Additionally, the backend frame saving is communicated between Live View and cuda_take in this loop.
     * \author Noah Levy
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
    /*! \brief Calls to start collecting dark frames in cuda_take. */
    qDebug() << "Starting to record Dark Frames";
    to.startCapturingDSFMask();
}
void frameWorker::finishCapturingDSFMask()
{
    /*! \brief Communicates to cuda_take to stop collecting dark frames. */
    qDebug() << "Stop recording Dark Frames";
    to.finishCapturingDSFMask();
}
void frameWorker::toggleUseDSF(bool t)
{
    /*! \brief Switches the boolean variable to use the DSF mask in the front and backend.
     * \param t State variable for the "Use Dark Subtraction Filter" checkbox. */
    to.useDSF = t;
    crosshair_useDSF = t;
}
void frameWorker::startSavingRawData(unsigned int framenum, QString name)
{
    /*! \brief Calls to start saving frames in cuda_take at a specified location
     * \param framenum Number of frames to save
     * \param name Location of target file */
    qDebug() << "Start Saving Frames @ " << name;

    to.startSavingRaws(name.toUtf8().constData(), framenum);
}
void frameWorker::stopSavingRawData()
{
    /*! \brief Calls to stop saving frames in cuda_take. */
    to.stopSavingRaws();
}
void frameWorker::updateCrossDiplay( bool checked )
{
    /*! \brief Communicates whether or not to display the crosshair on the frame to all frameviews. */
    displayCross = checked;
}
void frameWorker::setStdDev_N(int newN)
{
    /*! \brief Communicates changes in the standard deviation boxcar length to cuda_take.
     *  \param newN Value from the Std. Dev. N slider */
    to.setStdDev_N(newN);
}

void frameWorker::updateDelta()
{
    /*! \brief Calculates the framerate and emits updateFPS() to display. */
    delta = (float)(c * 1000) / (float)(deltaTimer.elapsed());
    emit updateFPS();
}
void frameWorker::stop()
{
    /*! \brief Ends the event loop and sets up the workerThread to be deallocated later. */
#ifdef VERBOSE
    qDebug() << "stop frameWorker";
#endif
    doRun = false;
}
