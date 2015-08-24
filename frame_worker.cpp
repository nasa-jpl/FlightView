#include "frame_worker.h"
#include <QDebug>
#include <QCoreApplication>
#include <QMutexLocker>
#include <QSharedPointer>
#include <QDebug>

static const char notAllowedChars[]   = ",^@=+{}[]~!?:&*\"|#%<>$\"'();`' ";

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

    base_ceiling = max_val[camera_type()];
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
bool frameWorker::usingDSF()
{
    /*! \brief Returns whether or not the dark subtraction filter is being used. */
    return to.useDSF;
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
    unsigned long count = 0;
    QTime clock;
    clock.start();
    unsigned int last_savenum = 0;
    unsigned int save_num;
    frame_c *workingFrame;

    while(doRun) {
        QCoreApplication::processEvents();
        usleep(50); //So that CPU utilization is not 100%
        workingFrame = &to.frame_ring_buffer[count % CPU_FRAME_BUFFER_SIZE];
        if(std_dev_processing_frame != NULL) {
            if(std_dev_processing_frame->has_valid_std_dev == 2) {
                std_dev_frame = std_dev_processing_frame;
            }
        }
        if(workingFrame->async_filtering_done != 0) {
            curFrame = workingFrame;
            if (curFrame->has_valid_std_dev == 1) {
                std_dev_processing_frame = curFrame;
            }
            save_num = to.save_framenum.load(std::memory_order_relaxed);
            if (!to.saving_list.empty()) {
                save_num = 1;
            }
            if(save_num != last_savenum)
                emit savingFrameNumChanged(to.save_framenum);
            last_savenum = save_num;
            count++;
            if (count % 50 == 0 && count != 0) {
                delta = 50.0 / clock.restart() * 1000.0;
                emit updateFPS();
            }
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
}
void frameWorker::startSavingRawData(unsigned int framenum, QString verifiedName)
{
    /*! \brief Calls to start saving frames in cuda_take at a specified location
     * \param framenum Number of frames to save
     * \param name Location of target file */
    to.startSavingRaws(verifiedName.toUtf8().constData(), framenum);
}
void frameWorker::stopSavingRawData()
{
    /*! \brief Calls to stop saving frames in cuda_take. */
    to.stopSavingRaws();
}
bool frameWorker::validateFileName(const QString &name, QString *errorMessage = 0)
{
    // No filename
    if (name.isEmpty()) {
        if(errorMessage)
            *errorMessage = tr("File name is empty.");
        return false;
    }

    // Characters
    for (const char *c = notAllowedChars; *c; c++) {
        if (name.contains(QLatin1Char(*c))) {
            if (errorMessage) {
                const QChar qc = QLatin1Char(*c);
                *errorMessage = tr("Invalid character \"%1\" in file name.").arg(qc);
            }
            return false;
        }
    }

    // Starts with slash
    if (name.at(0) != '/') {
        if (errorMessage)
            *errorMessage = tr("File name must specify a path. Please specify the directory from root at which "
                           " to save the file."); // Upper case code
        return false;
    }
    return true;
}
void frameWorker::skipFirstRow(bool checked)
{
    /*! \brief Selects whether or not to skip the last row for profiles.
     * \param checked True skips the first row, false uses it.
     * Using the first row or not means to include it in the range of the mean profile at
     * the backend. If the first row of the image contains metadata, it may throw off the
     * average of a horizontal profile if included.
     */
    isSkippingFirst = checked;
    if (isSkippingFirst && crossStartRow < 1)
        crossStartRow = 1;
    else if (!isSkippingFirst && crossStartRow == 1)
        crossStartRow = 0;
    to.updateVertRange(crossStartRow, crossHeight);
}
void frameWorker::skipLastRow(bool checked)
{
    /*! \brief Selects whether or not to skip the last row for profiles.
     * \param checked True skips the last row, false uses it.
     * Using the last row or not means to include it in the range of the mean profile at
     * the backend. If the last row of the image contains metadata, it may throw off the
     * average of a horizontal profile if included.
     */
    isSkippingLast = checked;
    if (isSkippingLast && (crossHeight == int(frHeight) || crossHeight < 0))
        crossHeight = frHeight - 1;
    else if (!isSkippingLast && crossHeight == int(frHeight - 1))
        crossHeight = frHeight;
    to.updateVertRange(crossStartRow, crossHeight);
}
void frameWorker::updateMeanRange(int linesToAverage, image_t profile)
{
    /*! \brief Communicates the range of coordinates to average in the backend.
     * \param linesToAverage Number of rows or columns to average in the backend.
     * The linesToAverage parameter is used to determine the start and end coordinates for the mean filter at the backend. The initial conditions are set to
     * average the entire image, then are adjusted based on the image type and the location of the crosshair.
     * \author JP Ryan
     */
    crossStartCol = 0;
    crossStartRow = 0;
    crossWidth = frWidth;
    crossHeight = frHeight;
    if (profile == VERTICAL_CROSS) {
        horizLinesAvgd = linesToAverage;
        if ((crosshair_x + (linesToAverage / 2)) > (int)frWidth) {
            crossStartCol = frWidth - linesToAverage;
            crossWidth = frWidth;
        } else if ((crosshair_x - (linesToAverage / 2)) < 0) {
            crossWidth = linesToAverage;
        } else {
            crossStartCol = crosshair_x - (linesToAverage/2);
            crossWidth = crosshair_x + (linesToAverage/2);
        }
    } else if (profile == HORIZONTAL_CROSS) {
        vertLinesAvgd = linesToAverage;
        if (crosshair_y + (linesToAverage / 2) > (int)frHeight) {
            crossStartRow = frHeight - linesToAverage;
            crossHeight = frHeight;
        } else if (crosshair_y - (linesToAverage / 2) < 0) {
            crossHeight = linesToAverage;
        } else {
            crossStartRow = crosshair_y - (linesToAverage / 2);
            crossHeight = crosshair_y + (linesToAverage / 2);
        }  
    }
    crossStartRow = isSkippingFirst && crossStartRow == 0 ? 1 : crossStartRow;
    crossHeight = isSkippingLast && crossHeight == int(frHeight) ? frHeight - 1 : frHeight;
    to.updateVertRange(crossStartRow, crossHeight);
    to.updateHorizRange(crossStartCol, crossWidth);
}
void frameWorker::setCrosshairBackend(int pos_x, int pos_y)
{
    /*! \brief Determines the range of values to render when a new crosshair is selected.
     * \param pos_x The x position of the crosshair.
     * \param pos_y The y position of the crosshair.
     * The crosshair must be within the bounds of the image. It must also only transmit its value when
     * a new crosshair is selected, and only when it is on the screen (not equal to -1).
     * The cross range must not display when there are not multiple lines to average. The range must also account for whether
     * or not the option to skip the first or last row has been selected. Ranges will adjust automatically at the edge of the
     * frame to preserve the same number of lines to average.
     */
    bool repeat = crosshair_x == pos_x && crosshair_y == pos_y;
    crosshair_x = pos_x;
    crosshair_y = pos_y;    
    if (!(crosshair_x == -1 && crosshair_y == -1) && !repeat) {
        crosshair_x = crosshair_x < -1 ? 0 : crosshair_x;
        crosshair_x = crosshair_x >= (int)frWidth ? frWidth : crosshair_x;
        crosshair_y = crosshair_y < -1 ? 0 : crosshair_y;
        crosshair_y = crosshair_y >= (int)frHeight ? frHeight : crosshair_y;
        qDebug()<<"x="<<crosshair_x<<"y="<<crosshair_y;
    }

    crossStartCol = -1;
    crossStartRow = isSkippingFirst ? 1 : -1;
    crossWidth = frWidth;
    crossHeight = isSkippingLast ? frHeight - 1 : frHeight;

    if ((crosshair_x + (horizLinesAvgd / 2)) > (int)frWidth) {
        crossStartCol = frWidth - horizLinesAvgd;
        crossWidth = frWidth;
    } else if ((crosshair_x - (horizLinesAvgd / 2)) < 0) {
        crossWidth = horizLinesAvgd;
    } else {
        crossStartCol = crosshair_x - (horizLinesAvgd / 2);
        crossWidth = crosshair_x + (horizLinesAvgd / 2);
    }

    if(crosshair_y + (vertLinesAvgd / 2) > (int)frHeight) {
        crossStartRow = frHeight - vertLinesAvgd;
        crossHeight = isSkippingLast ? frHeight - 1 : frHeight;
    } else if(crosshair_y - (vertLinesAvgd / 2) < 0) {
        crossHeight = vertLinesAvgd;
    } else {
        crossStartRow = crosshair_y - (vertLinesAvgd / 2);
        crossHeight = crosshair_y + (vertLinesAvgd / 2);
    }
    if(crosshair_x == -1 && crosshair_y == -1)
        displayCross = false;
}
void frameWorker::update_FFT_range(FFT_t type, int tapNum)
{
    to.changeFFTtype(type);
    switch (type) {
    case PLANE_MEAN:
        crossStartRow = isSkippingFirst ? 1 : 0;
        crossHeight = isSkippingLast ? frHeight - 1 : frHeight;
        crossStartCol = 0;
        crossWidth = frWidth;
        to.updateHorizRange(crossStartCol, crossWidth);
        to.updateVertRange(crossStartRow, crossHeight);
        break;
    case VERT_CROSS:
        updateMeanRange(horizLinesAvgd, VERTICAL_CROSS);
        break;
    case TAP_PROFIL:
        tapPrfChanged(tapNum);
        break;
    }
}
void frameWorker::tapPrfChanged(int tapNum)
{
    crossStartCol = tapNum * TAP_WIDTH;
    crossWidth = (tapNum + 1) * TAP_WIDTH;
    crossStartRow = isSkippingFirst ? 1 : 0;
    crossHeight = isSkippingLast ? frHeight - 1 : frHeight;
    to.updateHorizRange(crossStartCol, crossWidth);
    to.updateVertRange(crossStartRow, crossHeight);
}
void frameWorker::updateCrossDiplay(bool checked)
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
void frameWorker::stop()
{
    /*! \brief Ends the event loop and sets up the workerThread to be deallocated later. */
#ifdef VERBOSE
    qDebug() << "stop frameWorker";
#endif
    doRun = false;
}
