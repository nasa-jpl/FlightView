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
    rfft_data_vec = QVector<double>(MEAN_BUFFER_LENGTH/2);
    rfft_data = new float[MEAN_BUFFER_LENGTH/2]; //This seemingly should not need to exist
    while(1)
    {
        //QCoreApplication::processEvents();
        //fr = to.getRawData(); //This now blocks
        to.waitRawPtr(); //Wait for new data to come in
        updateFFTVector();
        emit newFrameAvailable(); //This onyl emits when there is a new frame
        //if(to.std_dev_ready())
        {
         //   emit std_dev_ready();
        }
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



uint16_t * frameWorker::getImagePtr()
{
    //return NULL;
    return to.getImagePtr();
}
uint16_t * frameWorker::getRawPtr()
{
    //return NULL;
    return to.getRawPtr();
}

unsigned int frameWorker::getFrameHeight()
{
    return to.frHeight;
}
unsigned int frameWorker::getDataHeight()
{
    return to.dataHeight;
}

unsigned int frameWorker::getFrameWidth()
{
    return to.frWidth;
}
float * frameWorker::getDSF()
{
    return to.getDarkSubtractedData();
}

float * frameWorker::getStdDevData()
{
    return to.getStdDevData();
}
uint32_t * frameWorker::getHistogramData()
{
    return to.getHistogramData();
}
std::vector<float> *frameWorker::getHistogramBins()
{
    return to.getHistogramBins();
}
float * frameWorker::getHorizontalMean()
{
    return to.getHorizontalMean();
}
float * frameWorker::getVerticalMean()
{
    return to.getVerticalMean();
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
void frameWorker::updateFFTVector() //This would make more sense in fft_widget, but it cannot run in the gui thread.
{
    rfft_data = to.getRealFFTMagnitude();
    for(unsigned int i = 0; i < MEAN_BUFFER_LENGTH/2; i++)
    {
        rfft_data_vec[i] =rfft_data[i];
    }
}
