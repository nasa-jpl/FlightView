#include "frame_worker.h"
#include <QDebug>
#include <QCoreApplication>
#include <QMutexLocker>
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
    //rfft_data_vec = QVector<double>(FFT_INPUT_LENGTH/2);
    histo_data_vec = QVector<double>(NUMBER_OF_BINS);

    frHeight = to.getFrameHeight();
    frWidth = to.getFrameWidth();
    dataHeight = to.getDataHeight();

    unsigned long c = 0;

    while(1)
    {
        QCoreApplication::processEvents();
        //fr = to.getRawData(); //This now blocks
        usleep(500); //So that CPU utilization is not 100%
        if(to.frame_list.size() > 3 && to.frame_list.back()->async_filtering_done != 0)
        {
           // qDebug() << "on frame ?" << to.frame_list.size();
            curFrame = to.frame_list.back();
            to.frame_list.pop_back();

            //QMutexLocker ml(&vector_mutex);
            QSharedPointer<QVector<double>> fft_mags = updateFFTVector();
            //updateFFTVector();
            updateHistogramVector();
            //memcpy(raw_data,to.getRawPtr(),dataHeight*frWidth*sizeof(uint16_t));
            //memcpy(image_data,to.getImagePtr(),frHeight*frWidth*sizeof(uint16_t));

            emit newFrameAvailable(curFrame); //This onyl emits when there is a new frame
            emit newFFTMagAvailable(fft_mags);
            if(to.std_dev_ready())
            {
                emit std_dev_ready();
            }

            if(old_save_framenum != to.save_framenum)
            {
                old_save_framenum = to.save_framenum;
                emit savingFrameNumChanged(to.save_framenum);
            }

        }
    }
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




std::vector<float> *frameWorker::getHistogramBins()
{
    return NULL;
    //return histogram_bins;
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
QSharedPointer<QVector<double>> frameWorker::updateFFTVector() //This would make more sense in fft_widget, but it cannot run in the gui thread.
{
    QSharedPointer<QVector<double>> fft_magnitude_vector = QSharedPointer<QVector<double>>(new QVector<double>(FFT_INPUT_LENGTH/2));
    double max = 0;
    for(unsigned int i = 0; i < FFT_INPUT_LENGTH/2; i++)
    {
        (*fft_magnitude_vector)[i] = curFrame->fftMagnitude[i];
        if(i!=0 && curFrame->fftMagnitude[i] > max)
        {
            max = curFrame->fftMagnitude[i];
        }
    }
    //printf("%f max freq bins nonconst\n",max);
    //printf("const term in arr:%f in vec:%f\n",rfft_data[0],rfft_data_vec[0]);
    return fft_magnitude_vector;

}
void frameWorker::updateHistogramVector()
{

    histoDataMax = 0;
    for(int i = 0; i < histo_data_vec.size();i++)
    {
        histo_data_vec[i] = (double)curFrame->std_dev_histogram[i];
        if(histoDataMax < histo_data_vec[i])
        {
            histoDataMax = histo_data_vec[i];
        }
    }
    //printf("hist datamax %f\n",histoDataMax);
}
void frameWorker::toggleUseDSF(bool t)
{
    to.useDSF = t;
}
