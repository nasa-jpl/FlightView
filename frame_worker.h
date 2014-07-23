#ifndef FRAME_WORKER_H
#define FRAME_WORKER_H

#include <QObject>
#include <QSharedPointer>
#include <QThread>
#include <QMutex>
#include <QVector>
#include "take_object.hpp"
#include "frame_c_meta.h"
#include <memory>

class frameWorker : public QObject
{
    Q_OBJECT
public:

    explicit frameWorker(QObject *parent = 0);
    virtual ~frameWorker();

    std::vector<float> *getHistogramBins();
    unsigned int getFrameHeight();
    unsigned int getDataHeight();
    unsigned int getFrameWidth();

    bool dsfMaskCollected();

    camera_t camera_type();

    double histoDataMax;

    QMutex vector_mutex;
    QVector<double> histo_data_vec;
    //QVector<double> rfft_data_vec;

    unsigned int old_save_framenum;
signals:
    void newFrameAvailable(frame_c *);
    void newFFTMagAvailable(QSharedPointer<QVector<double>>);
    void std_dev_ready();
    void savingFrameNumChanged(unsigned int n);
public slots:
    void captureFrames();
    void startCapturingDSFMask();
    void finishCapturingDSFMask();
    void loadDSFMask(QString);
    void toggleUseDSF(bool t);
    void startSavingRawData(unsigned int framenum,QString name);
    void stopSavingRawData();


    void setStdDev_N(int newN);

private:
    take_object to;
    //std::list
    QSharedPointer<QVector<double>> updateFFTVector();
    void updateHistogramVector();

    unsigned int frHeight;
    unsigned int frWidth;
    unsigned int dataHeight;


    float * histogram_bins;
    //std::shared_ptr<frame_c> curFrame;
    frame_c * curFrame;
};


#endif // FRAME_WORKER_H
