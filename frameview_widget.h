#ifndef FRAMEVIEW_WIDGET_H
#define FRAMEVIEW_WIDGET_H

#include <QWidget>
#include <QThread>
#include <QImage>
#include <QVBoxLayout>
#include <QLabel>
#include <QTimer>
#include <QPushButton>
#include "frame_worker.h"
class frameview_widget : public QWidget
{
    Q_OBJECT
    QThread * workerThread;
    frameWorker *fw;
    QVBoxLayout * layout ;
    QLabel * imageLabel;
    QLabel * fpsLabel;
    unsigned int fps;
    QTimer * fpstimer;
    QPushButton * toggleGrayScaleButton;
    bool outputGrayScale;

public:
    explicit frameview_widget(QWidget *parent = 0);
    QImage * image;
signals:
    void startCapturing();
public slots:
    void handleNewFrame();
    void updateFPS();
    void toggleGrayScale();
private:

};

#endif // FRAMEVIEW_WIDGET_H
