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
#include "image_type.h"
class frameview_widget : public QWidget
{
    Q_OBJECT
    QVBoxLayout * layout ;
    QLabel * imageLabel;
    QLabel * fpsLabel;
    unsigned int fps;
    QTimer * fpstimer;
    bool outputGrayScale;
    frameWorker * fw;
    image_t image_type;
    float ceiling;
    float floor;


public:
    explicit frameview_widget(frameWorker * fw,image_t image_type ,QWidget *parent = 0);
    QImage * image;
    QPushButton * toggleGrayScaleButton;
signals:
    void startCapturing();
public slots:
    void handleNewFrame();
    void updateFPS();
    void toggleGrayScale();
    void updateCeiling(int c);
    void updateFloor(int f);
};

#endif // FRAMEVIEW_WIDGET_H
