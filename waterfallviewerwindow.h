#ifndef WATERFALVIEWERWINDOW_H
#define WATERFALVIEWERWINDOW_H

#include <QMainWindow>

#include "waterfall.h"

namespace Ui {
class waterfallViewerWindow;
}

class waterfallViewerWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit waterfallViewerWindow(QWidget *parent = nullptr);
    ~waterfallViewerWindow();

public slots:
    void setup(frameWorker *fw, int vSize, int hSize, startupOptionsType options);

    void changeRGB(int r, int g, int b);
    void setRGBLevels(double r, double g, double b, double gamma, bool reprocess);
    void setRGBLevelsAndReprocess(double r, double g, double b, double gamma);
    void changeWFLength(int length);
    void setSpecOpacity(unsigned char opacity);
    void updateCeiling(int c);
    void updateFloor(int f);
    void setUseDSF(bool useDSF);
    void setRecordWFImage(bool recordImageOn);
    void setSecondaryWF(bool isSecondary);
    void setSpecImage(bool followMe, QImage *specImage);
    void useEntireScreen();
    void debugThis();

signals:
    void statusMessageOutSig(QString);
    void wfReadySig();

private slots:
    void on_closeBtn_clicked();

    void on_showWindowBoarderBtn_clicked(bool checked);

    void on_maximizeBtn_clicked();

private:
    Ui::waterfallViewerWindow *ui;
    waterfall *wf = NULL;
    //bool independentLengthControl = true;
    QString sliderStylesheet;
};

#endif // WATERFALVIEWERWINDOW_H
