#ifndef RGBADJUSTMENTS_H
#define RGBADJUSTMENTS_H

#include <QDialog>

namespace Ui {
class rgbAdjustments;
}

class rgbAdjustments : public QDialog
{
    Q_OBJECT

public:
    explicit rgbAdjustments(QWidget *parent = nullptr);
    ~rgbAdjustments();

public slots:
    void setRGBLevels(double r, double g, double b, double gamma);

signals:
    void haveRGBLevels(double r, double g, double b, double gamma, bool reprocess);
    void setTargetFPS_primary(int targetFPS);
    void setTargetFPS_secondary(int targetFPS);
    void setTargetFPS_render(int targetFPS);


private slots:
    void on_gammaSpin_valueChanged(double arg1);

    void on_gammaEnableChk_clicked();

    void on_closeBtn_clicked();

    void on_reprocessBtn_clicked();

    void on_setPrimaryFPSBtn_clicked();

    void on_setSecondaryFPSBtn_clicked();

    void on_setRenderFPSBtn_clicked();

private:
    Ui::rgbAdjustments *ui;
    double redLevel = 1.0;
    double greenLevel = 1.0;
    double blueLevel = 1.0;
    double gammaLevel = 1.0;
    double oldGammaLevel = 1.0;
    bool emitUpdateSignal = true;
};

#endif // RGBADJUSTMENTS_H
