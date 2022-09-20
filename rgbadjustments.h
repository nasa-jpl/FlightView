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
    void setRGBLevels(double r, double g, double b);

signals:
    void haveRGBLevels(double r, double g, double b);

private:
    Ui::rgbAdjustments *ui;
    double redLevel = 1.0;
    double greenLevel = 1.0;
    double blueLevel = 1.0;
    bool emitUpdateSignal = true;
};

#endif // RGBADJUSTMENTS_H
