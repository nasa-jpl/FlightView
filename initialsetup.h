#ifndef INITIALSETUP_H
#define INITIALSETUP_H

#include <QDialog>
#include <QFileDialog>

#include "startupOptions.h"

namespace Ui {
class initialSetup;
}

class initialSetup : public QDialog
{
    Q_OBJECT

public:
    explicit initialSetup(QWidget *parent = nullptr);
    ~initialSetup();

    void acceptOptions(startupOptionsType *opts);
    startupOptionsType *getOptions();

private slots:
    void on_selectButton_clicked();

    void on_heightSpin_valueChanged(int arg1);

    void on_widthSpin_valueChanged(int arg1);

    void on_xioPathText_editingFinished();

    void on_xioPathText_returnPressed();

    void on_buttonBox_rejected();

    void on_buttonBox_accepted();

    void on_fpsSpin_valueChanged(double arg1);

private:
    Ui::initialSetup *ui;
    startupOptionsType *options;
};

#endif // INITIALSETUP_H
