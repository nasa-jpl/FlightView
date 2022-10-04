#include "rgbadjustments.h"
#include "ui_rgbadjustments.h"

rgbAdjustments::rgbAdjustments(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::rgbAdjustments)
{
    ui->setupUi(this);

    this->setWindowTitle("RGB Levels");

    emitUpdateSignal = false;

    connect(ui->redSlider, &QSlider::valueChanged,
            [=](const int &newValue) {
        ui->redSpin->blockSignals(true);
        ui->redSpin->setValue(newValue);
        ui->redSpin->blockSignals(false);
        redLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel, gammaLevel);
    });

    connect(ui->greenSlider, &QSlider::valueChanged,
            [=](const int &newValue) {
        ui->greenSpin->blockSignals(true);
        ui->greenSpin->setValue(newValue);
        ui->greenSpin->blockSignals(false);
        greenLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel, gammaLevel);
    });

    connect(ui->blueSlider, &QSlider::valueChanged,
            [=](const int &newValue) {
        ui->blueSpin->blockSignals(true);
        ui->blueSpin->setValue(newValue);
        ui->blueSpin->blockSignals(false);
        blueLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel, gammaLevel);
    });

    connect(ui->redSpin, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
            [=](const int &newValue) {
        ui->redSlider->blockSignals(true);
        ui->redSlider->setValue(newValue);
        ui->redSlider->blockSignals(false);
        redLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel, gammaLevel);
    });

    connect(ui->greenSpin, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
            [=](const int &newValue) {
        ui->greenSlider->blockSignals(true);
        ui->greenSlider->setValue(newValue);
        ui->greenSlider->blockSignals(false);
        greenLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel, gammaLevel);
    });

    connect(ui->blueSpin, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
            [=](const int &newValue) {
        ui->blueSlider->blockSignals(true);
        ui->blueSlider->setValue(newValue);
        ui->blueSlider->blockSignals(false);
        blueLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel, gammaLevel);
    });

    connect(ui->gammaSlider, &QSlider::valueChanged,
            [=](const int &newValue) {
        ui->gammaSpin->setValue(newValue / 1000.0);
    });

    emitUpdateSignal = true;
    ui->gammaSlider->setEnabled(false);
    ui->gammaSpin->setEnabled(false);
}

rgbAdjustments::~rgbAdjustments()
{
    delete ui;
}

void rgbAdjustments::setRGBLevels(double r, double g, double b, double gamma)
{
    emitUpdateSignal = false;

    ui->redSpin->setValue((int)(r*100));
    ui->greenSpin->setValue((int)(g*100));
    ui->blueSpin->setValue((int)(b*100));
    ui->gammaSpin->setValue(gamma);

    emitUpdateSignal = true;
}

void rgbAdjustments::on_gammaSpin_valueChanged(double newValue)
{
    ui->gammaSlider->blockSignals(true);
    ui->gammaSlider->setValue(newValue * 1000);
    ui->gammaSlider->blockSignals(false);
    gammaLevel = newValue;
    if(emitUpdateSignal)
        emit haveRGBLevels(redLevel, greenLevel, blueLevel, gammaLevel);
}

void rgbAdjustments::on_gammaEnableChk_clicked()
{
    bool checked = ui->gammaEnableChk->isChecked();
    if(checked)
    {
        ui->gammaSpin->setEnabled(true);
        ui->gammaSlider->setEnabled(true);
        ui->gammaSpin->setValue(oldGammaLevel);
    } else {
        oldGammaLevel = gammaLevel;
        gammaLevel = 1.0;
        ui->gammaSpin->setValue(gammaLevel);
        ui->gammaSpin->setEnabled(false);
        ui->gammaSlider->setEnabled(false);
    }
}
