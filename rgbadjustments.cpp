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
            emit haveRGBLevels(redLevel, greenLevel, blueLevel);
    });

    connect(ui->greenSlider, &QSlider::valueChanged,
            [=](const int &newValue) {
        ui->greenSpin->blockSignals(true);
        ui->greenSpin->setValue(newValue);
        ui->greenSpin->blockSignals(false);
        greenLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel);
    });

    connect(ui->blueSlider, &QSlider::valueChanged,
            [=](const int &newValue) {
        ui->blueSpin->blockSignals(true);
        ui->blueSpin->setValue(newValue);
        ui->blueSpin->blockSignals(false);
        blueLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel);
    });

    connect(ui->redSpin, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
            [=](const int &newValue) {
        ui->redSlider->blockSignals(true);
        ui->redSlider->setValue(newValue);
        ui->redSlider->blockSignals(false);
        redLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel);
    });

    connect(ui->greenSpin, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
            [=](const int &newValue) {
        ui->greenSlider->blockSignals(true);
        ui->greenSlider->setValue(newValue);
        ui->greenSlider->blockSignals(false);
        greenLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel);
    });

    connect(ui->blueSpin, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
            [=](const int &newValue) {
        ui->blueSlider->blockSignals(true);
        ui->blueSlider->setValue(newValue);
        ui->blueSlider->blockSignals(false);
        blueLevel = newValue / 100.0;
        if(emitUpdateSignal)
            emit haveRGBLevels(redLevel, greenLevel, blueLevel);
    });

    emitUpdateSignal = true;
}

rgbAdjustments::~rgbAdjustments()
{
    delete ui;
}

void rgbAdjustments::setRGBLevels(double r, double g, double b)
{
    emitUpdateSignal = false;

    ui->redSpin->setValue((int)(r*100));
    ui->greenSpin->setValue((int)(g*100));
    ui->blueSpin->setValue((int)(b*100));

    emitUpdateSignal = true;
}
