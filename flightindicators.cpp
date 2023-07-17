#include "flightindicators.h"
#include "ui_flightindicators.h"

flightIndicators::flightIndicators(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::flightIndicators)
{
    ui->setupUi(this);
    qDebug() << "Setup complete for Flight Indicators";
}

flightIndicators::~flightIndicators()
{
    delete ui;
}
