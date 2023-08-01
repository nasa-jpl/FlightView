#include "flightindicators.h"
#include "ui_flightindicators.h"

flightIndicators::flightIndicators(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::flightIndicators)
{
    ui->setupUi(this);
    ssm("Setting up.");
    clock = new QTimer();
    clock->setInterval(1000);
    connect(clock, SIGNAL(timeout()), this, SLOT(updateTimeDate()));
    clock->start();
    alertLabelFont = ui->lastRecLabel->font();
    alertLabelFont.setBold(true);
    alertLabelFont.setPointSize(alertLabelFont.pointSize()*1.20);
    defLabelFont = ui->lastRecLabel->font();

    int ledSize = 18;
    ui->diskLED->setSizeCustom(ledSize);
    ui->imageLED->setSizeCustom(ledSize);
    ui->gpsLinkLED->setSizeCustom(ledSize);
    ui->gpsTroubleLED->setSizeCustom(ledSize);

    ssm("Setup complete.");
}

flightIndicators::~flightIndicators()
{
    ssm("Running deconstructor");
    delete ui;
}

void flightIndicators::ssm(QString stat)
{
    // Send Status Message
    qDebug() << "[Flight Indicators]: " << stat;
}

void flightIndicators::updateTimeDate()
{
    QDateTime now = QDateTime::currentDateTimeUtc();
    QString timeString;
    QString dateString;

    timeString.append(now.toString("hh:mm:ss"));
    ui->utcTimeLabel->setText(timeString);

    dateString = now.toString("yyyy-MM-dd");
    ui->utcDateLabel->setText(dateString);
}

void flightIndicators::updateLastRec()
{
    QDateTime now = QDateTime::currentDateTimeUtc();
    QString timeString;

    timeString.append(now.toString("hh:mm"));
    updateLastRec(timeString);
}

void flightIndicators::updateLastRec(QString hhmm)
{
    ui->lastRecLabel->setText(hhmm);
    nowRecording = true;
    ui->lastRecLabel->setFont(alertLabelFont);
}

void flightIndicators::updateLastIssue(QString message)
{
    ui->lastIssueLabel->setText(message);
}

void flightIndicators::doneRecording()
{
    nowRecording = false;
    ui->lastRecLabel->setFont(defLabelFont);
}

fiUI_t flightIndicators::getElements()
{
    fiUI_t e;
    e.diskLED = ui->diskLED;
    e.imageLED = ui->imageLED;
    e.imageLabel = ui->imageLabel;
    e.gpsLinkLED = ui->gpsLinkLED;
    e.gpsTroubleLED = ui->gpsTroubleLED;
    e.clearErrorsBtn = ui->clearErrorsBtn;

    e.latLabel = ui->latLabel;
    e.longLabel = ui->longLabel;
    e.headingLabel = ui->headingLabel;
    e.alignmentLabel = ui->alignmentLabel;
    e.lastIssueLabel = ui->lastIssueLabel;
    e.groundSpeedLabel = ui->groundSpeedLabel;
    e.altitudeLabel = ui->altitudeLabel;
    e.lastRecLabel = ui->lastRecLabel;
    return e;
}

void flightIndicators::debugThis()
{
    //ui->latLabel
}


void flightIndicators::on_clearErrorsBtn_clicked()
{
    emit clearErrors();
}

