#ifndef FLIGHTINDICATORS_H
#define FLIGHTINDICATORS_H

#include <QWidget>
#include <QDebug>
#include <QDateTime>
#include <QTimer>
#include <QFont>

#include <QLabel>
#include <QPushButton>
#include <gpsGUI/qledlabel.h>


struct fiUI_t {
    // COL 1:
    QLedLabel *diskLED = NULL;
    QLabel *imageLabel = NULL;
    QLedLabel *imageLED = NULL;
    QLedLabel *gpsLinkLED = NULL;
    QLedLabel *gpsTroubleLED = NULL;
    QPushButton *clearErrorsBtn = NULL;

    // COL 2:
    QLabel *latLabel = NULL;
    QLabel *longLabel = NULL;
    QLabel *headingLabel = NULL;
    QLabel *alignmentLabel = NULL;
    QLabel *lastIssueLabel = NULL;

    // COL 3:
    QLabel *groundSpeedLabel = NULL;
    QLabel *altitudeLabel = NULL;
    QLabel *lastRecLabel = NULL;
};

namespace Ui {
class flightIndicators;
}

class flightIndicators : public QWidget
{
    Q_OBJECT

public:
    explicit flightIndicators(QWidget *parent = nullptr);
    ~flightIndicators();

    Ui::flightIndicators *ui;

public slots:
    void updateTimeDate();
    void updateLastRec(QString hhmm);
    void updateLastRec();
    void doneRecording();
    fiUI_t getElements();


private:
    QTimer *clock;
    void ssm(QString stat);
    bool nowRecording = false;
    QFont defLabelFont;
    QFont alertLabelFont;
    void debugThis();
};

#endif // FLIGHTINDICATORS_H
