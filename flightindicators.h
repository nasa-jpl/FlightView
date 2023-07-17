#ifndef FLIGHTINDICATORS_H
#define FLIGHTINDICATORS_H

#include <QWidget>
#include <QDebug>

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


private:
};

#endif // FLIGHTINDICATORS_H
