#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "controlsbox.h"
#include "frameview_widget.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();
private:
    QTabWidget * tabWidget;
    QWidget * mainwidget;
    ControlsBox * controlbox;
    frameview_widget * unfiltered_widget;
    QWidget * filtered_widget;
public slots:
    void updateFPS(unsigned int fps);

};

#endif // MAINWINDOW_H
