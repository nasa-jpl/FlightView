#ifndef CONTROLSBOX_H
#define CONTROLSBOX_H

#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QCheckBox>
#include <QSlider>
#include <QLineEdit>

class ControlsBox : public QGroupBox
{
    Q_OBJECT
public:
    explicit ControlsBox(QWidget *parent = 0);
private:
    QWidget * CollectionButtonsBox;
    QWidget * ThresholdingSlidersBox;
    QWidget * SaveButtonsBox;


    QPushButton * run_collect_button;
    QPushButton * run_display_button;
    QPushButton * stop_collect_button;
    QPushButton * stop_display_button;
    QPushButton * collect_dark_frames_button;
    QPushButton * stop_dark_collection_button;
    QLabel * fps_label;
    QCheckBox *show_dark_subtracted_cbox;
    QSlider * ceiling_slider;
    QSlider * floor_slider;

    QLineEdit * ceiling_edit;
    QLineEdit * floor_edit;

    QPushButton * save_frame_button;
    QPushButton * save_dark_button;

    QPushButton * save_frames_button;
    QPushButton * stop_saving_frames_button;
    QLineEdit * frames_save_num_edit;
    QLineEdit * filename_edit;
    QPushButton * set_filename_button;




signals:
    
public slots:
    

};

#endif // CONTROLSBOX_H
