#ifndef CONTROLSBOX_H
#define CONTROLSBOX_H

#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QCheckBox>
#include <QSlider>
#include <QLineEdit>
#include <QSpinBox>
#include <QTabWidget>
#include "frameview_widget.h"

#include "boost/shared_array.hpp"
const static int max = (1<<16)-1;
const static int min = -1*max;
class ControlsBox : public QGroupBox
{
    Q_OBJECT
public:
    explicit ControlsBox(QTabWidget * tw, QWidget *parent = 0);

    QHBoxLayout controls_layout;
    QGridLayout collections_layout;
    QGridLayout sliders_layout;
    QGridLayout save_layout;
    QGroupBox single_save_box;
    QVBoxLayout single_save_layout;
    QWidget CollectionButtonsBox;
    QWidget ThresholdingSlidersBox;
    QWidget SaveButtonsBox;


    QPushButton run_collect_button;
    QPushButton run_display_button;
    QPushButton stop_collect_button;
    QPushButton stop_display_button;
    QPushButton collect_dark_frames_button;
    QPushButton stop_dark_collection_button;
    QPushButton load_mask_from_file;
    QLabel fps_label;
    QCheckBox show_dark_subtracted_cbox;
    QSlider ceiling_slider;
    QSlider floor_slider;

    QSpinBox ceiling_edit;
    QSpinBox floor_edit;

    QPushButton save_frame_button;
    QPushButton save_dark_button;

    QPushButton save_frames_button;
    QPushButton stop_saving_frames_button;
    QSpinBox frames_save_num_edit;
    QLineEdit filename_edit;
    QPushButton set_filename_button;

private:
    QTabWidget *qtw;
    frameview_widget * cur_frameview;

signals:
    void mask_selected(const char * file_name);
    void startSaving(const char * fname);
public slots:
    void getMaskFile();
    void save_button_slot();
    void tabChangedSlot(int index);


};

#endif // CONTROLSBOX_H
