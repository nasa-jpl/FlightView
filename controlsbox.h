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
#include <QButtonGroup>
#include <QFileDialog>
#include "frameview_widget.h"
#include <stdint.h>

#include "boost/shared_array.hpp"
const static int BIG_MAX = (1<<16)-1;
//const static int BIG_MAX = UINT16_MAX; //Only C++11 :(

const static int BIG_MIN = -1*BIG_MAX;

const static int LIL_MAX = 2000;
const static int LIL_MIN = -2000;
const static int BIG_TICK = 400;
const static int LIL_TICK = 1;
class ControlsBox : public QGroupBox
{
    Q_OBJECT
public:
    explicit ControlsBox(frameWorker * fw, QTabWidget * tw, QWidget *parent = 0);

    frameWorker * fw;
    QHBoxLayout controls_layout;
    QGridLayout collections_layout;
    QGridLayout sliders_layout;
    QGridLayout save_layout;
    QGroupBox save_continous_box;
    QVBoxLayout save_continous_layout;
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
    QPushButton select_save_location;
    QLabel fps_label;

    QCheckBox low_increment_cbox;
    QSlider std_dev_N_slider;
    QSlider ceiling_slider;
    QSlider floor_slider;

    QSpinBox std_dev_N_edit;
    QSpinBox ceiling_edit;
    QSpinBox floor_edit;

    QPushButton save_finite_button;
    QPushButton start_saving_frames_button;
    QPushButton stop_saving_frames_button;
    QSpinBox frames_save_num_edit;
    QLineEdit filename_edit;
    QPushButton set_filename_button;

    QCheckBox useDSFCbox;
    QTimer backendDeltaTimer;
private:
    QTabWidget *qtw;
    QWidget * cur_frameview;
    //view_widget_interface * cur_frameview;

signals:
    void mask_selected(const char * file_name);
    void startSavingContinous(const char * fname);
    void startSavingFinite(unsigned int length, QString fname);

    void stopSaving();
    void startDSFMaskCollection();
    void stopDSFMaskCollection();
public slots:
    void getMaskFile();
    void save_continous_button_slot();
    void stop_continous_button_slot();
    void save_finite_button_slot();
    void updateBackendDelta();
    void tabChangedSlot(int index);
    void increment_slot(bool t);
    void showSaveDialog();
    void updateSaveFrameNum_slot(unsigned int n);
private slots:
    void start_dark_collection_slot();
    void stop_dark_collection_slot();

};

#endif // CONTROLSBOX_H
