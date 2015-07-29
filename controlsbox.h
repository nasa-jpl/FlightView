#ifndef CONTROLSBOX_H
#define CONTROLSBOX_H

// Qt GUI includes
#include <QButtonGroup>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QTabWidget>
#include <QVBoxLayout>

// standard includes
#include <stdint.h>

// boost includes - is this still needed?
#include "boost/shared_array.hpp"

// liveview includes
#include "frameview_widget.h"
#include "pref_window.h"

// regular slider range
const static int BIG_MAX = (1<<16) - 1;
const static int BIG_MIN = -1*BIG_MAX;
const static int BIG_TICK = 400;

// slider low increment range
const static int LIL_MAX = 2000;
const static int LIL_MIN = -2000;
const static int LIL_TICK = 1;

class ControlsBox : public QGroupBox
{
    Q_OBJECT

public:
    explicit ControlsBox(frameWorker* fw, QTabWidget* tw, QWidget* parent = 0);

    frameWorker* fw;
    preferenceWindow* prefWindow;
    QHBoxLayout controls_layout;

    //LEFT SIDE BUTTONS (Collections)
    QGridLayout collections_layout;
    QWidget CollectionButtonsBox;
    QPushButton collect_dark_frames_button;
    QPushButton stop_dark_collection_button;
    QPushButton load_mask_from_file;
    QPushButton pref_button;
    QString fps;
    QLabel fps_label;
    QLabel server_ip_label;
    QLabel server_port_label;

    //MIDDLE BUTTONS (Sliders)
    QGridLayout sliders_layout;
    QWidget ThresholdingSlidersBox;
    QSlider std_dev_N_slider;
    QSlider lines_slider;
    QSlider ceiling_slider;
    QSlider floor_slider;
    QSpinBox std_dev_N_edit;
    QSpinBox line_average_edit;
    QSpinBox ceiling_edit;
    QSpinBox floor_edit;
    QLabel std_dev_n_label;
    QLabel lines_label;
    QCheckBox low_increment_cbox;
    QCheckBox use_DSF_cbox;

    //RIGHT SIDE BUTTONS (save)
    QGridLayout save_layout;
    QWidget SaveButtonsBox;
    QPushButton save_finite_button;
    QPushButton start_saving_frames_button;
    QPushButton stop_saving_frames_button;
    QPushButton select_save_location;
    QSpinBox frames_save_num_edit;
    QLineEdit filename_edit;
    QPushButton set_filename_button;

    QElapsedTimer backendDeltaTimer;

private:
    QTabWidget *qtw;
    QWidget * cur_frameview;
    int ceiling_maximum;

signals:
    void startSavingFinite(unsigned int length, QString fname);
    void stopSaving();

    void startDSFMaskCollection();
    void stopDSFMaskCollection();
    void mask_selected(QString file_name, unsigned int bytes_to_read, long offset);

public slots:
    void tab_changed_slot(int index);

private slots:
    void increment_slot(bool t);
    void update_backend_delta();

    void show_save_dialog();
    void save_finite_button_slot();
    void stop_continous_button_slot();
    void updateSaveFrameNum_slot(unsigned int n);

    void getMaskFile();
    void start_dark_collection_slot();
    void stop_dark_collection_slot();
    void use_DSF_general(bool);

    void load_pref_window();

};

#endif // CONTROLSBOX_H
