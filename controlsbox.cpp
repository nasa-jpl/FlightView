#include "controlsbox.h"
#include "frameview_widget.h"
#include "mean_profile_widget.h"
#include "fft_widget.h"
#include "histogram_widget.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QFileDialog>
ControlsBox::ControlsBox(frameWorker *fw, QTabWidget *tw, QWidget *parent) :
    QGroupBox(parent)
{
    this->fw = fw;
    cur_frameview=NULL;
    qtw = tw;
    //Collection Buttons


    run_collect_button.setText("Run Collect");
    run_display_button.setText("Run Display");
    run_display_button.setEnabled(false);
    stop_collect_button.setText("Stop Collect");
    stop_display_button.setText("Stop Display");
    stop_display_button.setEnabled(false);

    collect_dark_frames_button.setText("Collect Dark Frames");
    stop_dark_collection_button.setText("Stop Dark Collection");
    stop_dark_collection_button.setEnabled(false);
    load_mask_from_file.setText("Load Mask From File");
    load_mask_from_file.setEnabled(false);
    fps_label.setText("Warning: no data recieved");

    select_save_location.setText("Select save location");
    frames_save_num_edit.setMaximum(100000);
    //First Row
    collections_layout.addWidget(&run_display_button,1,1,1,1);
    //collections_layout->addWidget(&run_display_button,1,2);
    collections_layout.addWidget(&stop_display_button,1,2,1,1);
    //collections_layout->addWidget(&stop_display_button,1,4);

    //Second Row
    collections_layout.addWidget(&collect_dark_frames_button,2,1,1,1);
    collections_layout.addWidget(&stop_dark_collection_button,2,2,1,1);
    collections_layout.addWidget(&load_mask_from_file,3,2,1,1);

    //Third Row
    collections_layout.addWidget(&fps_label,3,1,1,1);

    CollectionButtonsBox.setLayout(&collections_layout);

    //Slider Thresholding Buttons
    //QGridLayout * sliders_layout = new QGridLayout();
    std_dev_N_slider.setOrientation(Qt::Horizontal);
    std_dev_N_slider.setMinimum(1);
    std_dev_N_slider.setMaximum(MAX_N-1);
    std_dev_N_slider.setValue(std_dev_N_slider.maximum());


    ceiling_slider.setOrientation(Qt::Horizontal);
    ceiling_slider.setMaximum(BIG_MAX);
    ceiling_slider.setMinimum(BIG_MIN);

    floor_slider.setOrientation(Qt::Horizontal);
    floor_slider.setMaximum(BIG_MAX);
    floor_slider.setMinimum(BIG_MIN);

    std_dev_N_edit.setMinimum(1);
    std_dev_N_edit.setMaximum(MAX_N-1);
    std_dev_N_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);
    std_dev_N_edit.setValue(std_dev_N_edit.maximum());

    ceiling_edit.setMaximum(BIG_MAX);
    ceiling_edit.setMinimum(BIG_MIN);
    ceiling_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);

    floor_edit.setMaximum(BIG_MAX);
    floor_edit.setMinimum(BIG_MIN);
    floor_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);

    low_increment_cbox.setText("Slider Low Increment?");
    ceiling_slider.setTickInterval(BIG_TICK);
    floor_slider.setTickInterval(BIG_TICK);

    useDSFCbox.setText("Use Dark Subtracted data for mean profiles and FFT?");
    //First Row
    sliders_layout.addWidget(new QLabel("Std. Dev. N:"),1,1,1,1);
    sliders_layout.addWidget(&std_dev_N_slider,1,2,1,7);
    sliders_layout.addWidget(&std_dev_N_edit,1,10,1,1);
    //Second Row
    sliders_layout.addWidget(&low_increment_cbox,2,1,1,1);
    sliders_layout.addWidget(&useDSFCbox,2,2,1,1);

    //Third Row
    sliders_layout.addWidget(new QLabel("Ceiling:"),3,1,1,1);
    sliders_layout.addWidget(&ceiling_slider,3,2,1,7);
    sliders_layout.addWidget(&ceiling_edit,3,10,1,1);

    //Fourth Row
    sliders_layout.addWidget(new QLabel("Floor:"),4,1,1,1);
    sliders_layout.addWidget(&floor_slider,4,2,1,7);
    sliders_layout.addWidget(&floor_edit,4,10,1,1);
    ThresholdingSlidersBox.setLayout(&sliders_layout);

    //Save Buttons

    save_finite_button.setText("Save Frames");

    start_saving_frames_button.setText("Start Saving");
    stop_saving_frames_button.setText("Stop Saving");
    stop_saving_frames_button.setEnabled(false);

    //save_button_group.addButton(&start_saving_frames_button);
    //save_button_group.addButton(&stop_saving_frames_button);
    //stop_saving_frames_button.setEnabled(false);
    // save_button_group.setExclusive(true);
    frames_save_num_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);
    frames_save_num_edit.setMinimum(1);
    /*
    QGridLayout * save_layout = new QGridLayout();
    QGroupBox * single_save_box = new QGroupBox();
    QVBoxLayout * single_save_layout = new QVBoxLayout;
    */
    //Column 1
    save_layout.addWidget(&select_save_location,1,1,1,1);
    save_layout.addWidget(new QLabel("Frames to save:"),2,1,1,1);
    save_layout.addWidget(new QLabel("Filename:"),3,1,1,1);

    //Column 2
    //save_layout.addWidget(&start_saving_frames_button,1,2,1,1);
    save_layout.addWidget(&save_finite_button,1,2,1,1);
    save_layout.addWidget(&frames_save_num_edit,2,2,1,2);
    save_layout.addWidget(&filename_edit,3,2,1,2);
    //Column 3
    save_layout.addWidget(&stop_saving_frames_button,1,3,1,1);

    SaveButtonsBox.setLayout(&save_layout);

    controls_layout.addWidget(&CollectionButtonsBox,2);
    controls_layout.addWidget(&ThresholdingSlidersBox,3);
    controls_layout.addWidget(&SaveButtonsBox,2);
    this->setLayout(&controls_layout);
    this->setMaximumHeight(150);


    connect(&std_dev_N_edit,SIGNAL(valueChanged(int)),&std_dev_N_slider,SLOT(setValue(int)));
    connect(&std_dev_N_slider,SIGNAL(valueChanged(int)),&std_dev_N_edit,SLOT(setValue(int)));

    connect(&ceiling_edit,SIGNAL(valueChanged(int)),&ceiling_slider,SLOT(setValue(int)));
    connect(&ceiling_slider,SIGNAL(valueChanged(int)),&ceiling_edit,SLOT(setValue(int)));

    connect(&floor_edit,SIGNAL(valueChanged(int)),&floor_slider,SLOT(setValue(int)));
    connect(&floor_slider,SIGNAL(valueChanged(int)),&floor_edit,SLOT(setValue(int)));

    connect(&load_mask_from_file,SIGNAL(clicked()),this,SLOT(getMaskFile()));

    connect(&low_increment_cbox,SIGNAL(toggled(bool)),this,SLOT(increment_slot(bool)));

    connect(&save_finite_button,SIGNAL(clicked()),this,SLOT(save_finite_button_slot()));
    //connect(&start_saving_frames_button,SIGNAL(clicked()),this,SLOT(start_continous_button_slot()));
    connect(&stop_saving_frames_button,SIGNAL(clicked()),this,SLOT(stop_continous_button_slot()));


    connect(&select_save_location,SIGNAL(clicked()),this,SLOT(showSaveDialog()));

    connect(&collect_dark_frames_button,SIGNAL(clicked()),this,SLOT(start_dark_collection_slot()));
    connect(&stop_dark_collection_button,SIGNAL(clicked()),this,SLOT(stop_dark_collection_slot()));

    connect(&backendDeltaTimer,SIGNAL(timeout()),this,SLOT(updateBackendDelta()));
    backendDeltaTimer.start(1000);
}
void ControlsBox::showSaveDialog()
{
    //QFileDialog save_location_dialog;
    //save_location_dialog.getSaveFileName()
    QString dialog_file_name = QFileDialog::getSaveFileName(this,tr("Save frames as raw"),filename_edit.text(),tr("Raw (*.raw *.bin *.hsi *.img)"));
    if(!dialog_file_name.isEmpty())
    {
        filename_edit.setText(dialog_file_name);
    }
}

void ControlsBox::getMaskFile()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Select mask file"),
                                                    "",
                                                    tr("Files (*.raw)"));
    if(fileName.isEmpty())
    {
        return;
    }
    std::string utf8_text = fileName.toUtf8().constData();
    emit mask_selected(utf8_text.c_str());

}
void ControlsBox::increment_slot(bool t)
{
    if(t)
    {
        ceiling_slider.setTickInterval(LIL_TICK);
        floor_slider.setTickInterval(LIL_TICK);
        ceiling_slider.setMaximum(LIL_MAX);
        ceiling_slider.setMinimum(LIL_MIN);
        floor_slider.setMaximum(LIL_MAX);
        floor_slider.setMinimum(LIL_MIN);
    }
    else
    {
        ceiling_slider.setTickInterval(BIG_TICK);
        floor_slider.setTickInterval(BIG_TICK);
        ceiling_slider.setMaximum(BIG_MAX);
        ceiling_slider.setMinimum(BIG_MIN);
        floor_slider.setMaximum(BIG_MAX);
        floor_slider.setMinimum(BIG_MIN);
    }
}

void ControlsBox::save_continous_button_slot()
{
    emit startSavingContinous(filename_edit.text().toLocal8Bit().data());
    stop_saving_frames_button.setEnabled(true);

    start_saving_frames_button.setEnabled(false);
    save_finite_button.setEnabled(false);
    frames_save_num_edit.setEnabled(false);


}
void ControlsBox::stop_continous_button_slot()
{
    emit stopSaving();

    stop_saving_frames_button.setEnabled(false);
    start_saving_frames_button.setEnabled(true);
    save_finite_button.setEnabled(true);
    frames_save_num_edit.setEnabled(true);

}
void ControlsBox::updateSaveFrameNum_slot(unsigned int n)
{

    if(n == 0)
    {
        stop_saving_frames_button.setEnabled(false);
        start_saving_frames_button.setEnabled(true);
        save_finite_button.setEnabled(true);
        frames_save_num_edit.setEnabled(true);
    }
    frames_save_num_edit.setValue(n);
}

void ControlsBox::save_finite_button_slot()
{
    qDebug() << "fname: " << filename_edit.text();
    emit startSavingFinite(frames_save_num_edit.value(),filename_edit.text());
    stop_saving_frames_button.setEnabled(true);
    start_saving_frames_button.setEnabled(false);
    save_finite_button.setEnabled(false);
    frames_save_num_edit.setEnabled(false);




}

void ControlsBox::tabChangedSlot(int index)
{
    //Multiple inheritance totally failed me. Trying to get QObject pure virtual interfaces is like boxing with satan
    frameview_widget * fvw = qobject_cast<frameview_widget*>(qtw->widget(index));
    mean_profile_widget * mpw = qobject_cast<mean_profile_widget*>(qtw->widget(index));
    fft_widget * ffw = qobject_cast<fft_widget*>(qtw->widget(index));
    histogram_widget * hwt  = qobject_cast<histogram_widget*>(qtw->widget(index));
    if(qobject_cast<frameview_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<frameview_widget*>(cur_frameview),SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<frameview_widget*>(cur_frameview), SLOT(updateFloor(int)));
    }
    else if(qobject_cast<mean_profile_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<mean_profile_widget*>(cur_frameview),SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<mean_profile_widget*>(cur_frameview), SLOT(updateFloor(int)));
    }
    else if(qobject_cast<fft_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<fft_widget*>(cur_frameview),SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<fft_widget*>(cur_frameview), SLOT(updateFloor(int)));
    }
    else if(qobject_cast<histogram_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<histogram_widget*>(cur_frameview),SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<histogram_widget*>(cur_frameview), SLOT(updateFloor(int)));
    }
    cur_frameview = qtw->widget(index);

    if(fvw != NULL)
    {
        ceiling_edit.setValue(fvw->getCeiling());
        floor_edit.setValue(fvw->getFloor());
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), fvw, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), fvw, SLOT(updateFloor(int)));
        useDSFCbox.setEnabled(false);
        fvw->rescaleRange();

    }
    else if(mpw != NULL)
    {
        ceiling_edit.setValue(mpw->getCeiling());
        floor_edit.setValue(mpw->getFloor());
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), mpw, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), mpw, SLOT(updateFloor(int)));
        useDSFCbox.setEnabled(true);
        mpw->rescaleRange();
    }
    else if(ffw != NULL)
    {

        ceiling_edit.setValue(ffw->getCeiling());
        floor_edit.setValue(ffw->getFloor());
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), ffw, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), ffw, SLOT(updateFloor(int)));
        useDSFCbox.setEnabled(true);
        ffw->rescaleRange();
    }
    else if(hwt != NULL)
    {
        ceiling_edit.setValue(hwt->getCeiling());
        floor_edit.setValue(hwt->getFloor());
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), hwt, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), hwt, SLOT(updateFloor(int)));
        useDSFCbox.setEnabled(false);
        hwt->rescaleRange();
    }
}
void ControlsBox::updateBackendDelta()
{
    static unsigned long old_c = 0;
    unsigned int delta = fw->c-old_c;
    old_c = fw->c;
    fps_label.setText(QString("FPS @ backend:%1").arg(delta));
}
void ControlsBox::start_dark_collection_slot()
{
    collect_dark_frames_button.setEnabled(false);
    stop_dark_collection_button.setEnabled(true);
    emit startDSFMaskCollection();
}
void ControlsBox::stop_dark_collection_slot()
{
    collect_dark_frames_button.setEnabled(true);
    stop_dark_collection_button.setEnabled(false);
    emit stopDSFMaskCollection();


}
