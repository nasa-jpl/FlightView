#include "controlsbox.h"
#include "frameview_widget.h"
#include "profile_widget.h"
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
    qtw = tw;
    cur_frameview = NULL;
    //Collection Buttons

    prefWindow = new preferenceWindow( fw, tw );

    run_collect_button.setText("Run Collect");
    run_display_button.setText("Run Display");
    run_display_button.setEnabled(false);
    //stop_collect_button.setText("Stop Collect");
    //stop_display_button.setText("Stop Display");
    stop_display_button.setEnabled(false);

    collect_dark_frames_button.setText("Collect Dark Frames");
    stop_dark_collection_button.setText("Stop Dark Collection");
    stop_dark_collection_button.setEnabled(false);
    load_mask_from_file.setText("Load Mask From File");
    load_mask_from_file.setEnabled(false);
    fps_label.setText("Warning: no data recieved");
    server_ip_label.setText( "Server IP: Not Connected!" );
    server_port_label.setText( "Port Number: Not Connected!" );

    pref_button.setText("Change Preferences");

    select_save_location.setText("Select save location");
    frames_save_num_edit.setMaximum(100000);
    //First Row
    //collections_layout.addWidget(&run_display_button,1,1,1,1);
    //collections_layout->addWidget(&run_display_button,1,2);
    //collections_layout.addWidget(&stop_display_button,1,2,1,1);
    //collections_layout->addWidget(&stop_display_button,1,4);

    //Second Row
    collections_layout.addWidget(&collect_dark_frames_button,1,1,1,1);
    collections_layout.addWidget(&stop_dark_collection_button,1,2,1,1);
    collections_layout.addWidget(&load_mask_from_file,2,2,1,1);
    collections_layout.addWidget(&pref_button,3,2,1,1);

    //Third Row
    collections_layout.addWidget(&fps_label,2,1,1,1);
    collections_layout.addWidget(&server_ip_label,3,1,1,1);
    collections_layout.addWidget(&server_port_label,4,1,1,1);

    CollectionButtonsBox.setLayout(&collections_layout);

    //Slider Thresholding Buttons
    //QGridLayout * sliders_layout = new QGridLayout();
    std_dev_N_slider.setOrientation(Qt::Horizontal);
    std_dev_N_slider.setMinimum(1);
    std_dev_N_slider.setMaximum(MAX_N-1);
    std_dev_N_slider.setValue(std_dev_N_slider.maximum());

    lines_label.setText("Lines to Average:");
    lines_slider.setOrientation(Qt::Horizontal);
    lines_slider.setMinimum(1);

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

    line_average_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);
    line_average_edit.setMinimum(1);
    line_average_edit.setSingleStep(2);

    low_increment_cbox.setText("Slider Low Increment?");
    ceiling_slider.setTickInterval(BIG_TICK);
    floor_slider.setTickInterval(BIG_TICK);

    lines_slider.setSingleStep(2);
    lines_slider.setTickInterval(2);

    useDSFCbox.setText("Use Dark Subtracted data for profiles and FFT?");
    //First Row
    std_dev_n_label.setText("Std. Dev. N:");
    sliders_layout.addWidget(&std_dev_n_label,1,1,1,1);
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

    connect(&lines_slider,SIGNAL(valueChanged(int)),&line_average_edit,SLOT(setValue(int)));
    connect(&line_average_edit,SIGNAL(valueChanged(int)),&lines_slider,SLOT(setValue(int)));

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

    connect(&pref_button,SIGNAL(clicked()),this,SLOT(load_pref_window()));

    connect(fw,SIGNAL(updateFPS()),this,SLOT(updateBackendDelta()));
    backendDeltaTimer.start();
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
    //QString fileName = QFileDialog::getOpenFileName(this, tr("Select mask file","",tr("Files (*.raw)"));
    QFileDialog dialog(0);

    dialog.setFilter(QDir::Writable | QDir::Files);
    QString fileName = dialog.getOpenFileName(this, tr("Select mask file"),"",tr("Files (*.raw)"));
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

        floor_slider.setMaximum(LIL_MAX);
        floor_slider.setMinimum(LIL_MIN);
        floor_edit.setMaximum(LIL_MAX);
        floor_edit.setMinimum(LIL_MIN);
    }
    else
    {
        ceiling_slider.setTickInterval(BIG_TICK);
        floor_slider.setTickInterval(BIG_TICK);
        ceiling_slider.setMaximum(ceiling_maximum);
        ceiling_slider.setMinimum(-1*ceiling_maximum/4);
        ceiling_edit.setMaximum(ceiling_maximum);
        ceiling_edit.setMinimum(-1*ceiling_maximum/4);

        floor_slider.setMaximum(ceiling_maximum);
        floor_slider.setMinimum(-1*ceiling_maximum/4);
        floor_edit.setMaximum(ceiling_maximum);
        floor_edit.setMinimum(-1*ceiling_maximum/4);
    }

    //Sad.... this is what not being able to inherit from a common view_widget_interface does...
    if(qobject_cast<frameview_widget*>(cur_frameview) != NULL)
        qobject_cast<frameview_widget*>(cur_frameview)->slider_low_inc = t;
    else if(qobject_cast<profile_widget*>(cur_frameview) != NULL)
        qobject_cast<profile_widget*>(cur_frameview)->slider_low_inc = t;
    else if(qobject_cast<fft_widget*>(cur_frameview) != NULL)
        qobject_cast<fft_widget*>(cur_frameview)->slider_low_inc = t;
    else if(qobject_cast<histogram_widget*>(cur_frameview) != NULL)
        qobject_cast<histogram_widget*>(cur_frameview)->slider_low_inc = t;


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
    // Noah: Multiple inheritance totally failed me. Trying to get QObject pure virtual interfaces is like boxing with satan. This hideous function is the best I can do...
    // JP sez: You know what this has going for it??? It works!! :^) >:^( ;^)

    frameview_widget * fvw = qobject_cast<frameview_widget*>(qtw->widget(index));
    profile_widget * mpw = qobject_cast<profile_widget*>(qtw->widget(index));
    fft_widget * ffw = qobject_cast<fft_widget*>(qtw->widget(index));
    histogram_widget * hwt  = qobject_cast<histogram_widget*>(qtw->widget(index));
    if(qobject_cast<frameview_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<frameview_widget*>(cur_frameview),SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<frameview_widget*>(cur_frameview), SLOT(updateFloor(int)));
    }
    else if(qobject_cast<profile_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<profile_widget*>(cur_frameview),SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<profile_widget*>(cur_frameview), SLOT(updateFloor(int)));
        disconnect(&lines_slider, SIGNAL(valueChanged(int)), qobject_cast<profile_widget*>(cur_frameview), SLOT(updateCrossRange(int)));
    }
    else if(qobject_cast<fft_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<fft_widget*>(cur_frameview),SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<fft_widget*>(cur_frameview), SLOT(updateFloor(int)));
        disconnect(&lines_slider, SIGNAL(valueChanged(int)),qobject_cast<fft_widget*>(cur_frameview), SLOT(updateCrossRange(int)));
        disconnect(qobject_cast<fft_widget*>(cur_frameview)->vCrossButton,SIGNAL(toggled(bool)),&lines_slider,SLOT(setEnabled(bool)));
        disconnect(qobject_cast<fft_widget*>(cur_frameview)->vCrossButton,SIGNAL(toggled(bool)),&line_average_edit,SLOT(setEnabled(bool)));
    }
    else if(qobject_cast<histogram_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<histogram_widget*>(cur_frameview),SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<histogram_widget*>(cur_frameview), SLOT(updateFloor(int)));
    }
    cur_frameview = qtw->widget(index);

    if(mpw)
    {
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), mpw, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), mpw, SLOT(updateFloor(int)));
        ceiling_edit.setValue(mpw->getCeiling());
        floor_edit.setValue(mpw->getFloor());

        int frameMax;
        switch(mpw->itype)
        {
        case VERTICAL_CROSS:
            frameMax = fw->getFrameWidth() - 1;
            lines_slider.setMaximum(frameMax);
            line_average_edit.setMaximum(frameMax);
            mpw->updateCrossRange(fw->horizLinesAvgd);
            lines_slider.setValue(fw->horizLinesAvgd);
            lines_slider.setEnabled( true );
            line_average_edit.setEnabled( true );
            break;
        case VERTICAL_MEAN:
            frameMax = fw->getFrameWidth();
            lines_slider.setMaximum(frameMax);
            line_average_edit.setMaximum(frameMax);
            mpw->updateCrossRange(frameMax);
            lines_slider.setValue(frameMax);
            lines_slider.setEnabled( false );
            line_average_edit.setEnabled( false );
            break;
        case HORIZONTAL_CROSS:
            frameMax = fw->getFrameHeight() - 1;
            lines_slider.setMaximum(frameMax);
            line_average_edit.setMaximum(frameMax);
            mpw->updateCrossRange(mpw->vertLinesAvgd);
            lines_slider.setValue(mpw->vertLinesAvgd);
            lines_slider.setEnabled( true );
            line_average_edit.setEnabled( true );
            break;
        case HORIZONTAL_MEAN:
            frameMax = fw->getFrameHeight();
            lines_slider.setMaximum(frameMax);
            line_average_edit.setMaximum(frameMax);
            mpw->updateCrossRange(frameMax);
            lines_slider.setValue(frameMax);
            lines_slider.setEnabled( false );
            line_average_edit.setEnabled( false );
            break;
        default:
            break;
        }
        connect(&lines_slider, SIGNAL(valueChanged(int)), mpw, SLOT(updateCrossRange(int)));

        useDSFCbox.setEnabled(true);
        std_dev_n_label.setVisible(false);
        std_dev_N_slider.setVisible(false);
        std_dev_N_edit.setVisible(false);
        sliders_layout.removeWidget(&std_dev_n_label);
        sliders_layout.removeWidget(&std_dev_N_slider);
        sliders_layout.removeWidget(&std_dev_N_edit);

        lines_label.setVisible( true );
        lines_slider.setVisible( true );
        line_average_edit.setVisible( true );

        sliders_layout.addWidget(&lines_label,1,1,1,1);
        sliders_layout.addWidget(&lines_slider,1,2,1,7);
        sliders_layout.addWidget(&line_average_edit,1,10,1,1);

        ceiling_maximum = mpw->slider_max;
        low_increment_cbox.setChecked(mpw->slider_low_inc);
        this->increment_slot(low_increment_cbox.isChecked());
        mpw->rescaleRange();
    }
    else if( ffw )
    {
        ceiling_edit.setValue(ffw->getCeiling());
        floor_edit.setValue(ffw->getFloor());
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), ffw, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), ffw, SLOT(updateFloor(int)));

        useDSFCbox.setEnabled(true);
        std_dev_n_label.setVisible(false);
        std_dev_N_slider.setVisible(false);
        std_dev_N_edit.setVisible(false);
        sliders_layout.removeWidget(&std_dev_n_label);
        sliders_layout.removeWidget(&std_dev_N_slider);
        sliders_layout.removeWidget(&std_dev_N_edit);

        lines_label.setVisible( true );
        lines_slider.setVisible( true );
        line_average_edit.setVisible( true );

        sliders_layout.addWidget(&lines_label,1,1,1,1);
        sliders_layout.addWidget(&lines_slider,1,2,1,7);
        sliders_layout.addWidget(&line_average_edit,1,10,1,1);

        ceiling_maximum = ffw->slider_max;
        low_increment_cbox.setChecked(ffw->slider_low_inc);
        this->increment_slot(low_increment_cbox.isChecked());

        ffw->updateFFT();
        int frameMax = 0;
        ffw->rescaleRange();
        frameMax = fw->getFrameWidth() - 1;
        lines_slider.setMaximum(frameMax);
        line_average_edit.setMaximum(frameMax);
        lines_slider.setValue(fw->horizLinesAvgd);
        if(ffw->vCrossButton->isChecked())
        {
            lines_slider.setEnabled( true );
            line_average_edit.setEnabled( true );
        }
        else
        {
            lines_slider.setEnabled( false );
            line_average_edit.setEnabled( false );
        }
        connect(ffw->vCrossButton,SIGNAL(toggled(bool)),&lines_slider,SLOT(setEnabled(bool)));
        connect(ffw->vCrossButton,SIGNAL(toggled(bool)),&line_average_edit,SLOT(setEnabled(bool)));
        connect(&lines_slider, SIGNAL(valueChanged(int)), ffw, SLOT(updateCrossRange(int)));
    }
    else
    {
        lines_slider.setVisible( false );
        line_average_edit.setVisible( false );
        lines_label.setVisible( false );
        sliders_layout.removeWidget(&lines_slider);
        sliders_layout.removeWidget(&line_average_edit);
        sliders_layout.removeWidget(&lines_label);

        sliders_layout.addWidget(&std_dev_n_label,1,1,1,1);
        sliders_layout.addWidget(&std_dev_N_slider,1,2,1,7);
        sliders_layout.addWidget(&std_dev_N_edit,1,10,1,1);
        std_dev_n_label.setVisible( true );
        std_dev_N_slider.setVisible( true );
        std_dev_N_edit.setVisible( true );
        fw->to.updateVertRange(0,fw->getFrameHeight());
        fw->to.updateHorizRange(0,fw->getFrameWidth());

        if(fvw != NULL)
        {
            ceiling_edit.setValue(fvw->getCeiling());
            floor_edit.setValue(fvw->getFloor());
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), fvw, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), fvw, SLOT(updateFloor(int)));
            useDSFCbox.setEnabled(false);
            if(fvw->image_type == STD_DEV)
            {
                std_dev_N_slider.setEnabled(true);
                std_dev_N_edit.setEnabled(true);
            }
            else
            {
                std_dev_N_slider.setEnabled(false);
                std_dev_N_edit.setEnabled(false);
            }
            ceiling_maximum = fvw->slider_max;
            low_increment_cbox.setChecked(fvw->slider_low_inc);
            this->increment_slot(low_increment_cbox.isChecked());
            fvw->rescaleRange();
        }
        else if(hwt != NULL)
        {
            ceiling_edit.setValue(hwt->getCeiling());
            floor_edit.setValue(hwt->getFloor());
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), hwt, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), hwt, SLOT(updateFloor(int)));
            useDSFCbox.setEnabled(false);
            std_dev_N_slider.setEnabled(true);
            std_dev_N_edit.setEnabled(true);
            ceiling_maximum = hwt->slider_max;
            low_increment_cbox.setChecked(hwt->slider_low_inc);
            this->increment_slot(low_increment_cbox.isChecked());
            hwt->rescaleRange();
        }
    }
}
void ControlsBox::updateBackendDelta()
{
    fps = QString::number(fw->delta, 'f', 1);
    fps_label.setText(QString("FPS @ backend:%1").arg(fps));
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
void ControlsBox::load_pref_window()
{
    QPoint pos = prefWindow->pos();
    if( pos.x() < 0 )
        pos.setX(0);
    if( pos.y() < 0 )
        pos.setY(0);
    prefWindow->move(pos);
    prefWindow->show();
}
