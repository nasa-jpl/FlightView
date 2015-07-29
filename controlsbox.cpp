#include "controlsbox.h"
#include "fft_widget.h"
#include "frameview_widget.h"
#include "histogram_widget.h"
#include "playback_widget.h"
#include "profile_widget.h"

ControlsBox::ControlsBox(frameWorker *fw, QTabWidget *tw, QWidget *parent) :
    QGroupBox(parent)
{
    this->fw = fw;
    qtw = tw;
    cur_frameview = NULL;

    prefWindow = new preferenceWindow( fw, tw );

/* ====================================================================== */
    // LEFT SIDE BUTTONS (Collections)
    collect_dark_frames_button.setText("Record Dark Frames");
    stop_dark_collection_button.setText("Stop Dark Frames");
    stop_dark_collection_button.setEnabled(false);
    load_mask_from_file.setText("Load Dark Mask");
    load_mask_from_file.setEnabled(false);
    pref_button.setText("Change Preferences");
    fps_label.setText("Warning: No Data Recieved");
    server_ip_label.setText("Server IP: Not Connected!");
    server_port_label.setText("Port Number: Not Connected!");

    //First Row
    collections_layout.addWidget(&collect_dark_frames_button,1,1,1,1);
    collections_layout.addWidget(&stop_dark_collection_button,1,2,1,1);

    //Second Row
    collections_layout.addWidget(&fps_label,2,1,1,1);
    collections_layout.addWidget(&load_mask_from_file,2,2,1,1);

    //Third Row
    collections_layout.addWidget(&pref_button,3,2,1,1);
    collections_layout.addWidget(&server_ip_label,3,1,1,1);

    //Fourth Row
    collections_layout.addWidget(&server_port_label,4,1,1,1);

    CollectionButtonsBox.setLayout(&collections_layout);

/* ======================================================================== */
    //MIDDLE BUTTONS (Sliders)
    std_dev_n_label.setText("Std. Dev. N:");
    std_dev_N_slider.setOrientation(Qt::Horizontal);
    std_dev_N_slider.setMinimum(1);
    std_dev_N_slider.setMaximum(MAX_N-1);
    std_dev_N_slider.setValue(std_dev_N_slider.maximum());

    lines_label.setText("Lines to Average:");
    lines_slider.setOrientation(Qt::Horizontal);
    lines_slider.setMinimum(1); // We don't set the maximum of this slider until later, it is dependent on the type of profile
    lines_slider.setSingleStep(2);
    lines_slider.setTickInterval(2);

    ceiling_slider.setOrientation(Qt::Horizontal);
    ceiling_slider.setMaximum(BIG_MAX);
    ceiling_slider.setMinimum(BIG_MIN);
    ceiling_slider.setTickInterval(BIG_TICK);

    floor_slider.setOrientation(Qt::Horizontal);
    floor_slider.setMaximum(BIG_MAX);
    floor_slider.setMinimum(BIG_MIN);
    floor_slider.setTickInterval(BIG_TICK);

    std_dev_N_edit.setMinimum(1);
    std_dev_N_edit.setMaximum(MAX_N-1);
    std_dev_N_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);
    std_dev_N_edit.setValue(std_dev_N_edit.maximum());

    line_average_edit.setMinimum(1); // We don't set the maximum of this slider until later, it is dependent on the type of profile
    line_average_edit.setSingleStep(2);
    line_average_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);

    ceiling_edit.setMinimum(BIG_MIN);
    ceiling_edit.setMaximum(BIG_MAX);
    ceiling_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);

    floor_edit.setMinimum(BIG_MIN);
    floor_edit.setMaximum(BIG_MAX);
    floor_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);

    low_increment_cbox.setText("Precision Slider");
    use_DSF_cbox.setText("Apply Dark Subtraction Filter");

    //First Row
    sliders_layout.addWidget(&std_dev_n_label,1,1,1,1);
    sliders_layout.addWidget(&std_dev_N_slider,1,2,1,7);
    sliders_layout.addWidget(&std_dev_N_edit,1,10,1,1);

    //Second Row
    sliders_layout.addWidget(&low_increment_cbox,2,1,1,1);
    sliders_layout.addWidget(&use_DSF_cbox,2,2,1,1);

    //Third Row
    sliders_layout.addWidget(new QLabel("Ceiling:"),3,1,1,1);
    sliders_layout.addWidget(&ceiling_slider,3,2,1,7);
    sliders_layout.addWidget(&ceiling_edit,3,10,1,1);

    //Fourth Row
    sliders_layout.addWidget(new QLabel("Floor:"),4,1,1,1);
    sliders_layout.addWidget(&floor_slider,4,2,1,7);
    sliders_layout.addWidget(&floor_edit,4,10,1,1);

    ThresholdingSlidersBox.setLayout(&sliders_layout);

/* ====================================================================== */
    //RIGHT SIDE BUTTONS (Save)
    select_save_location.setText("Select Save Location");

    save_finite_button.setText("Save Frames");

    start_saving_frames_button.setText("Start Saving");
    stop_saving_frames_button.setText("Stop Saving");
    stop_saving_frames_button.setEnabled(false);

    frames_save_num_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);
    frames_save_num_edit.setMinimum(1);
    frames_save_num_edit.setMaximum(100000);

    //First Row
    save_layout.addWidget(&select_save_location,1,1,1,1);
    save_layout.addWidget(new QLabel("Frames to save:"),2,1,1,1);
    save_layout.addWidget(new QLabel("Filename:"),3,1,1,1);

    //Second Row
    //save_layout.addWidget(&start_saving_frames_button,1,2,1,1);
    save_layout.addWidget(&save_finite_button,1,2,1,1);
    save_layout.addWidget(&frames_save_num_edit,2,2,1,2);
    save_layout.addWidget(&filename_edit,3,2,1,2);

    //Third Row
    save_layout.addWidget(&stop_saving_frames_button,1,3,1,1);

    SaveButtonsBox.setLayout(&save_layout);

/* =========================================================================== */
    // OVERALL LAYOUT
    controls_layout.addWidget(&CollectionButtonsBox,2);
    controls_layout.addWidget(&ThresholdingSlidersBox,3);
    controls_layout.addWidget(&SaveButtonsBox,2);
    this->setLayout(&controls_layout);
    this->setMaximumHeight(150);

/* =========================================================================== */
    //Connections
    connect(&collect_dark_frames_button,SIGNAL(clicked()),this,SLOT(start_dark_collection_slot()));
    connect(&stop_dark_collection_button,SIGNAL(clicked()),this,SLOT(stop_dark_collection_slot()));
    connect(&load_mask_from_file,SIGNAL(clicked()),this,SLOT(getMaskFile()));
    connect(&pref_button,SIGNAL(clicked()),this,SLOT(load_pref_window()));
    connect(fw,SIGNAL(updateFPS()),this,SLOT(update_backend_delta()));

    connect(&std_dev_N_edit,SIGNAL(valueChanged(int)),&std_dev_N_slider,SLOT(setValue(int)));
    connect(&std_dev_N_slider,SIGNAL(valueChanged(int)),&std_dev_N_edit,SLOT(setValue(int)));
    connect(&line_average_edit,SIGNAL(valueChanged(int)),&lines_slider,SLOT(setValue(int)));
    connect(&lines_slider,SIGNAL(valueChanged(int)),&line_average_edit,SLOT(setValue(int)));
    connect(&ceiling_edit,SIGNAL(valueChanged(int)),&ceiling_slider,SLOT(setValue(int)));
    connect(&ceiling_slider,SIGNAL(valueChanged(int)),&ceiling_edit,SLOT(setValue(int)));
    connect(&floor_edit,SIGNAL(valueChanged(int)),&floor_slider,SLOT(setValue(int)));
    connect(&floor_slider,SIGNAL(valueChanged(int)),&floor_edit,SLOT(setValue(int)));
    connect(&use_DSF_cbox,SIGNAL(toggled(bool)),this,SLOT(use_DSF_general(bool)));
    connect(&low_increment_cbox,SIGNAL(toggled(bool)),this,SLOT(increment_slot(bool)));

    connect(&save_finite_button,SIGNAL(clicked()),this,SLOT(save_finite_button_slot()));
    connect(&stop_saving_frames_button,SIGNAL(clicked()),this,SLOT(stop_continous_button_slot()));
    connect(&select_save_location,SIGNAL(clicked()),this,SLOT(show_save_dialog()));

    backendDeltaTimer.start();
}

// public slot(s)
void ControlsBox::tab_changed_slot(int index)
{
    // Noah: Multiple inheritance totally failed me. Trying to get QObject pure virtual interfaces is like boxing with satan. This hideous function is the best I can do...
    // JP sez: You know what this has going for it??? It works!! :^) >:^( ;^)

    frameview_widget * fvw = qobject_cast<frameview_widget*>(qtw->widget(index));
    profile_widget * mpw = qobject_cast<profile_widget*>(qtw->widget(index));
    fft_widget * ffw = qobject_cast<fft_widget*>(qtw->widget(index));
    histogram_widget * hwt  = qobject_cast<histogram_widget*>(qtw->widget(index));
    playback_widget* pbw = qobject_cast<playback_widget*>(qtw->widget(index));

    /* When we change tabs, we need to disconnect widget-specfic controls so that we can create new connections between the same buttons
     * in the controls box. Any connection that we make when opening a tab needs to be disconnected when we move away from the tab. Otherwise,
     * a button will connect multiple times and cause bugs. The qobject_cast lines attempts to create a pointer to a specific widget based on
     * the general QWidget of the current tab that we have open in the QTabWidget in mainwindow.cpp.  */
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
    else if(qobject_cast<playback_widget*>(cur_frameview) != NULL)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), qobject_cast<playback_widget*>(cur_frameview), SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), qobject_cast<playback_widget*>(cur_frameview), SLOT(updateFloor(int)));
        disconnect(this, SIGNAL(mask_selected(QString, unsigned int, long)), qobject_cast<playback_widget*>(cur_frameview), \
                   SLOT(loadDSF(QString, unsigned int, long)));
        load_mask_from_file.setEnabled(false); // The playback widget is the only widget which currently uses the load mask button
        qobject_cast<playback_widget*>(cur_frameview)->stop();
    }

    // Next, we set up the widget of the current tab.
    cur_frameview = qtw->widget(index);
    if(mpw)
    {
        ceiling_edit.setValue(mpw->getCeiling());
        floor_edit.setValue(mpw->getFloor());
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), mpw, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), mpw, SLOT(updateFloor(int)));

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
            mpw->updateCrossRange(fw->vertLinesAvgd);
            lines_slider.setValue(fw->vertLinesAvgd);
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

        use_DSF_cbox.setEnabled(true);
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

        ffw->updateFFT();
        int frameMax = 0;
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

        use_DSF_cbox.setEnabled(true);
        ceiling_maximum = ffw->slider_max;
        low_increment_cbox.setChecked(ffw->slider_low_inc);
        this->increment_slot(low_increment_cbox.isChecked());

        ffw->rescaleRange();
    }
    else
    {
        // the other widgets, which do not use the lines_slider, should hide it and its related controls
        lines_slider.setVisible( false );
        line_average_edit.setVisible( false );
        lines_label.setVisible( false );
        sliders_layout.removeWidget(&lines_slider);
        sliders_layout.removeWidget(&line_average_edit);
        sliders_layout.removeWidget(&lines_label);
        fw->to.updateVertRange(0,fw->getFrameHeight()); // resets the range of the mean to standard dimensions
        fw->to.updateHorizRange(0,fw->getFrameWidth());

        std_dev_n_label.setVisible( true );
        std_dev_N_slider.setVisible( true );
        std_dev_N_edit.setVisible( true );
        sliders_layout.addWidget(&std_dev_n_label,1,1,1,1);
        sliders_layout.addWidget(&std_dev_N_slider,1,2,1,7);
        sliders_layout.addWidget(&std_dev_N_edit,1,10,1,1);

        if(fvw != NULL)
        {
            ceiling_edit.setValue(fvw->getCeiling());
            floor_edit.setValue(fvw->getFloor());
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), fvw, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), fvw, SLOT(updateFloor(int)));

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

            use_DSF_cbox.setEnabled(false);
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

            std_dev_N_slider.setEnabled(true);
            std_dev_N_edit.setEnabled(true);

            use_DSF_cbox.setEnabled(false);
            ceiling_maximum = hwt->slider_max;
            low_increment_cbox.setChecked(hwt->slider_low_inc);
            this->increment_slot(low_increment_cbox.isChecked());

            hwt->rescaleRange();
        }
        else if(pbw != NULL)
        {
            ceiling_edit.setValue(pbw->getCeiling());
            floor_edit.setValue(pbw->getFloor());
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), pbw, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), pbw, SLOT(updateFloor(int)));

            std_dev_N_slider.setEnabled(false);
            std_dev_N_edit.setEnabled(false);
            load_mask_from_file.setEnabled(true);
            connect(this, SIGNAL(mask_selected(QString, unsigned int, long)), pbw, SLOT(loadDSF(QString, unsigned int, long)));

            use_DSF_cbox.setEnabled(true);
            ceiling_maximum = pbw->slider_max;
            low_increment_cbox.setChecked(pbw->slider_low_inc);
            this->increment_slot(low_increment_cbox.isChecked());

            pbw->rescaleRange();
        }
    }
}

// private slots
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
    else if(qobject_cast<playback_widget*>(cur_frameview) != NULL)
        qobject_cast<playback_widget*>(cur_frameview)->slider_low_inc = t;
}
void ControlsBox::update_backend_delta()
{
    fps = QString::number(fw->delta, 'f', 1);
    fps_label.setText(QString("FPS @ backend:%1").arg(fps));
}
void ControlsBox::show_save_dialog()
{
    QString dialog_file_name = QFileDialog::getSaveFileName(this,tr("Save frames as raw"),filename_edit.text(),tr("Raw (*.raw *.bin *.hsi *.img)"));
    if(!dialog_file_name.isEmpty())
    {
        filename_edit.setText(dialog_file_name);
    }
}
void ControlsBox::save_finite_button_slot()
{
#ifdef VERBOSE
    qDebug() << "fname: " << filename_edit.text();
#endif
    emit startSavingFinite(frames_save_num_edit.value(),filename_edit.text());
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
void ControlsBox::start_dark_collection_slot()
{
    collect_dark_frames_button.setEnabled(false);
    stop_dark_collection_button.setEnabled(true);
    emit startDSFMaskCollection();
}
void ControlsBox::stop_dark_collection_slot()
{
    emit stopDSFMaskCollection();
    collect_dark_frames_button.setEnabled(true);
    stop_dark_collection_button.setEnabled(false);
}
void ControlsBox::getMaskFile()
{
    // Step 1: Open a dialog to select the mask file location
    QFileDialog location_dialog(0);
    location_dialog.setFilter(QDir::Writable | QDir::Files);
    QString fileName = location_dialog.getOpenFileName(this, tr("Select mask file"),"",tr("Files (*.raw)"));
    if(fileName.isEmpty())
    {
        return;
    }

    // Step 2: Calculate the number of frames
    FILE* mask_file;
    unsigned long mask_size = 0;
    unsigned long frame_size = fw->getFrameHeight()*fw->getFrameWidth();
    unsigned long num_frames = 0;
    mask_file = fopen(fileName.toStdString().c_str(), "rb");
    if(!mask_file)
    {
        std::cerr << "Error: Mask file could not be loaded." << std::endl;
        return;
    }
    fseek (mask_file, 0, SEEK_END); // non-portable
    mask_size = ftell(mask_file);
    num_frames = mask_size / (frame_size*sizeof(uint16_t));
    fclose(mask_file);
    if(!mask_size)
    {
        std::cerr << "Error: Mask file contains no data." << std::endl;
        return;
    }

    // Step 3: Open a new dialog to select which frames to read
    QDialog bytes_dialog;
    bytes_dialog.setWindowTitle("Select Read Area");
    QLabel status_label;
    status_label.setText(tr("The selected file contains %1 frames.\nPlease select which frames to use for the Dark Subtraction Mask.") \
                         .arg(num_frames));
    QSpinBox left_bound;
    QSpinBox right_bound;
    left_bound.setMinimum(1);
    left_bound.setMaximum(num_frames);
    left_bound.setValue(1);
    right_bound.setMinimum(1);
    right_bound.setMaximum(num_frames);
    right_bound.setValue(num_frames);
    QPushButton* select_range = new QPushButton(tr("&Select Range"));
    select_range->setDefault(true);
    QPushButton* cancel = new QPushButton(tr("&Cancel"));
    QDialogButtonBox* buttons = new QDialogButtonBox(Qt::Horizontal);
    buttons->addButton(select_range, QDialogButtonBox::AcceptRole);
    buttons->addButton(cancel, QDialogButtonBox::RejectRole);
    connect(buttons,SIGNAL(rejected()),&bytes_dialog,SLOT(reject()));
    connect(buttons,SIGNAL(accepted()),&bytes_dialog,SLOT(accept()));
    QGridLayout bd_layout;
    bd_layout.addWidget(&status_label, 0, 0, 1, 4);
    bd_layout.addWidget(new QLabel("Read from frame:"), 1, 0, 1, 1);
    bd_layout.addWidget(&left_bound, 1, 1, 1, 1);
    bd_layout.addWidget(new QLabel(" to "), 1, 2, 1, 1);
    bd_layout.addWidget(&right_bound, 1, 3, 1, 1);
    bd_layout.addWidget(buttons, 2, 1, 1, 4);
    bytes_dialog.setLayout(&bd_layout);
    bytes_dialog.show();
    int result = bytes_dialog.exec();

    // Step 4: Check that the given range is acceptable
    if(result == QDialog::Accepted)
    {
        int lo_val = left_bound.value();
        int hi_val = right_bound.value();
        int elem_to_read = hi_val - lo_val + 1;
        long offset = (lo_val-1)*frame_size;
        if(elem_to_read > 0)
        {
            elem_to_read *= frame_size;
            emit mask_selected(fileName, (unsigned int)elem_to_read, offset);
        }
        else if(elem_to_read == 0)
        {
            elem_to_read = frame_size;
            emit mask_selected(fileName, (unsigned int)elem_to_read, offset);
        }
        else
        {
            std::cerr << "Error: The selected range of dark frames is invalid." << std::endl;
            return;
        }
    }
    else
    {
        return;
    }
}
void ControlsBox::use_DSF_general(bool checked)
{
    playback_widget* pbw = qobject_cast<playback_widget*>(cur_frameview);
    if(pbw)
    {
        pbw->toggleUseDSF(checked);
    }
    else
    {
        fw->toggleUseDSF(checked);
    }
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
