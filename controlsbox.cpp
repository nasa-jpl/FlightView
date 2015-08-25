#include "controlsbox.h"

#include <QStyle>

static const char notAllowedChars[]   = ",^@=+{}[]~!?:&*\"|#%<>$\"'();`' ";

ControlsBox::ControlsBox(frameWorker *fw, QTabWidget *tw, QWidget *parent) :
    QGroupBox(parent)
{
    /*! \brief The main controls for liveview2
     * The ControlsBox is a wrapper GUI class which contains the (mostly) static controls between widgets.
     * After establishing the buttons in the constructor, the class will call the function tab_changed_slot(int index)
     * to establish widget-specific controls and settings. For instance, all profile widgets and FFTs make use
     * of the Lines To Average slider rather than the disabled Std Dev N slider. As Qt does not support
     * a pure virtual interface for widgets, each widget must make a connection to its own version of
     * updateCeiling(int c), updateFloor(int f), and any other widget specfic action within its case in
     * tab_changed_slot. The beginning of this function specifies the behavior for when tabs are exited -
     * all connections made must be disconnected to prevent overlap and repetition.
     * \author JP Ryan
     * \author Noah Levy
     */
    this->fw = fw;
    qtw = tw;
    current_tab = qobject_cast<frameview_widget*>(qtw->widget(qtw->currentIndex()));
    old_tab = NULL;

    prefWindow = new preferenceWindow(fw, tw);

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

    collections_layout = new QGridLayout();
    //First Row
    collections_layout->addWidget(&collect_dark_frames_button, 1, 1, 1, 1);
    collections_layout->addWidget(&stop_dark_collection_button, 1, 2, 1, 1);

    //Second Row
    collections_layout->addWidget(&fps_label, 2, 1, 1, 1);
    collections_layout->addWidget(&load_mask_from_file, 2, 2, 1, 1);

    //Third Row
    collections_layout->addWidget(&pref_button, 3, 2, 1, 1);
    collections_layout->addWidget(&server_ip_label, 3, 1, 1, 1);

    //Fourth Row
    collections_layout->addWidget(&server_port_label, 4, 1, 1, 1);

    CollectionButtonsBox.setLayout(collections_layout);

/* ======================================================================== */
    //MIDDLE BUTTONS (Sliders)
    std_dev_n_label = new QLabel("Std. Dev. N:", this);
    std_dev_N_slider = new QSlider(this);
    std_dev_N_slider->setOrientation(Qt::Horizontal);
    std_dev_N_slider->setMinimum(1);
    std_dev_N_slider->setMaximum(MAX_N-1);
    std_dev_N_slider->setValue(std_dev_N_slider->maximum());

    lines_label = new QLabel("Lines to Average:", this);
    lines_slider = new QSlider(this);
    lines_slider->setOrientation(Qt::Horizontal);
    lines_slider->setMinimum(1); // We don't set the maximum of this slider until later, it is dependent on the type of profile
    lines_slider->setSingleStep(2);
    lines_slider->setTickInterval(2);

    ceiling_slider.setOrientation(Qt::Horizontal);
    ceiling_slider.setMaximum(BIG_MAX);
    ceiling_slider.setMinimum(BIG_MIN);
    ceiling_slider.setTickInterval(BIG_TICK);

    floor_slider.setOrientation(Qt::Horizontal);
    floor_slider.setMaximum(BIG_MAX);
    floor_slider.setMinimum(BIG_MIN);
    floor_slider.setTickInterval(BIG_TICK);

    std_dev_N_edit = new QSpinBox(this);
    std_dev_N_edit->setMinimum(1);
    std_dev_N_edit->setMaximum(MAX_N-1);
    std_dev_N_edit->setButtonSymbols(QAbstractSpinBox::NoButtons);
    std_dev_N_edit->setValue(std_dev_N_edit->maximum());

    line_average_edit = new QSpinBox(this);
    line_average_edit->setMinimum(1); // We don't set the maximum of this slider until later, it is dependent on the type of profile
    line_average_edit->setSingleStep(2);
    line_average_edit->setButtonSymbols(QAbstractSpinBox::NoButtons);

    ceiling_edit.setMinimum(BIG_MIN + 1);
    ceiling_edit.setMaximum(ceiling_maximum);
    ceiling_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);

    floor_edit.setMinimum(BIG_MIN);
    floor_edit.setMaximum(ceiling_maximum - 1);
    floor_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);

    low_increment_cbox.setText("Precision Slider");
    use_DSF_cbox.setText("Apply Dark Subtraction Filter");

    sliders_layout = new QGridLayout();
    //First Row
    sliders_layout->addWidget(std_dev_n_label, 1, 1, 1, 1);
    sliders_layout->addWidget(std_dev_N_slider, 1, 2, 1, 7);
    sliders_layout->addWidget(std_dev_N_edit, 1, 10, 1, 1);

    //Second Row
    sliders_layout->addWidget(&low_increment_cbox, 2, 1, 1, 1);
    sliders_layout->addWidget(&use_DSF_cbox, 2, 2, 1, 1);

    //Third Row
    sliders_layout->addWidget(new QLabel("Ceiling:"),3,1,1,1);
    sliders_layout->addWidget(&ceiling_slider,3,2,1,7);
    sliders_layout->addWidget(&ceiling_edit,3,10,1,1);

    //Fourth Row
    sliders_layout->addWidget(new QLabel("Floor:"), 4, 1, 1, 1);
    sliders_layout->addWidget(&floor_slider, 4, 2, 1, 7);
    sliders_layout->addWidget(&floor_edit, 4, 10, 1, 1);

    ThresholdingSlidersBox.setLayout(sliders_layout);

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

    save_layout = new QGridLayout();
    //First Row
    save_layout->addWidget(&select_save_location, 1, 1, 1, 1);
    save_layout->addWidget(new QLabel("Frames to save:"), 2, 1, 1, 1);
    save_layout->addWidget(new QLabel("Filename:"), 3, 1, 1, 1);

    //Second Row
    //save_layout.addWidget(&start_saving_frames_button, 1, 2, 1, 1);
    save_layout->addWidget(&save_finite_button, 1, 2, 1, 1);
    save_layout->addWidget(&frames_save_num_edit, 2, 2, 1, 2);
    save_layout->addWidget(&filename_edit, 3, 2, 1, 2);

    //Third Row
    save_layout->addWidget(&stop_saving_frames_button, 1, 3, 1, 1);

    SaveButtonsBox.setLayout(save_layout);

/* =========================================================================== */
    // OVERALL LAYOUT
    controls_layout.addWidget(&CollectionButtonsBox, 2);
    controls_layout.addWidget(&ThresholdingSlidersBox, 3);
    controls_layout.addWidget(&SaveButtonsBox, 2);
    this->setLayout(&controls_layout);
    this->setMaximumHeight(150);

/* =========================================================================== */
    //Connections
    connect(&collect_dark_frames_button, SIGNAL(clicked()), this, SLOT(start_dark_collection_slot()));
    connect(&stop_dark_collection_button, SIGNAL(clicked()), this, SLOT(stop_dark_collection_slot()));
    connect(&load_mask_from_file, SIGNAL(clicked()), this, SLOT(getMaskFile()));
    connect(&pref_button, SIGNAL(clicked()), this, SLOT(load_pref_window()));
    connect(fw, SIGNAL(updateFPS()), this, SLOT(update_backend_delta()));

    connect(std_dev_N_edit, SIGNAL(valueChanged(int)), std_dev_N_slider, SLOT(setValue(int)));
    connect(std_dev_N_slider, SIGNAL(valueChanged(int)), std_dev_N_edit, SLOT(setValue(int)));
    connect(line_average_edit, SIGNAL(valueChanged(int)), lines_slider, SLOT(setValue(int)));
    connect(lines_slider, SIGNAL(valueChanged(int)), line_average_edit, SLOT(setValue(int)));
    connect(lines_slider, SIGNAL(valueChanged(int)), this, SLOT(transmitChange(int)));
    connect(&ceiling_edit, SIGNAL(valueChanged(int)), &ceiling_slider, SLOT(setValue(int)));
    connect(&ceiling_slider, SIGNAL(valueChanged(int)), &ceiling_edit, SLOT(setValue(int)));
    connect(&floor_edit, SIGNAL(valueChanged(int)), &floor_slider, SLOT(setValue(int)));
    connect(&floor_slider, SIGNAL(valueChanged(int)), &floor_edit, SLOT(setValue(int)));
    connect(&use_DSF_cbox, SIGNAL(toggled(bool)), this, SLOT(use_DSF_general(bool)));
    connect(&low_increment_cbox, SIGNAL(toggled(bool)), this, SLOT(increment_slot(bool)));

    connect(&save_finite_button, SIGNAL(clicked()), this, SLOT(save_finite_button_slot()));
    connect(&stop_saving_frames_button, SIGNAL(clicked()), this, SLOT(stop_continous_button_slot()));
    connect(&select_save_location, SIGNAL(clicked()), this, SLOT(show_save_dialog()));
}
void ControlsBox::closeEvent(QCloseEvent *e)
{
    /* Note: minor hack below */
    Q_UNUSED(e);
    prefWindow->close();
}

// public slot(s)
void ControlsBox::tab_changed_slot(int index)
{
    /*! \brief The function which adjusts the controls for the widgets when the tab in the main window is changed
     * Observe the structure before getting confused: First, we find the soon-to-be current widget by attempting to
     * make a non-null cast to the widget at the index we just switched to.
     * Then we disconnect the connections that we made setting up the "cur_frameview", i.e, the widget we are now leaving.
     * Then we make new connections and adjustments for the widget we just made a valid pointer to. Crucially, the top slider
     * differs depending on the widget. FFT widgets and Profile Widgets use it, all others use the Std Dev N slider.
     * \author Noah Levy
     * \author JP Ryan
     */
    current_tab = qtw->widget(index);
    disconnect_old_tab();
    attempt_pointers(current_tab);
    if (p_profile) {
        int frameMax, startVal;
        bool enable;
        use_DSF_cbox.setEnabled(true);
        use_DSF_cbox.setChecked(fw->usingDSF());
        ceiling_maximum = p_profile->slider_max;
        low_increment_cbox.setChecked(p_profile->slider_low_inc);
        increment_slot(low_increment_cbox.isChecked());
        ceiling_edit.setValue(p_profile->getCeiling());
        floor_edit.setValue(p_profile->getFloor());
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_profile, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), p_profile, SLOT(updateFloor(int)));

        frameMax = p_profile->itype == HORIZONTAL_MEAN || p_profile->itype == HORIZONTAL_CROSS \
                ? fw->getFrameHeight() : fw->getFrameWidth();
        startVal = p_profile->itype == HORIZONTAL_CROSS ? fw->vertLinesAvgd : frameMax;
        startVal = p_profile->itype == VERTICAL_CROSS ? fw->horizLinesAvgd : startVal;
        enable = (p_profile->itype == VERTICAL_CROSS || p_profile->itype == HORIZONTAL_CROSS) && fw->crosshair_x != -1;
        lines_slider->setMaximum(frameMax);
        line_average_edit->setMaximum(frameMax);
        lines_slider->setValue(startVal);
        fw->updateMeanRange(lines_slider->value(), p_profile->itype);
        lines_slider->setEnabled(enable);
        line_average_edit->setEnabled(enable);
        display_lines_slider();

        p_profile->rescaleRange();
    } else if (p_fft) {
        use_DSF_cbox.setEnabled(true);
        use_DSF_cbox.setChecked(fw->usingDSF());
        ceiling_maximum = p_fft->slider_max;
        low_increment_cbox.setChecked(p_fft->slider_low_inc);
        increment_slot(low_increment_cbox.isChecked());
        ceiling_edit.setValue(p_fft->getCeiling());
        floor_edit.setValue(p_fft->getFloor());
        connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_fft, SLOT(updateCeiling(int)));
        connect(&floor_slider, SIGNAL(valueChanged(int)), p_fft, SLOT(updateFloor(int)));

        p_fft->updateFFT();
        lines_slider->setMaximum(fw->getFrameWidth());
        line_average_edit->setMaximum(fw->getFrameWidth());
        lines_slider->setValue(fw->horizLinesAvgd);
        if (p_fft->vCrossButton->isChecked() && fw->crosshair_x != -1) {
            lines_slider->setEnabled(true);
            line_average_edit->setEnabled(true);
            transmitChange(fw->horizLinesAvgd);
        } else {
            lines_slider->setEnabled(false);
            line_average_edit->setEnabled(false);
        }
        connect(p_fft->vCrossButton, SIGNAL(toggled(bool)), this, SLOT(fft_slider_enable(bool)));
        connect(p_fft->vCrossButton, SIGNAL(toggled(bool)), this, SLOT(fft_slider_enable(bool)));
        display_lines_slider();

        p_fft->rescaleRange();
    } else {
        display_std_dev_slider();
        if (p_frameview) {
            ceiling_maximum = p_frameview->slider_max;
            low_increment_cbox.setChecked(p_frameview->slider_low_inc);
            increment_slot(low_increment_cbox.isChecked());
            ceiling_edit.setValue(p_frameview->getCeiling());
            floor_edit.setValue(p_frameview->getFloor());
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_frameview, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), p_frameview, SLOT(updateFloor(int)));

            if (p_frameview->image_type == STD_DEV) {
                std_dev_N_slider->setEnabled(true);
                std_dev_N_edit->setEnabled(true);
            } else {
                std_dev_N_slider->setEnabled(false);
                std_dev_N_edit->setEnabled(false);
            }
            use_DSF_cbox.setEnabled(false);
            use_DSF_cbox.setChecked(fw->usingDSF());
            fw->setCrosshairBackend(fw->crosshair_x, fw->crosshair_y);
            p_frameview->rescaleRange();
        } else if (p_histogram) {
            ceiling_maximum = p_histogram->slider_max;
            low_increment_cbox.setChecked(p_histogram->slider_low_inc);
            increment_slot(low_increment_cbox.isChecked());
            ceiling_edit.setValue(p_histogram->getCeiling());
            floor_edit.setValue(p_histogram->getFloor());
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_histogram, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), p_histogram, SLOT(updateFloor(int)));
            std_dev_N_slider->setEnabled(true);
            std_dev_N_edit->setEnabled(true);
            use_DSF_cbox.setEnabled(false);
            use_DSF_cbox.setChecked(fw->usingDSF());
            p_histogram->rescaleRange();
        } else if (p_playback) {
            ceiling_maximum = p_playback->slider_max;
            low_increment_cbox.setChecked(p_playback->slider_low_inc);
            this->increment_slot(low_increment_cbox.isChecked());
            ceiling_edit.setValue(p_playback->getCeiling());
            floor_edit.setValue(p_playback->getFloor());
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_playback, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), p_playback, SLOT(updateFloor(int)));
            std_dev_N_slider->setEnabled(false);
            std_dev_N_edit->setEnabled(false);
            load_mask_from_file.setEnabled(true);
            connect(this, SIGNAL(mask_selected(QString, unsigned int, long)), p_playback, SLOT(loadDSF(QString, unsigned int, long)));
            use_DSF_cbox.setEnabled(true);
            use_DSF_cbox.setChecked(p_playback->usingDSF());
            p_playback->rescaleRange();
        }
    }
}

// private slots
void ControlsBox::increment_slot(bool t)
{
    /*! \brief Handles the Precision Slider range adjustment
     * If the slider is set to increment low, we set the tick and step to 1, the max ceiling to 2000 (or whatever the values
     * of LIL_TICK and LIL_MAX happen to be. If the increment is set high, the max is set to BIG_MAX and the slider will scroll
     * through at the rate of BIG_TICK, which is 400. We then attempt to cast a pointer to the currently displayed widget, and
     * then adjust its copy of the variable slider_low_inc, which holds whether or not the precision slider mode is active.
     * \author Noah Levy
     */
    int minimum;
    int maximum = t ? LIL_MAX : BIG_MAX;;
    int tick = t ? LIL_TICK : BIG_TICK;
    attempt_pointers(current_tab);
    bool use_zero_min = p_histogram || p_fft;
    if (p_frameview)
        use_zero_min = p_frameview->image_type == STD_DEV;
    if(use_zero_min)
        minimum = 0;
    else
        minimum = t ? LIL_MIN : BIG_MIN;
    ceiling_slider.setMaximum(maximum);
    ceiling_slider.setMinimum(minimum);
    ceiling_slider.setSingleStep(tick);
    ceiling_edit.setMaximum(maximum);
    ceiling_edit.setMinimum(minimum);
    ceiling_edit.setSingleStep(tick);
    floor_slider.setMaximum(maximum);
    floor_slider.setMinimum(minimum);
    floor_slider.setSingleStep(tick);
    floor_edit.setMaximum(maximum);
    floor_edit.setMinimum(minimum);
    floor_edit.setSingleStep(tick);

    if (p_frameview)
        p_frameview->slider_low_inc = t;
    else if (p_histogram)
        p_histogram->slider_low_inc = t;
    else if (p_profile)
        p_profile->slider_low_inc = t;
    else if (p_fft)
        p_fft->slider_low_inc = t;
    else if (p_playback)
        p_playback->slider_low_inc = t;
}
void ControlsBox::attempt_pointers(QWidget *tab)
{
    p_frameview = qobject_cast<frameview_widget*>(tab);
    p_histogram = qobject_cast<histogram_widget*>(tab);
    p_profile = qobject_cast<profile_widget*>(tab);
    p_fft = qobject_cast<fft_widget*>(tab);
    p_playback = qobject_cast<playback_widget*>(tab);
}
void ControlsBox::disconnect_old_tab()
{
    attempt_pointers(old_tab);
    if (p_frameview) {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_frameview,SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_frameview, SLOT(updateFloor(int)));
    } else if (p_histogram) {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_histogram,SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_histogram, SLOT(updateFloor(int)));
    } else if (p_profile) {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_profile,SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_profile, SLOT(updateFloor(int)));
    } else if (p_fft) {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_fft,SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_fft, SLOT(updateFloor(int)));
        disconnect(p_fft->vCrossButton, SIGNAL(toggled(bool)), lines_slider,SLOT(setEnabled(bool)));
        disconnect(p_fft->vCrossButton, SIGNAL(toggled(bool)), line_average_edit,SLOT(setEnabled(bool)));
    } else if (p_playback) {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_playback, SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_playback, SLOT(updateFloor(int)));
        disconnect(this, SIGNAL(mask_selected(QString, unsigned int, long)), p_playback, \
                   SLOT(loadDSF(QString, unsigned int, long)));
        load_mask_from_file.setEnabled(false); // The playback widget is the only widget which currently uses the load mask button
        p_playback->stop();
    }
    old_tab = current_tab;
}
void ControlsBox::display_std_dev_slider()
{
    lines_slider->setVisible(false);
    line_average_edit->setVisible(false);
    lines_label->setVisible(false);
    sliders_layout->removeWidget(lines_slider);
    sliders_layout->removeWidget(line_average_edit);
    sliders_layout->removeWidget(lines_label);
    fw->to.updateVertRange(0, fw->getFrameHeight()); // resets the range of the mean to standard dimensions
    fw->to.updateHorizRange(0, fw->getFrameWidth());

    std_dev_n_label->setVisible(true);
    std_dev_N_slider->setVisible(true);
    std_dev_N_edit->setVisible(true);
    sliders_layout->addWidget(std_dev_n_label, 1, 1, 1, 1);
    sliders_layout->addWidget(std_dev_N_slider, 1, 2, 1, 7);
    sliders_layout->addWidget(std_dev_N_edit, 1, 10, 1, 1);
}
void ControlsBox::display_lines_slider()
{
    std_dev_n_label->setVisible(false);
    std_dev_N_slider->setVisible(false);
    std_dev_N_edit->setVisible(false);
    sliders_layout->removeWidget(std_dev_n_label);
    sliders_layout->removeWidget(std_dev_N_slider);
    sliders_layout->removeWidget(std_dev_N_edit);

    lines_label->setVisible(true);
    lines_slider->setVisible(true);
    line_average_edit->setVisible(true);
    sliders_layout->addWidget(lines_label, 1, 1, 1, 1);
    sliders_layout->addWidget(lines_slider, 1, 2, 1, 7);
    sliders_layout->addWidget(line_average_edit, 1, 10, 1, 1);
}
void ControlsBox::update_backend_delta()
{
    /*! \brief Update the QLabel text of the frame rate measured by the frameWorker event loop.
    * \author Noah Levy
    */
    fps = QString::number(fw->delta, 'f', 1);
    fps_label.setText(QString("FPS @ backend:%1").arg(fps));
}
void ControlsBox::show_save_dialog()
{
    /*! \brief Display the file dialog to specify the path for saving raw frames.
     * \author Noah Levy
     */
    QString dialog_file_name = QFileDialog::getSaveFileName(this, tr("Save frames as raw"), "/home/", tr("Raw (*.raw *.bin *.hsi *.img)"));
    if (!dialog_file_name.isEmpty())
        filename_edit.setText(dialog_file_name);
}
void ControlsBox::save_remote_slot(const QString &unverifiedName, unsigned int nFrames)
{
    filename_edit.setText(unverifiedName);
    frames_save_num_edit.setValue(nFrames);
    save_finite_button.click();
}
void ControlsBox::save_finite_button_slot()
{
    /*! \brief Emit the signal to save frames at the backend.
     * \author Noah Levy
     */
#ifdef VERBOSE
    qDebug() << "fname: " << filename_edit.text();
#endif
    QString errorString;
    if (validateFileName(filename_edit.text(), &errorString)) {
        emit startSavingFinite(frames_save_num_edit.value(), filename_edit.text());
        previousNumSaved = frames_save_num_edit.value();
        stop_saving_frames_button.setEnabled(true);
        start_saving_frames_button.setEnabled(false);
        save_finite_button.setEnabled(false);
        frames_save_num_edit.setEnabled(false);
    } else {
        QDialog *errorDialog = new QDialog(this);
        QLabel *message = new QLabel(errorString);
        QPushButton *okButton = new QPushButton("Ok");
        connect(okButton, SIGNAL(clicked()), errorDialog, SLOT(close()));
        QVBoxLayout layout;
        layout.addWidget(message);
        layout.addWidget(okButton);
        errorDialog->setLayout(&layout);
        errorDialog->setWindowTitle("Frame Save Error");
        errorDialog->setAttribute(Qt::WA_DeleteOnClose);
        errorDialog->show();
    }
}
void ControlsBox::stop_continous_button_slot()
{
    /*! \brief Emit the signal to stop saving frames at the backend. Handled automatically.
     * \author: Noah Levy
     */
    emit stopSaving();
    stop_saving_frames_button.setEnabled(false);
    start_saving_frames_button.setEnabled(true);
    save_finite_button.setEnabled(true);
    frames_save_num_edit.setEnabled(true);
    frames_save_num_edit.setValue(previousNumSaved);
}
void ControlsBox::updateSaveFrameNum_slot(unsigned int n)
{
    /*! \brief Change the value in the box which displays the number of frames to save.
     * As frames are saved, the number in the readout will decrease as a sanity check.
     * \author Noah Levy
     */
    if (n == 0) {
        stop_saving_frames_button.setEnabled(false);
        start_saving_frames_button.setEnabled(true);
        save_finite_button.setEnabled(true);
        frames_save_num_edit.setEnabled(true);
    }
    frames_save_num_edit.setValue(n);
}
bool ControlsBox::validateFileName(const QString &name, QString *errorMessage)
{
    // No filename
    if (name.isEmpty()) {
        if(errorMessage)
            *errorMessage = tr("File name is empty.");
        return false;
    }

    // Characters
    for (const char *c = notAllowedChars; *c; c++) {
        if (name.contains(QLatin1Char(*c))) {
            if (errorMessage) {
                const QChar qc = QLatin1Char(*c);
                *errorMessage = tr("Invalid character \"%1\" in file name.").arg(qc);
            }
            return false;
        }
    }

    // Starts with slash
    if (name.at(0) != '/') {
        if (errorMessage)
            *errorMessage = tr("File name must specify a path. \nPlease specify the directory from root \nat which"
                           " to save the file."); // Upper case code
        return false;
    }
    return true;
}
void ControlsBox::start_dark_collection_slot()
{ /*! \brief Begins recording dark frames in the backend
   *  \author JP Ryan
   */
    collect_dark_frames_button.setEnabled(false);
    stop_dark_collection_button.setEnabled(true);
    emit startDSFMaskCollection();
}
void ControlsBox::stop_dark_collection_slot()
{
    /*! \brief Stops recording dark frames in the backend
     *  \author JP Ryan
     */
    emit stopDSFMaskCollection();
    collect_dark_frames_button.setEnabled(true);
    stop_dark_collection_button.setEnabled(false);
}
void ControlsBox::getMaskFile()
{
    /*! \brief Load a file containing Dark Frames, and create a mask from the specified region of the file.
     * First opens a file dialog, verifies the file, then opens a QDialog which allows the user to select a range
     * of frames which contain dark that will then be averaged into a mask. A widget which uses this signal must be
     * able to receive the signal maskSelected(QString filename, unsigned int elem_to_read, long offset)
     * \author JP Ryan
     */
    /* Step 1: Open a dialog to select the mask file location */
    QFileDialog location_dialog(0);
    location_dialog.setFilter(QDir::Writable | QDir::Files);
    QString fileName = location_dialog.getOpenFileName(this, tr("Select mask file"),"",tr("Files (*.raw)"));
    if (fileName.isEmpty())
        return;

    /* Step 2: Calculate the number of frames */
    FILE * mask_file;
    unsigned long mask_size = 0;
    unsigned long frame_size = fw->getDataHeight() * fw->getFrameWidth();
    unsigned long num_frames = 0;
    mask_file = fopen(fileName.toStdString().c_str(), "rb");
    if (!mask_file) {
        std::cerr << "Error: Mask file could not be loaded." << std::endl;
        return;
    }
    fseek (mask_file, 0, SEEK_END); // non-portable
    mask_size = ftell(mask_file);
    num_frames = mask_size / (frame_size*sizeof(uint16_t));
    fclose(mask_file);
    if (!mask_size) {
        std::cerr << "Error: Mask file contains no data." << std::endl;
        return;
    }

    /* Step 3: Open a new dialog to select which frames to read */
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
    QPushButton *select_range = new QPushButton(tr("&Select Range"));
    select_range->setDefault(true);
    QPushButton *cancel = new QPushButton(tr("&Cancel"));
    QDialogButtonBox *buttons = new QDialogButtonBox(Qt::Horizontal);
    buttons->addButton(select_range, QDialogButtonBox::AcceptRole);
    buttons->addButton(cancel, QDialogButtonBox::RejectRole);
    connect(buttons, SIGNAL(rejected()), &bytes_dialog, SLOT(reject()));
    connect(buttons, SIGNAL(accepted()), &bytes_dialog, SLOT(accept()));
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

    /* Step 4: Check that the given range is acceptable */
    if(result == QDialog::Accepted) {
        int lo_val = left_bound.value();
        int hi_val = right_bound.value();
        int elem_to_read = hi_val - lo_val + 1;
        long offset = (lo_val-1)*frame_size;
        if (elem_to_read > 0) {
            elem_to_read *= frame_size;
            emit mask_selected(fileName, (unsigned int)elem_to_read, offset);
        } else if (elem_to_read == 0) {
            elem_to_read = frame_size;
            emit mask_selected(fileName, (unsigned int)elem_to_read, offset);
        } else {
            std::cerr << "Error: The selected range of dark frames is invalid." << std::endl;
            return;
        }
    } else {
        return;
    }
}
void ControlsBox::use_DSF_general(bool checked)
{
    /*! \brief Toggles the use of the static dark frame mask for the playback widget.
     * A stupid little function I made because the playback widget uses the pre-loaded mask
     * as opposed to the others which use the recorded mask.
     * \author JP Ryan
     */
    if (p_playback)
        p_playback->toggleUseDSF(checked);
    else
        fw->toggleUseDSF(checked);
}
void ControlsBox::load_pref_window()
{
    /*! \brief Displays the preference window one the screen
     *  \author JP Ryan
     */
    QPoint pos = prefWindow->pos();
    if (pos.x() < 0)
        pos.setX(0);
    if (pos.y() < 0)
        pos.setY(0);
    prefWindow->move(pos);
    prefWindow->show();
}
void ControlsBox::transmitChange(int linesToAverage)
{
    if (p_profile)
        fw->updateMeanRange(linesToAverage, p_profile->itype);
    else if (p_fft)
        fw->updateMeanRange(linesToAverage, VERTICAL_CROSS);
    else
        fw->updateMeanRange(linesToAverage, BASE);
}
void ControlsBox::fft_slider_enable(bool toggled)
{
    bool enable = fw->crosshair_x != -1 && toggled;
    lines_slider->setEnabled(enable);
    line_average_edit->setEnabled(enable);
}
