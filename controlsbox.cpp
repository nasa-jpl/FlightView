#include "controlsbox.h"
#include <stdio.h>

static const char notAllowedChars[]   = ",^@=+{}[]~!?:&*\"|#%<>$\"'();`'";

ControlsBox::ControlsBox(frameWorker *fw, QTabWidget *tw, startupOptionsType options, QWidget *parent) :
    QGroupBox(parent)
{
    /*! \brief The main controls for LiveView
     * The ControlsBox is a wrapper GUI class which contains the (mostly) static controls between widgets.
     * After establishing the buttons in the constructor, the class will call the function tab_changed_slot(int index)
     * to establish widget-specific controls and settings. For instance, all profile widgets and FFTs make use
     * of the Lines To Average slider rather than the disabled Std Dev N slider. As Qt does not support
     * a pure virtual interface for widgets, each widget must make a connection to its own version of
     * updateCeiling(int c), updateFloor(int f), and any other widget specfic action within its case in
     * tab_changed_slot. The beginning of this function specifies the behavior for when tabs are exited -
     * all connections made must be disconnected to prevent overlap and repetition.
     * \author Jackie Ryan
     * \author Noah Levy
     */
    this->fw = fw;
    frWidth = fw->getFrameWidth();
    frHeight = fw->getFrameHeight();
    qtw = tw;
    current_tab = qobject_cast<frameview_widget*>(qtw->widget(qtw->currentIndex()));
    old_tab = NULL;
    this->options = options;
    if(options.dataLocationSet && !options.dataLocation.isEmpty())
    {
        fnamegen.setMainDirectory(options.dataLocation);
        emit statusMessage(QString("Setting data storage location to %1")\
                           .arg(options.dataLocation));
    } else {
        fnamegen.setMainDirectory("/tmp");
        emit warningMessage("Data storage location not set, defaulting to /tmp");
    }
    if(!fnamegen.createDirectory())
    {
        emit errorMessage(QString("Could not create directory %1").arg(options.dataLocation));
    }

    settings = new QSettings();
    setDefaultSettings();
    loadSettings();

    prefWindow = new preferenceWindow(fw, tw, prefs);
    connect(prefWindow, &preferenceWindow::statusMessage,
            [=](const QString msg)
    {
        emit statusMessage(msg);
    });


    setupUI.acceptOptions(&options);
    setupUI.setModal(true);

/* ====================================================================== */
    // LEFT SIDE BUTTONS (Collections)
    collect_dark_frames_button.setText("Record Dark Frames");
    stop_dark_collection_button.setText("Stop Dark Frames");
    stop_dark_collection_button.setEnabled(false);
    load_mask_from_file.setText("Load Dark Mask");
    load_mask_from_file.setEnabled(false);
    pref_button.setText("Preferences");
    fps_label.setText("Warning: No Data Recieved");
    server_ip_label.setText("Server IP: Not Connected!");
    server_port_label.setText("Port Number: Not Connected!");

    // Overlay controls:

    //left:
    overlay_lh_width = new QSlider(this);
    overlay_lh_width->setMinimum(1);
    overlay_lh_width->setMaximum(160);
    overlay_lh_width->setToolTip(QString("Sets the width"));
    overlay_lh_width->setTickInterval(1);
    overlay_lh_width->setOrientation(Qt::Horizontal);

    overlay_lh_width_label = new QLabel(this);
    overlay_lh_width_label->setText("Left Width:");

    overlay_lh_width_spin = new QSpinBox(this);
    overlay_lh_width_spin->setMinimum(1);
    overlay_lh_width_spin->setMaximum(160);
    overlay_lh_width_spin->setButtonSymbols(QAbstractSpinBox::NoButtons);

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
    collections_layout->addWidget(&showConsoleLogBtn, 4, 2, 1, 1);
    if((!options.flightMode) && options.xioCam)
    {
        pausePlaybackChk.setText("Pause");
        pausePlaybackChk.setChecked(false);
        collections_layout->addWidget(&pausePlaybackChk, 2, 3, 1, 1);
        frameNumberLabel.setText("0");
        collections_layout->addWidget(&frameNumberLabel, 3, 3, 1, 1);
        collections_layout->addWidget(&showXioSetupBtn, 4, 3, 1, 1);
    }

    //Fifth Row:
    collections_layout->addWidget(overlay_lh_width_label, 5,1,1,1);
    collections_layout->addWidget(overlay_lh_width, 5,2,1,7);
    collections_layout->addWidget(overlay_lh_width_spin, 5,10,1,1);

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

    red_slider.setOrientation(Qt::Horizontal);
    red_slider.setMaximum(frHeight-1);
    red_slider.setMinimum(0);
    red_slider.setValue((frHeight-1)*0.166);
    red_slider.setTickInterval(1);
    red_slider.hide();

    green_slider.setOrientation(Qt::Horizontal);
    green_slider.setMaximum(frHeight-1);
    green_slider.setMinimum(0);
    green_slider.setValue((frHeight-1)*0.5);
    green_slider.setTickInterval(1);
    green_slider.hide();

    blue_slider.setOrientation(Qt::Horizontal);
    blue_slider.setMaximum(frHeight-1);
    blue_slider.setMinimum(0);
    blue_slider.setValue((frHeight-1)*0.833);
    blue_slider.setTickInterval(1);
    blue_slider.hide();

    redSpin.setMinimum(0);
    redSpin.setMaximum(frHeight-1);
    redSpin.setValue(red_slider.value());
    redSpin.setSingleStep(1);
    redSpin.hide();

    greenSpin.setMinimum(0);
    greenSpin.setMaximum(frHeight-1);
    greenSpin.setValue(green_slider.value());
    greenSpin.setSingleStep(1);
    greenSpin.hide();

    blueSpin.setMinimum(0);
    blueSpin.setMaximum(frHeight-1);
    blueSpin.setValue(blue_slider.value());
    blueSpin.setSingleStep(1);
    blueSpin.hide();

    QStringList rgbPresets;
    // default names are = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};

    for(int p=0; p < 10; p++)
    {
        rgbPresets.append(prefs.presetName[p]);
    }

    rgbPresets.append("Rename..."); // index = 10;

    rgbPresetCombo.addItems(rgbPresets);
    rgbPresetCombo.setToolTip("Recall an RGB preset here.\n To save a preset to the settings file, adjust the sliders and then press Save Preset.");
    rgbPresetCombo.hide();

    wflength_slider.setOrientation(Qt::Horizontal);
    wflength_slider.setMaximum(1024);
    wflength_slider.setMinimum(100);
    wflength_slider.setValue(500);
    wflength_slider.setTickInterval(1);
    wflength_slider.hide();

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

    show_rgb_lines_cbox.setText("Show RGB Lines");
    show_rgb_lines_cbox.setToolTip("Shows the RGB lines on the flight interface frame view\n at all times if checked. Otherwise just for 30 seconds");

    //center:
    overlay_cent_width = new QSlider(this);
    overlay_cent_width->setMinimum(1);
    overlay_cent_width->setMaximum(160);
    overlay_cent_width->setToolTip(QString("Sets the width"));
    overlay_cent_width->setTickInterval(1);
    overlay_cent_width->setOrientation(Qt::Horizontal);

    overlay_cent_width_label = new QLabel(this);
    overlay_cent_width_label->setText("Center Width:");

    overlay_cent_width_spin = new QSpinBox(this);
    overlay_cent_width_spin->setMinimum(1);
    overlay_cent_width_spin->setMaximum(160);
    overlay_cent_width_spin->setButtonSymbols(QAbstractSpinBox::NoButtons);

    //right:
    overlay_rh_width = new QSlider(this);
    overlay_rh_width->setMinimum(1);
    overlay_rh_width->setMaximum(160);
    overlay_rh_width->setToolTip(QString("Sets the width"));
    overlay_rh_width->setTickInterval(1);
    overlay_rh_width->setOrientation(Qt::Horizontal);

    overlay_rh_width_label = new QLabel(this);
    overlay_rh_width_label->setText("Right Width:");

    overlay_rh_width_spin = new QSpinBox(this);
    overlay_rh_width_spin->setMinimum(1);
    overlay_rh_width_spin->setMaximum(160);
    overlay_rh_width_spin->setButtonSymbols(QAbstractSpinBox::NoButtons);

    // off by default:

    overlay_lh_width->setVisible(false);
    overlay_lh_width->setEnabled(false);
    overlay_lh_width_label->setVisible(false);
    overlay_lh_width_label->setEnabled(false);
    overlay_lh_width_spin->setVisible(false);
    overlay_lh_width_spin->setEnabled(false);

    overlay_cent_width->setVisible(false);
    overlay_cent_width->setEnabled(false);
    overlay_cent_width_label->setVisible(false);
    overlay_cent_width_label->setEnabled(false);
    overlay_cent_width_spin->setVisible(false);
    overlay_cent_width_spin->setEnabled(false);

    overlay_rh_width->setVisible(false);
    overlay_rh_width->setEnabled(false);
    overlay_rh_width_label->setVisible(false);
    overlay_rh_width_label->setEnabled(false);
    overlay_rh_width_spin->setVisible(false);
    overlay_rh_width_spin->setEnabled(false);

    sliders_layout = new QGridLayout();

    //First Row
    sliders_layout->addWidget(std_dev_n_label, 1, 1, 1, 1);
    sliders_layout->addWidget(std_dev_N_slider, 1, 2, 1, 7);
    sliders_layout->addWidget(std_dev_N_edit, 1, 10, 1, 1);

    //Second Row
    sliders_layout->addWidget(&low_increment_cbox, 2, 1, 1, 1);
    sliders_layout->addWidget(&use_DSF_cbox, 2, 2, 1, 1);
    sliders_layout->addWidget(&show_rgb_lines_cbox, 2, 3, 1, 1);

    //Third Row
    sliders_layout->addWidget(new QLabel("Ceiling:"),3,1,1,1);
    sliders_layout->addWidget(&ceiling_slider,3,2,1,7);
    sliders_layout->addWidget(&ceiling_edit,3,10,1,1);

    //Fourth Row
    sliders_layout->addWidget(new QLabel("Floor:"), 4, 1, 1, 1);
    sliders_layout->addWidget(&floor_slider, 4, 2, 1, 7);
    sliders_layout->addWidget(&floor_edit, 4, 10, 1, 1);

    //Fifth Row:
    sliders_layout->addWidget(overlay_cent_width_label, 5,1,1,1);
    sliders_layout->addWidget(overlay_cent_width, 5,2,1,7);
    sliders_layout->addWidget(overlay_cent_width_spin, 5,10,1,1);

    // RGB waterfall support:
    red_label.setText("Red Band:");
    green_label.setText("Green Band:");
    blue_label.setText("Blue Band:");
    wflength_label.setText("WF Length:");

    // 6:
    sliders_layout->addWidget(&red_label, 6,1,1,1);
    sliders_layout->addWidget(&red_slider, 6,2,1,7);
    sliders_layout->addWidget(&redSpin, 6,10,1,1);

    // 7:
    sliders_layout->addWidget(&green_label, 7,1,1,1);
    sliders_layout->addWidget(&green_slider, 7,2,1,7);
    sliders_layout->addWidget(&greenSpin, 7,10,1,1);

    // 8:
    sliders_layout->addWidget(&blue_label, 8,1,1,1);
    sliders_layout->addWidget(&blue_slider, 8,2,1,7);
    sliders_layout->addWidget(&blueSpin, 8,10,1,1);

    // 9:
    sliders_layout->addWidget(&wflength_label, 9,1,1,1);
    sliders_layout->addWidget(&wflength_slider, 9,2,1,7);
    sliders_layout->addWidget(&rgbPresetCombo, 9,10,1,1);

    ThresholdingSlidersBox.setLayout(sliders_layout);

/* ====================================================================== */
    //RIGHT SIDE BUTTONS (Save)
    select_save_location.setText("Save Location");
    showConsoleLogBtn.setText("Log");
    showConsoleLogBtn.setToolTip("Show console log");

    showXioSetupBtn.setText("Xio Setup");
    showXioSetupBtn.setToolTip("Setup reading XIO files");

    save_finite_button.setText("Save Frames");

    start_saving_frames_button.setText("Start Saving");
    stop_saving_frames_button.setText("Stop Saving");
    stop_saving_frames_button.setEnabled(false);

    frames_save_num_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);
    frames_save_num_edit.setMinimum(0);
    frames_save_num_edit.setMaximum(10000000);
    
    frames_save_num_avgs_edit.setButtonSymbols(QAbstractSpinBox::NoButtons);
    frames_save_num_avgs_edit.setMinimum(1);
    frames_save_num_avgs_edit.setMaximum(10000000);

    debugButton.setText("Debug");
#ifndef QT_DEBUG
    // Hide the debug button on release builds:
    debugButton.hide();
#endif
    saveRGBPresetButton.setText("Save Preset");
    saveRGBPresetButton.setToolTip("Press to save the current RGB sliders to the selected preset slot.\nSlots may be named in the settings file.");
    saveRGBPresetButton.setVisible(false);

    diskSpaceBar.setToolTip("Disk Space Available\nas percent, at default data store location.");
    diskSpaceBar.setMaximum(100);
    diskSpaceBar.setMinimum(0);
    diskSpaceBar.setVisible(false);

    diskSpaceLabel.setText("Disk:");
    diskSpaceLabel.setVisible(false);

    save_layout = new QGridLayout();
    //First Row
    if(!options.flightMode)
    {
        // row, col, row span, col span
        save_layout->addWidget(&select_save_location, 1, 1, 1, 2);
    }
    save_layout->addWidget(new QLabel("Total #Frames / #Averaged:"), 2, 1, 1, 3);
    if(options.flightMode)
    {
        save_layout->addWidget(new QLabel("Data Location:"), 3, 1, 1, 1);
    } else {
        save_layout->addWidget(new QLabel("Filename:"), 3, 1, 1, 1);
    }

    //Second Row
    //save_layout.addWidget(&start_saving_frames_button, 1, 2, 1, 1);
    save_layout->addWidget(&save_finite_button,        1, 4, 1, 1);
    save_layout->addWidget(&frames_save_num_edit, 2, 4, 1, 1);
    save_layout->addWidget(&filename_edit, 3, 2, 1, 4);
    save_layout->addWidget(&debugButton, 4, 5, 1, 1);
    save_layout->addWidget(&saveRGBPresetButton, 4, 1, 1, 1);
    save_layout->addWidget(&diskSpaceLabel, 4, 2, 1, 1);
    save_layout->addWidget(&diskSpaceBar, 4, 3, 1, 2);
    if(options.flightMode)
    {
        filename_edit.setToolTip("Specified using --datastoragelocation option");
        if(options.dataLocationSet)
        {
            filename_edit.setText(options.dataLocation);
        } else {
            filename_edit.setText("undefined");
        }
        filename_edit.setEnabled(false);
    } else {
        filename_edit.setToolTip("Leave blank for automatic filename");
        filename_edit.setEnabled(true);
    }

    //Third Row
    save_layout->addWidget(&stop_saving_frames_button, 1, 5, 1, 1);
    save_layout->addWidget(&frames_save_num_avgs_edit, 2, 5, 1, 1);
    
    //Forth Row (overlay plot only)
    //To Do: Add lh_select_slider (4th row) and then place width in 5th row
    save_layout->addWidget(overlay_rh_width_label,4,1,1,1 );
    save_layout->addWidget(overlay_rh_width, 4,2,1,3);
    save_layout->addWidget(overlay_rh_width_spin, 4,5,1,1);

    SaveButtonsBox.setLayout(save_layout);

/* =========================================================================== */
    // OVERALL LAYOUT
    controls_layout.addWidget(&CollectionButtonsBox, 2);
    controls_layout.addWidget(&ThresholdingSlidersBox, 3);
    controls_layout.addWidget(&SaveButtonsBox, 2);
    this->setLayout(&controls_layout);
    this->setMaximumHeight(200);

/* =========================================================================== */
    //Connections
    connect(&collect_dark_frames_button, SIGNAL(clicked()), this, SLOT(start_dark_collection_slot()));
    connect(&stop_dark_collection_button, SIGNAL(clicked()), this, SLOT(stop_dark_collection_slot()));
    connect(&load_mask_from_file, SIGNAL(clicked()), this, SLOT(getMaskFile()));
    connect(&pref_button, SIGNAL(clicked()), this, SLOT(load_pref_window()));
    connect(&showConsoleLogBtn, &QPushButton::pressed,
            [&]() {
            emit showConsoleLog();
    });
    connect(&showXioSetupBtn, &QPushButton::pressed,
            [&]() {
            showSetup();
    });

    connect(&pausePlaybackChk, &QCheckBox::stateChanged, [this](int state) {
        if (static_cast<Qt::CheckState>(state) == Qt::CheckState::Checked) {
            emit setCameraPause(true);
        } else {
            emit setCameraPause(false);
        }
    });

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

    connect(&floor_slider, SIGNAL(valueChanged(int)), this, SLOT(updateFloor(int)));
    connect(&floor_edit, SIGNAL(valueChanged(int)), this, SLOT(updateFloor(int)));

    connect(&ceiling_slider, SIGNAL(valueChanged(int)), this, SLOT(updateCeiling(int)));
    connect(&ceiling_edit, SIGNAL(valueChanged(int)), this, SLOT(updateCeiling(int)));


//    connect(&ceiling_slider, &QSlider::valueChanged,
//            [&](int value) {
//            if(use_DSF_cbox.isChecked())
//            {
//                prefs.dsfCeiling = value;
//            } else {
//                prefs.frameViewCeiling = value;
//            }

//    });

//    connect(&floor_slider, &QSlider::valueChanged,
//            [&](int value) {
//            if(use_DSF_cbox.isChecked())
//            {
//                prefs.dsfFloor = value;
//            } else {
//                prefs.frameViewFloor = value;
//            }

//    });

//    connect(&use_DSF_cbox, &QCheckBox::toggled,
//            [&](bool toggled) {
//            if(toggled)
//            {
//                // DSF
//                floor_slider.setValue(prefs.dsfFloor);
//                ceiling_slider.setValue(prefs.dsfCeiling);
//            } else {
//                // Raw (no subtraction)
//                floor_slider.setValue(prefs.frameViewFloor);
//                ceiling_slider.setValue(prefs.frameViewCeiling);
//            }
//    });

    connect(&use_DSF_cbox, SIGNAL(toggled(bool)), this, SLOT(use_DSF_general(bool)));
    //connect(&show_rgb_lines_cbox, SIGNAL(toggled(bool)), p_flight, SLOT(setShowRGBLines(bool)));
    connect(&low_increment_cbox, SIGNAL(toggled(bool)), this, SLOT(increment_slot(bool)));
    connect(&save_finite_button, SIGNAL(clicked()), this, SLOT(save_finite_button_slot()));
    connect(&stop_saving_frames_button, SIGNAL(clicked()), this, SLOT(stop_continous_button_slot()));
    connect(&select_save_location, SIGNAL(clicked()), this, SLOT(show_save_dialog()));
    connect(&debugButton, SIGNAL(clicked()), this, SLOT(debugThis()));

    // Overlay:
    connect(overlay_lh_width_spin, SIGNAL(valueChanged(int)), overlay_lh_width, SLOT(setValue(int)));
    connect(overlay_lh_width, SIGNAL(valueChanged(int)), overlay_lh_width_spin, SLOT(setValue(int)));

    connect(overlay_cent_width_spin, SIGNAL(valueChanged(int)), overlay_cent_width, SLOT(setValue(int)));
    connect(overlay_cent_width, SIGNAL(valueChanged(int)), overlay_cent_width_spin, SLOT(setValue(int)));

    connect(overlay_rh_width_spin, SIGNAL(valueChanged(int)), overlay_rh_width, SLOT(setValue(int)));
    connect(overlay_rh_width, SIGNAL(valueChanged(int)), overlay_rh_width_spin, SLOT(setValue(int)));

    // Waterfall:
    connect(&red_slider, SIGNAL(valueChanged(int)), this, SLOT(setRGBWaterfall(int)));
    connect(&green_slider, SIGNAL(valueChanged(int)), this, SLOT(setRGBWaterfall(int)));
    connect(&blue_slider, SIGNAL(valueChanged(int)), this, SLOT(setRGBWaterfall(int)));

    connect(&red_slider, &QSlider::sliderMoved,
            [&](int value) {
        redSpin.blockSignals(true);
        redSpin.setValue(value);
        redSpin.blockSignals(false);
        bandRed = value;
        prefs.bandRed[rgbPresetCombo.currentIndex()] = value;
    });

    connect(&green_slider, &QSlider::sliderMoved,
            [&](int value) {
        greenSpin.blockSignals(true);
        greenSpin.setValue(value);
        greenSpin.blockSignals(false);
        bandGreen = value;
        prefs.bandGreen[rgbPresetCombo.currentIndex()] = value;
    });

    connect(&blue_slider, &QSlider::sliderMoved,
            [&](int value) {
        blueSpin.blockSignals(true);
        blueSpin.setValue(value);
        blueSpin.blockSignals(false);
        bandBlue = value;
        prefs.bandBlue[rgbPresetCombo.currentIndex()] = value;
    });

    connect(&redSpin,  QOverload<int>::of(&QSpinBox::valueChanged),
            [&](int value) {
        red_slider.setValue(value);
        bandRed = value;
        prefs.bandRed[rgbPresetCombo.currentIndex()] = value;
    });

    connect(&greenSpin,  QOverload<int>::of(&QSpinBox::valueChanged),
            [&](int value) {
        green_slider.setValue(value);
        bandGreen = value;
        prefs.bandGreen[rgbPresetCombo.currentIndex()] = value;
    });


    connect(&blueSpin,  QOverload<int>::of(&QSpinBox::valueChanged),
            [&](int value) {
        blue_slider.setValue(value);
        bandBlue = value;
        prefs.bandBlue[rgbPresetCombo.currentIndex()] = value;
    });

    connect(&rgbPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index) {
        if(index == 10)
        {
            bool ok;
            QString text = QInputDialog::getText(this, tr("QInputDialog::getText()"),
                                                 tr("Preset Name:"), QLineEdit::Normal,
                                                 rgbPresetCombo.itemText(previousRGBPresetIndex), &ok);
            if (ok && !text.isEmpty())
            {
                rgbPresetCombo.setItemText(previousRGBPresetIndex, text);
                prefs.presetName[previousRGBPresetIndex] = text;
            }
            rgbPresetCombo.setCurrentIndex(previousRGBPresetIndex);
            return;
        }
        if(index < 0 || index > 9)
            return;
        bandRed = prefs.bandRed[index];
        bandGreen = prefs.bandGreen[index];
        bandBlue = prefs.bandBlue[index];
        redSpin.setValue(bandRed);
        greenSpin.setValue(bandGreen);
        blueSpin.setValue(bandBlue);
        previousRGBPresetIndex = index;
    });

    connect(&saveRGBPresetButton, &QPushButton::pressed,
            [&]() {
        saveSingleRGBPreset(rgbPresetCombo.currentIndex(),
                            bandRed,
                            bandGreen,
                            bandBlue);
    });

    connect(&show_rgb_lines_cbox, &QCheckBox::toggled,
            [&](bool checked) {
        if(checked)
            emit updateRGB(bandRed, bandGreen, bandBlue);
    });



    // Preferences:
    connect(prefWindow, SIGNAL(saveSettings()), this, SLOT(triggerSaveSettings()));

    rgbPresetCombo.setCurrentIndex(0);
    bandRed = prefs.bandRed[0];
    bandGreen = prefs.bandGreen[0];
    bandBlue = prefs.bandBlue[0];
    redSpin.setValue(bandRed);
    greenSpin.setValue(bandGreen);
    blueSpin.setValue(bandBlue);
    previousRGBPresetIndex = 0;

    ceiling_edit.blockSignals(true);
    ceiling_slider.blockSignals(true);
    floor_edit.blockSignals(true);
    floor_slider.blockSignals(true);
    if(use_DSF_cbox.isChecked())
    {
        ceiling_slider.setValue(prefs.dsfCeiling);
        ceiling_edit.setValue(prefs.dsfCeiling);
        floor_slider.setValue(prefs.dsfFloor);
        floor_edit.setValue(prefs.dsfFloor);
    } else {
        ceiling_slider.setValue(prefs.frameViewCeiling);
        ceiling_edit.setValue(prefs.frameViewCeiling);
        floor_slider.setValue(prefs.frameViewFloor);
        floor_edit.setValue(prefs.frameViewFloor);
    }
    ceiling_edit.blockSignals(false);
    ceiling_slider.blockSignals(false);
    floor_edit.blockSignals(false);
    floor_slider.blockSignals(false);
}
void ControlsBox::closeEvent(QCloseEvent *e)
{
    /* Note: minor hack below */
    Q_UNUSED(e);
    prefWindow->close();
}

void ControlsBox::getPrefsExternalTrig()
{
    // This is called by the MainWindow after we have setup.
    emit haveReadPreferences(prefs);
}

void ControlsBox::loadSettings()
{
    prefs.readFile = false;

    // [Camera]:
    settings->beginGroup("Camera");
    prefs.brightSwap14 = settings->value("brightSwap14", defaultPrefs.brightSwap14).toBool();
    prefs.brightSwap16 = settings->value("brightSwap16", defaultPrefs.brightSwap16).toBool();
    prefs.nativeScale = settings->value("nativeScale", defaultPrefs.nativeScale).toBool();
    prefs.skipFirstRow = settings->value("skipFirstRow", defaultPrefs.skipFirstRow).toBool();
    prefs.skipLastRow = settings->value("skipLastRow", defaultPrefs.skipLastRow).toBool();
    prefs.use2sComp = settings->value("use2sComp", defaultPrefs.use2sComp).toBool();
    settings->endGroup();

    // [Interface]:
    settings->beginGroup("Interface");
    prefs.frameColorScheme = settings->value("frameColorScheme", defaultPrefs.frameColorScheme).toInt();
    prefs.useDarkTheme = settings->value("useDarkTheme", defaultPrefs.useDarkTheme).toBool();
    prefs.frameViewCeiling = settings->value("frameViewCeiling", defaultPrefs.frameViewCeiling).toInt();
    prefs.frameViewFloor = settings->value("frameViewFloor", defaultPrefs.frameViewFloor).toInt();
    prefs.dsfCeiling = settings->value("dsfCeiling", defaultPrefs.dsfCeiling).toInt();
    prefs.dsfFloor = settings->value("dsfFloor", defaultPrefs.dsfFloor).toInt();
    prefs.fftCeiling = settings->value("fftCeiling", defaultPrefs.fftCeiling).toInt();
    prefs.fftFloor = settings->value("fftFloor", defaultPrefs.fftFloor).toInt();
    prefs.stddevCeiling = settings->value("stddevCeiling", defaultPrefs.stddevCeiling).toInt();
    prefs.stddevFloor = settings->value("stddevFloor", defaultPrefs.stddevFloor).toInt();
    prefs.preferredWindowWidth = settings->value("preferredWindowWidth", defaultPrefs.preferredWindowWidth).toInt();
    prefs.preferredWindowHeight = settings->value("preferredWindowHeight", defaultPrefs.preferredWindowHeight).toInt();
    settings->endGroup();

    // [RGB]:
    settings->beginGroup("RGB");
    int rgbArraySize = settings->beginReadArray("bandsRGB");
    if(rgbArraySize > 10)
        rgbArraySize = 10;


    for(int i=0; i < rgbArraySize; i++)
    {
        settings->setArrayIndex(i);
        prefs.bandRed[i] = settings->value("bandRed", defaultPrefs.bandRed[i]).toInt();
        prefs.bandGreen[i] = settings->value("bandGreen", defaultPrefs.bandGreen[i]).toInt();
        prefs.bandBlue[i] = settings->value("bandBlue", defaultPrefs.bandBlue[i]).toInt();
        prefs.presetName[i] = settings->value("bandName", QString("%1").arg(i+1)).toString();
    }

    settings->endArray();
    settings->endGroup();

    // [Flight]:
    settings->beginGroup("Flight");
    prefs.hidePlayback = settings->value("hidePlayback", defaultPrefs.hidePlayback).toBool();
    prefs.hideFFT = settings->value("hideFFT", defaultPrefs.hideFFT).toBool();
    prefs.hideVerticalOverlay = settings->value("hideVerticalOverlay", defaultPrefs.hideVerticalOverlay).toBool();

    prefs.hideVertMeanProfile = settings->value("hideVertMeanProfile", defaultPrefs.hideVertMeanProfile).toBool();
    prefs.hideVertCrosshairProfile = settings->value("hideVertCrosshairProfile", defaultPrefs.hideVertCrosshairProfile).toBool();
    prefs.hideHorizontalMeanProfile = settings->value("hideHorizontalMeanProfile", defaultPrefs.hideHorizontalMeanProfile).toBool();
    prefs.hideHorizontalCrosshairProfile = settings->value("hideHorizontalCrosshairProfile", defaultPrefs.hideHorizontalCrosshairProfile).toBool();
    prefs.hideHistogramView = settings->value("hideHistogramView", defaultPrefs.hideHistogramView).toBool();
    prefs.hideStddeviation = settings->value("hideStddeviation", defaultPrefs.hideStddeviation).toBool();
    prefs.hideWaterfallTab = settings->value("hideWaterfallTab", defaultPrefs.hideWaterfallTab).toBool();
    prefs.percentDiskWarning = settings->value("percentDiskWarning", defaultPrefs.percentDiskWarning).toInt();
    prefs.percentDiskStop = settings->value("percentDiskStop", defaultPrefs.percentDiskStop).toInt();
    settings->endGroup();

    prefs.readFile = true;
    updateUIToPrefs();
    emit statusMessage(QString("[Controls Box]: 2s compliment setting from preferences: %1").arg(prefs.use2sComp?"Enabled":"Disabled"));
    emit haveReadPreferences(prefs);
}

void ControlsBox::updateUIToPrefs()
{
    current_tab = qtw->widget(qtw->currentIndex());
    attempt_pointers(current_tab);

    if(p_profile || p_frameview || p_flight)
    {
        if (p_frameview && p_frameview->image_type == STD_DEV) {
            // Type is Standard Deviation Image
            floor_slider.setValue(prefs.stddevFloor);
            ceiling_slider.setValue(prefs.stddevCeiling);
            return;
        }

        // profile plots and frame view images generally
        // use the same levels.
        // So therefore, we just have a DSF and not-DSF level for each.
        if(use_DSF_cbox.isChecked())
        {
            floor_slider.setValue(prefs.dsfFloor);
            ceiling_slider.setValue(prefs.dsfCeiling);
        } else {
            floor_slider.setValue(prefs.frameViewFloor);
            ceiling_slider.setValue(prefs.frameViewCeiling);
        }
        return;
    }

    if(p_fft)
    {
        floor_slider.setValue(prefs.fftFloor);
        ceiling_slider.setValue(prefs.fftCeiling);
        return;
    }
    if(p_histogram)
    {
        floor_slider.setValue(prefs.stddevFloor);
        ceiling_slider.setValue(prefs.stddevCeiling);
        return;
    }
}

void ControlsBox::triggerSaveSettings()
{
    // Copy settings in from the preference window
    settingsT pwprefs = prefWindow->getPrefs();
    prefs.brightSwap14 = pwprefs.brightSwap14;
    prefs.brightSwap16 = pwprefs.brightSwap16;
    prefs.frameColorScheme = pwprefs.frameColorScheme;
    prefs.use2sComp = pwprefs.use2sComp;
    prefs.useDarkTheme = pwprefs.useDarkTheme;

    // Now save:
    saveSettings();
}

void ControlsBox::saveSettings()
{
    // [Camera]:
    settings->beginGroup("Camera");
    settings->setValue("brightSwap14", prefs.brightSwap14 );
    settings->setValue("brightSwap16", prefs.brightSwap16);
    settings->setValue("nativeScale", prefs.nativeScale);
    settings->setValue("skipFirstRow", prefs.skipFirstRow);
    settings->setValue("skipLastRow", prefs.skipLastRow);
    settings->setValue("use2sComp", prefs.use2sComp);
    settings->endGroup();

    // [Interface]:
    settings->beginGroup("Interface");
    settings->setValue("frameColorScheme", prefs.frameColorScheme);
    settings->setValue("useDarkTheme", prefs.useDarkTheme);
    settings->setValue("frameViewCeiling", prefs.frameViewCeiling);
    settings->setValue("frameViewFloor", prefs.frameViewFloor);
    settings->setValue("dsfCeiling", prefs.dsfCeiling);
    settings->setValue("dsfFloor", prefs.dsfFloor);
    settings->setValue("fftCeiling", prefs.fftCeiling);
    settings->setValue("fftFloor", prefs.fftFloor);
    settings->setValue("stddevCeiling", prefs.stddevCeiling);
    settings->setValue("stddevFloor", prefs.stddevFloor);
    settings->setValue("preferredWindowWidth", prefs.preferredWindowWidth);
    settings->setValue("preferredWindowHeight", prefs.preferredWindowHeight);

    settings->endGroup();

    // [RGB]:
    settings->beginGroup("RGB");
    settings->beginWriteArray("bandsRGB", 10);
    for(int i=0; i < 10; i++)
    {
        settings->setArrayIndex(i);
        settings->setValue("bandRed", prefs.bandRed[i]);
        settings->setValue("bandGreen", prefs.bandGreen[i]);
        settings->setValue("bandBlue", prefs.bandBlue[i]);
        settings->setValue("bandName", rgbPresetCombo.itemText(i));
    }
    settings->endArray();
    settings->endGroup();

    // [Flight]:
    settings->beginGroup("Flight");
    settings->setValue("hidePlayback", prefs.hidePlayback);
    settings->setValue("hideFFT", prefs.hideFFT);
    settings->setValue("hideVerticalOverlay", prefs.hideVerticalOverlay);
    settings->setValue("hideVertMeanProfile", prefs.hideVertMeanProfile);
    settings->setValue("hideVertCrosshairProfile", prefs.hideVertCrosshairProfile);
    settings->setValue("hideHorizontalMeanProfile", prefs.hideHorizontalMeanProfile);
    settings->setValue("hideHorizontalCrosshairProfile", prefs.hideHorizontalCrosshairProfile);
    settings->setValue("hideHistogramView", prefs.hideHistogramView);
    settings->setValue("hideStddeviation", prefs.hideStddeviation);
    settings->setValue("hideWaterfallTab", prefs.hideWaterfallTab);
    settings->setValue("percentDiskWarning", prefs.percentDiskWarning);
    settings->setValue("percentDiskStop", prefs.percentDiskStop);
    settings->endGroup();

    settings->sync();
    emit statusMessage("[Controls Box]: Saved settings.");
}

void ControlsBox::saveSingleRGBPreset(int index, int r, int g, int b)
{
    // [RGB]:
    settings->beginGroup("RGB");
    settings->beginWriteArray("bandsRGB", 10);

    settings->setArrayIndex(index);
    settings->setValue("bandRed", prefs.bandRed[index]);
    settings->setValue("bandGreen", prefs.bandGreen[index]);
    settings->setValue("bandBlue", prefs.bandBlue[index]);
    settings->setValue("bandName", rgbPresetCombo.itemText(index));

    settings->endArray();
    settings->endGroup();
    emit statusMessage(QString("[Controls Box]: Saved RGB setting to index %1 named %2.").arg(index).arg(rgbPresetCombo.itemText(index)));
}

void ControlsBox::setDefaultSettings()
{
    // [Camera]:
    defaultPrefs.brightSwap14 = false;
    defaultPrefs.brightSwap16 = false;
    defaultPrefs.nativeScale = true;
    defaultPrefs.skipFirstRow = true;
    defaultPrefs.skipLastRow = false;
    defaultPrefs.use2sComp = false;

    // [Interface]:
    defaultPrefs.frameColorScheme = 0; // Jet
    defaultPrefs.useDarkTheme = false;

    defaultPrefs.frameViewCeiling = 65000;
    defaultPrefs.frameViewFloor = 10000;
    defaultPrefs.dsfCeiling = 3000;
    defaultPrefs.dsfFloor = -2000;
    defaultPrefs.fftCeiling = 1000;
    defaultPrefs.fftFloor = 0;
    defaultPrefs.stddevCeiling = 1000;
    defaultPrefs.stddevFloor = 0;
    defaultPrefs.preferredWindowWidth = 1280;
    defaultPrefs.preferredWindowHeight = 1024;


    // [RGB]:
    for(int i=0; i < 10; i++)
    {
        defaultPrefs.bandRed[i] = 100;
        defaultPrefs.bandGreen[i] = 200;
        defaultPrefs.bandBlue[i] = 300;
        defaultPrefs.presetName[i] = QString("%1").arg(i+1);
    }

    // [Flight]:
    defaultPrefs.hidePlayback = true; // currently always hidden due to issues with the playback widget.
    defaultPrefs.hideFFT = true;
    defaultPrefs.hideVerticalOverlay = true;

    defaultPrefs.hideVertMeanProfile = false;
    defaultPrefs.hideVertCrosshairProfile = false;
    defaultPrefs.hideHorizontalMeanProfile = false;
    defaultPrefs.hideHorizontalCrosshairProfile = false;
    defaultPrefs.hideHistogramView = false;
    defaultPrefs.hideStddeviation = false;
    defaultPrefs.hideWaterfallTab = false;

    defaultPrefs.percentDiskWarning = 85;
    defaultPrefs.percentDiskStop = 99;
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
     * \author Jackie Ryan
     */

    bool see_it = false;
    current_tab = qtw->widget(index);
    disconnect_old_tab();
    attempt_pointers(current_tab);
    show_rgb_lines_cbox.setEnabled(false);
    show_rgb_lines_cbox.setVisible(false);
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
        startVal = (p_profile->itype == VERTICAL_CROSS || p_profile->itype == VERT_OVERLAY) ? fw->horizLinesAvgd : startVal;
        enable = (p_profile->itype == VERTICAL_CROSS || p_profile->itype == HORIZONTAL_CROSS || p_profile->itype == VERT_OVERLAY) && fw->crosshair_x != -1;
        lines_slider->setMaximum(frameMax);
        line_average_edit->setMaximum(frameMax);
        lines_slider->setValue(startVal);
        fw->updateMeanRange(lines_slider->value(), p_profile->itype);
        lines_slider->setEnabled(enable);
        lines_slider->setVisible(enable);
        line_average_edit->setEnabled(enable);
        line_average_edit->setVisible(enable);
        display_lines_slider();

        if(p_profile->itype == VERT_OVERLAY)
        {
            // vertical overlay has three additional sliders for setting the width.
            overlayControls(true);
            lines_slider->setEnabled(true);
            lines_slider->setVisible(true);
            lines_label->setVisible(true);
            lines_label->setText("L-R Span:");

            this->setMaximumHeight(175);

            lines_slider->setEnabled(true);

            connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_profile->overlay_img, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), p_profile->overlay_img, SLOT(updateFloor(int)));

            connect(overlay_lh_width, SIGNAL(valueChanged(int)), this, SLOT(updateOverlayParams(int)));
            connect(overlay_cent_width, SIGNAL(valueChanged(int)), this, SLOT(updateOverlayParams(int)));
            connect(overlay_rh_width, SIGNAL(valueChanged(int)), this, SLOT(updateOverlayParams(int)));

        } else {
            overlayControls(false);
            lines_label->setText("Lines to Average:");


            this->setMaximumHeight(150);


        }
        waterfallControls(false);
        p_profile->rescaleRange();
    } else if (p_fft) {
        overlayControls(false);
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
        p_fft->updateCeiling(prefs.fftCeiling);
        p_fft->updateFloor(prefs.fftFloor);

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
        waterfallControls(false);
    } else {
        display_std_dev_slider();
        if (p_frameview){
            ceiling_maximum = p_frameview->slider_max;
            floor_slider.blockSignals(true);
            ceiling_slider.blockSignals(true);
            floor_edit.blockSignals(true);
            ceiling_edit.blockSignals(true);
            low_increment_cbox.setChecked(p_frameview->slider_low_inc);

            increment_slot(low_increment_cbox.isChecked());

            if(p_frameview->image_type != STD_DEV)
            {
                use_DSF_cbox.setEnabled(true);
                if(use_DSF_cbox.isChecked())
                {
                    ceiling_edit.setValue(prefs.dsfCeiling);
                    ceiling_slider.setValue(prefs.dsfCeiling);
                    floor_edit.setValue(prefs.dsfFloor);
                    floor_slider.setValue(prefs.dsfFloor);
                    p_frameview->updateCeiling(prefs.dsfCeiling);
                    p_frameview->updateFloor(prefs.dsfFloor);
                } else {
                    ceiling_edit.setValue(prefs.frameViewCeiling);
                    ceiling_slider.setValue(prefs.frameViewCeiling);
                    floor_edit.setValue(prefs.frameViewFloor);
                    floor_slider.setValue(prefs.frameViewFloor);
                    p_frameview->updateCeiling(prefs.frameViewCeiling);
                    p_frameview->updateFloor(prefs.frameViewFloor);
                }
            } else {
                use_DSF_cbox.setEnabled(false);
            }
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_frameview, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), p_frameview, SLOT(updateFloor(int)));
            connect(&use_DSF_cbox, SIGNAL(clicked(bool)), p_frameview, SLOT(setUseDSF(bool)));
            // TODO: Maybe a careful copy is smarter than a live pointer?
            // The live pointer can be changed in real time from either side...
            p_frameview->setPrefsPtr(&this->prefs);
            if (p_frameview->image_type == STD_DEV) {
                p_frameview->updateCeiling(prefs.stddevCeiling);
                p_frameview->updateFloor(prefs.stddevFloor);
                std_dev_N_slider->setEnabled(true);
                std_dev_N_edit->setEnabled(true);
                low_increment_cbox.setChecked(true);
                floor_slider.setValue(prefs.stddevFloor);
                floor_edit.setValue(prefs.stddevFloor);
                ceiling_slider.setValue(prefs.stddevCeiling);
                ceiling_edit.setValue(prefs.stddevCeiling);
                use_DSF_cbox.setEnabled(false);
            } else {
                std_dev_N_slider->setEnabled(false);
                std_dev_N_edit->setEnabled(false);
            }
            use_DSF_cbox.setChecked(fw->usingDSF());
            fw->setCrosshairBackend(fw->crosshair_x, fw->crosshair_y);
            p_frameview->rescaleRange();
            waterfallControls(false);

            floor_edit.blockSignals(false);
            ceiling_edit.blockSignals(false);
            floor_slider.blockSignals(false);
            ceiling_slider.blockSignals(false);

        } else if (p_flight) {
            ceiling_maximum = p_flight->slider_max;
            low_increment_cbox.setChecked(p_flight->slider_low_inc);
            increment_slot(low_increment_cbox.isChecked());
            //ceiling_edit.setValue(p_flight->getCeiling());
            //floor_edit.setValue(p_flight->getFloor());
            if(use_DSF_cbox.isChecked())
            {
                // DSF
                ceiling_edit.setValue(prefs.dsfCeiling);
                floor_edit.setValue(prefs.dsfFloor);
                p_flight->updateFloor(prefs.dsfFloor);
                p_flight->updateCeiling(prefs.dsfCeiling);
            } else {
                ceiling_edit.setValue(prefs.frameViewCeiling);
                floor_edit.setValue(prefs.frameViewFloor);
                p_flight->updateFloor(prefs.frameViewFloor);
                p_flight->updateCeiling(prefs.frameViewCeiling);
            }

            waterfallControls(true);

            connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_flight, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), p_flight, SLOT(updateFloor(int)));
            connect(this, SIGNAL(updateRGB(int,int,int)), p_flight, SLOT(changeRGB(int,int,int)));
            connect(&wflength_slider, SIGNAL(valueChanged(int)), p_flight, SLOT(changeWFLength(int)));
            connect(fw, SIGNAL(updateFPS()), p_flight, SLOT(updateFPS()));
            connect(&use_DSF_cbox, SIGNAL(clicked(bool)), p_flight, SLOT(setUseDSF(bool)));
            connect(&show_rgb_lines_cbox, SIGNAL(toggled(bool)), p_flight, SLOT(setShowRGBLines(bool)));

            std_dev_n_label->hide();
            std_dev_N_slider->setEnabled(false);
            std_dev_N_edit->setEnabled(false);
            std_dev_N_slider->hide();
            std_dev_N_edit->hide();



            use_DSF_cbox.setEnabled(true);
            use_DSF_cbox.setChecked(false);
            show_rgb_lines_cbox.setEnabled(true);
            show_rgb_lines_cbox.setVisible(true);
            fw->setCrosshairBackend(fw->crosshair_x, fw->crosshair_y);
            p_flight->rescaleRange();

            //p_flight->updateFloor(floor_slider.value());
            //p_flight->updateCeiling(ceiling_slider.value());

            p_flight->changeWFLength(wflength_slider.value());
            this->setMaximumHeight(230);
        } else if (p_histogram) {
            ceiling_maximum = p_histogram->slider_max;
            low_increment_cbox.setChecked(p_histogram->slider_low_inc);
            increment_slot(low_increment_cbox.isChecked());

            // TODO: prefs.std...
            ceiling_edit.setValue(p_histogram->getCeiling());
            floor_edit.setValue(p_histogram->getFloor());
            connect(&ceiling_slider, SIGNAL(valueChanged(int)), p_histogram, SLOT(updateCeiling(int)));
            connect(&floor_slider, SIGNAL(valueChanged(int)), p_histogram, SLOT(updateFloor(int)));
            std_dev_N_slider->setEnabled(true);
            std_dev_N_edit->setEnabled(true);
            use_DSF_cbox.setEnabled(false);
            use_DSF_cbox.setChecked(fw->usingDSF());
            p_histogram->rescaleRange();
            waterfallControls(false);
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
            waterfallControls(false);
        }
    }
}

// private slots

void ControlsBox::updateFloor(int f)
{
    setLevelToPrefs(false, f);
}

void ControlsBox::updateCeiling(int c)
{
    setLevelToPrefs(true, c);
}

void ControlsBox::updateDiskSpace(quint64 total, quint64 available)
{
    float percent = (100.0*(total-available) / total);
    diskSpaceBar.setValue((int)percent);
    unsigned int frameSpaceBytes = 2 * frWidth * frHeight;
    float BytesPerSecond = frameSpaceBytes * fps_float;
    float hoursRemaining = 0.0;

    if(BytesPerSecond > 1024)
    {
        hoursRemaining = (available / BytesPerSecond) / 60.0 / 60.0;
        diskSpaceBar.setToolTip(QString("Available: %1 GiB\nHours: %2").arg(available / 1024.0 / 1024.0 / 1024.0, 6, 'f', 1, QChar('0'))
                                .arg(hoursRemaining, 4, 'f', 3, QChar('0')));
    } else {
        diskSpaceBar.setToolTip(QString("Available: %1 GiB\nHours: %2").arg(available / 1024.0 / 1024.0 / 1024.0, 6, 'f', 1, QChar('0'))
                                .arg("Unknown"));
    }

    if(percent > prefs.percentDiskStop)
    {
        if((diskErrorCounter++ % 10) == 0)
            emit statusMessage(QString("ERROR: Disk nearly full at %1 percent used. Stopping data recording.").arg(percent));
        this->stop_continous_button_slot();
    } else if (percent > prefs.percentDiskWarning) {
        if((diskWarningCounter++ % 10) == 0)
            emit statusMessage(QString("WARMING: Disk low, usage %1 percent.").arg(percent));
    } else {
        diskErrorCounter = 0;
        diskWarningCounter = 0;
    }
}

void ControlsBox::showSetup()
{
    // Pause this thread while open
    setupUI.acceptOptions(&options);
    int result = setupUI.exec();

    if(result)
    {
        // At this point, options have changed.
        // so we need to communicate this to fw.
        if(options.xioDirectoryArray != NULL)
            fw->useNewOptions(options);
        else
            abort();
    }
}

void ControlsBox::setLevelToPrefs(bool isCeiling, int val)
{
    // This function takes a level (int val) and
    // writes that value to the "prefs" structure.
    // It is called whenever a slider is moved
    // within the controls box.

    current_tab = qtw->widget(qtw->currentIndex());
    attempt_pointers(current_tab);
    if(p_profile || p_frameview || p_flight)
    {
        if (p_frameview && p_frameview->image_type == STD_DEV) {
            if(isCeiling)
                prefs.stddevCeiling = val;
            else
                prefs.stddevFloor = val;
            return;
        }

        // profile plots and frame view images generally
        // use the same levels.
        // So therefore, we just have a DSF and not-DSF level for each.
        if(use_DSF_cbox.isChecked())
        {
            if(isCeiling)
            {
                prefs.dsfCeiling = val;
                //qDebug() << "Setting DSF Ceiling to : " << val;
            }
            else
            {
                prefs.dsfFloor = val;
                //qDebug() << "Setting DSF Floor to : " << val;
            }
        } else {
            if(isCeiling)
                prefs.frameViewCeiling = val;
            else
                prefs.frameViewFloor = val;
        }
        return;
    }

    if(p_fft)
    {
        if(isCeiling)
            prefs.fftCeiling = val;
        else
            prefs.fftFloor = val;
        return;
    }
    if(p_histogram)
    {
        if(isCeiling)
            prefs.stddevCeiling = val;
        else
            prefs.stddevFloor = val;
        return;
    }
}

void ControlsBox::increment_slot(bool t)
{
    /*! \brief Handles the Precision Slider range adjustment
     * If the slider is set to increment low, we set the tick and step to 1, the max ceiling to 2000 (or whatever the values
     * of LIL_TICK and LIL_MAX happen to be. If the increment is set high, the max is set to BIG_MAX and the slider will scroll
     * through at the rate of BIG_TICK, which is 400. We then attempt to cast a pointer to the currently displayed widget, and
     * then adjust its copy of the variable slider_low_inc, which holds whether or not the precision slider mode is active.
     * \author Noah Levy
     */
    //qDebug() << "-  increment_slot: start std floor: " << prefs.stddevFloor << "ceiling: " << prefs.stddevCeiling;
    //qDebug() << " - increment_slot: floor slideR: " << floor_slider.value() << "ceiling slider: " << ceiling_slider.value();
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
    else if (p_flight)
        p_flight->slider_low_inc = t;
    else if (p_histogram)
        p_histogram->slider_low_inc = t;
    else if (p_profile)
        p_profile->slider_low_inc = t;
    else if (p_fft)
        p_fft->slider_low_inc = t;
    else if (p_playback)
        p_playback->slider_low_inc = t;

    //qDebug() << " - increment_slot: floor slideR: " << floor_slider.value() << "ceiling slider: " << ceiling_slider.value();
    //qDebug() << "-  increment_slot: end std floor: " << prefs.stddevFloor << "ceiling: " << prefs.stddevCeiling;

}
void ControlsBox::attempt_pointers(QWidget *tab)
{
    p_frameview = qobject_cast<frameview_widget*>(tab);
    p_flight = qobject_cast<flight_widget*>(tab);
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
    } else if (p_flight)
    {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_flight,SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_flight, SLOT(updateFloor(int)));
    } else if (p_histogram) {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_histogram,SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_histogram, SLOT(updateFloor(int)));
    } else if (p_profile) {
        disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_profile,SLOT(updateCeiling(int)));
        disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_profile, SLOT(updateFloor(int)));
        if(p_profile->itype == VERT_OVERLAY)
        {            
            disconnect(&ceiling_slider, SIGNAL(valueChanged(int)), p_profile->overlay_img, SLOT(updateCeiling(int)));
            disconnect(&floor_slider, SIGNAL(valueChanged(int)), p_profile->overlay_img, SLOT(updateFloor(int)));
            disconnect(overlay_lh_width, SIGNAL(valueChanged(int)), this, SLOT(updateOverlayParams(int)));
            disconnect(overlay_cent_width, SIGNAL(valueChanged(int)), this, SLOT(updateOverlayParams(int)));
            disconnect(overlay_rh_width, SIGNAL(valueChanged(int)), this, SLOT(updateOverlayParams(int)));

        }
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
    fps_float = fw->delta;
    fps = QString::number(fps_float, 'f', 1).rightJustified(6, ' ');
    fps_label.setText(QString("FPS @ backend:%1").arg(fps));
}
void ControlsBox::setFrameNumber(int number)
{
    QString frameNumberStr = QString::number(number).rightJustified(6, ' ');
    frameNumberLabel.setText(QString("Frame:%1").arg(frameNumberStr));
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
void ControlsBox::save_remote_slot(const QString &unverifiedName, unsigned int nFrames, unsigned int numAvgs)
{
    checkForOverwrites = false;
    filename_edit.setText(unverifiedName);
    frames_save_num_edit.setValue(nFrames);
    frames_save_num_avgs_edit.setValue(numAvgs);
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
    // TODO: for "flight mode", skip validation and auto-generate.
    // Store auto-gen basemane in local varaibles.
    // baseDirectory
    // baseName;
    // imageExtension
    // flightGPSlogExtension (secondary log file)
    // Master log file generated on start automatically? hmm

    if(options.flightMode)
    {
        // Generate filenames:

        fnamegen.generate(); // new timestamp
        QString rawDataFilename = fnamegen.getFullFilename("", "-scenedata", "raw");
        QString gpsLogFilename = fnamegen.getFullFilename("", "-scenegps", "bin");

        // Populate the text boxes with the filenames

        // If the number of frames to save is blank, continuous recording happens.
        // No averaging.
        // We might want to hard-code a zero for the number of frames to save,
        // which is a flag to continuously save. Or, we can keep it as-is
        // and allow the operator to specify a number of frames.
        emit startSavingFinite(frames_save_num_edit.value(), rawDataFilename, 1);
        emit statusMessage(QString("[Controls Box]: Saving data to file [%1]").arg(rawDataFilename));
        previousNumSaved = frames_save_num_edit.value();
        stop_saving_frames_button.setEnabled(true);
        start_saving_frames_button.setEnabled(false);
        save_finite_button.setEnabled(false);
        frames_save_num_edit.setEnabled(false);
        frames_save_num_avgs_edit.setEnabled(false);

        // This is a signal to start saving the secondary GPS log.
        emit startDataCollection(gpsLogFilename);
    } else {


        if (validateFileName(filename_edit.text()) == QDialog::Accepted) {
            if(frames_save_num_edit.value() == 0)
            {
                // Continuous Recording Mode
            } else if(frames_save_num_edit.value() < frames_save_num_avgs_edit.value()) {
                frames_save_num_edit.setValue(frames_save_num_avgs_edit.value());
            }
            emit startSavingFinite(frames_save_num_edit.value(), filename_edit.text(), frames_save_num_avgs_edit.value());
            emit statusMessage(QString("[Controls Box]: Saving data to file [%1]").arg(filename_edit.text()));
            previousNumSaved = frames_save_num_edit.value();
            stop_saving_frames_button.setEnabled(true);
            start_saving_frames_button.setEnabled(false);
            save_finite_button.setEnabled(false);
            frames_save_num_edit.setEnabled(false);
            frames_save_num_avgs_edit.setEnabled(false);
            // TODO: Filename generation for flight mode
            emit startDataCollection(filename_edit.text().append("-GPS-TEMP-SECONDARY.log"));
        }
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
    frames_save_num_avgs_edit.setEnabled(true);
    frames_save_num_edit.setValue(previousNumSaved);
    emit statusMessage(QString("[Controls Box]: Stopping save data."));
    emit stopDataCollection();
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
        frames_save_num_avgs_edit.setEnabled(true);
        frames_save_num_edit.setEnabled(true);
        frames_save_num_edit.setValue(previousNumSaved);
    } else {
        frames_save_num_edit.setValue(n);
    }
}
int ControlsBox::validateFileName(const QString &name)
{
    // TODO: Supply meaningful result from this function
    // And accept the result into the calling function, whatever that is
    int result = QDialog::Accepted;

    // No filename
    if (name.isEmpty()) {
        result = QMessageBox::critical(this, "Frame Save Error", \
                              tr("Filename is empty.\nPlease select a valid location to save data."), \
                              QMessageBox::Cancel);
        return -1;
    }

    // Characters
    QString qs;
    for (const char *c = notAllowedChars; *c; c++) {
        if (name.contains(QLatin1Char(*c)))
            qs.append(QLatin1Char(*c));
    }
    if (!qs.isEmpty()){
        result = QMessageBox::critical(this, "Frame Save Error", \
                                       tr("Invalid character(s) \"%1\" in file name.").arg(qs), \
                              QMessageBox::Cancel);
        return -1;
    }

    // Starts with slash
    if (name.at(0) != '/') {
        result = QMessageBox::critical(this, "Frame Save Error", \
                              tr("File name does not specify a valid path.\nPlease include the path to the folder in which to save data."), \
                              QMessageBox::Cancel);
        return -1;
    }

    // File exists
    QFileInfo checkFile(name);
    if (checkFile.exists() && checkFile.isFile() && checkForOverwrites)
    {
        result = (quint16) QMessageBox::warning(this, "Frame Save Warning", \
                             tr("File name already exists.\nOverwrite it?"), \
                             QMessageBox::Ok, QMessageBox::Cancel);
        // seems to return 1024 (2^10) if accepted and 4194304 (2^22) if rejected. Hmm... Are we casting incorrectly? Seems like a 32 bit span.
        // "OK" aka "Accepted" should return a 1, and cancel should return a 0.
        // HORRIBLE hack: (added casting to quint16 though)
        if(result == 1024)
            result = 1;
    }
    return result;
}
void ControlsBox::start_dark_collection_slot()
{ /*! \brief Begins recording dark frames in the backend
   *  \author Jackie Ryan
   */
    collect_dark_frames_button.setEnabled(false);
    stop_dark_collection_button.setEnabled(true);
    emit statusMessage(QString("[Controls Box]: Collecting dark frames."));
    emit startDSFMaskCollection();
}
void ControlsBox::stop_dark_collection_slot()
{
    /*! \brief Stops recording dark frames in the backend
     *  \author Jackie Ryan
     */
    emit stopDSFMaskCollection();
    collect_dark_frames_button.setEnabled(true);
    stop_dark_collection_button.setEnabled(false);
    emit statusMessage(QString("[Controls Box]: Stopped collecting dark frames."));
}
void ControlsBox::getMaskFile()
{
    /*! \brief Load a file containing Dark Frames, and create a mask from the specified region of the file.
     * First opens a file dialog, verifies the file, then opens a QDialog which allows the user to select a range
     * of frames which contain dark that will then be averaged into a mask. A widget which uses this signal must be
     * able to receive the signal maskSelected(QString filename, unsigned int elem_to_read, long offset)
     * \author Jackie Ryan
     */
    /* Step 1: Open a dialog to select the mask file location */
    QFileDialog location_dialog(0);
    location_dialog.setFilter(QDir::Writable | QDir::Files);
    QString fileName = location_dialog.getOpenFileName(this, tr("Select mask file"),"",tr("Files (*.raw)"));
    if (fileName.isEmpty())
        return;

    /* Step 2: Calculate the number of frames */
    FILE *mask_file;
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
     * \author Jackie Ryan
     */
    if (p_playback)
        p_playback->toggleUseDSF(checked);
    else
    {
        fw->toggleUseDSF(checked);
        if(p_frameview)
        {
            p_frameview->setUseDSF(checked);
        }
    }
}
void ControlsBox::load_pref_window()
{
    /*! \brief Displays the preference window one the screen
     *  \author Jackie Ryan
     */
    QPoint pos = prefWindow->pos();
    if (pos.x() < 0)
        pos.setX(0);
    if (pos.y() < 0)
        pos.setY(0);
    prefWindow->move(pos);
    prefWindow->show();
    prefWindow->setWindowState(Qt::WindowActive);
    prefWindow->raise();
}
void ControlsBox::transmitChange(int linesToAverage)
{
    volatile int lh_start, lh_end, cent_start, cent_end, rh_start, rh_end;
    volatile int lh_width = 20;
    volatile int cent_width = 20;
    volatile int rh_width = 20;

    if (p_profile) {

        // Unfortunately, updateMeanRange also updates the crosshair span
        // so it is not possible to update the mean range in take object
        // without also altering the span. It would be nice to be able to
        // only update the crosshairs and not touch the take object.
        if(p_profile->itype == VERT_OVERLAY)
        {
            fw->updateMeanRange(linesToAverage, p_profile->itype);
            // fw->redraw_crosshairs(linesToAverage);
            this->updateOverlayParams(0);
        } else {
            fw->updateMeanRange(linesToAverage, p_profile->itype);
            fw->updateOverlayParams(0, 0, 0, 0, 0, 0); // signal that there is not an overlay plot

        }
    } else if (p_fft) {
        if (p_fft->vCrossButton->isChecked())
            fw->updateMeanRange(linesToAverage, VERTICAL_CROSS);
    } else {
        fw->updateMeanRange(linesToAverage, BASE);
    }
}

void ControlsBox::updateOverlayParams(int dummy)
{
    int lh_start, lh_end, cent_start, cent_end, rh_start, rh_end;
    int lh_width = this->overlay_lh_width_spin->value();
    int cent_width = this->overlay_cent_width_spin->value();
    int rh_width = this->overlay_rh_width_spin->value();

    // update list of parameters.
    // Currently uses the crosshairs to determine L, C, R position
    // and the UI sliders determine the span of each averaging.
    lh_start = fw->crossStartCol - lh_width/2;
    lh_end = lh_start + lh_width;

    rh_start = fw->crossWidth - rh_width/2;
    rh_end = rh_start + rh_width;

    cent_start = fw->crosshair_x - cent_width/2;
    cent_end = fw->crosshair_x + cent_width/2;

    validateOverlayParams(lh_start, lh_end, cent_start, cent_end, rh_start, rh_end);

    /*
    std::cout << "----- begin ControlsBox::updateOverlayParams -----\n";
    std::cout << "fw->crossWidth: " << fw->crossWidth << " fw->crosshair_x: " << fw->crosshair_x << std::endl;
    std::cout << "lh_start:   " << lh_start <<   ", lh_end:   " << lh_end << std::endl;
    std::cout << "rh_start:   " << rh_start <<   ", rh_end:   " << rh_end << std::endl;
    std::cout << "cent_start: " << cent_start << ", cent_end: " << cent_end << std::endl;
    std::cout << "----- end ControlsBox::updateOverlayParams -----\n";
    */

    // Send to frame worker, which sends to take object which sends to the mean filter.
    fw->updateOverlayParams(lh_start, lh_end, cent_start, cent_end, rh_start, rh_end);
}

void ControlsBox::validateOverlayParams(int &lh_start, int &lh_end,\
                                        int &cent_start, int &cent_end,\
                                        int &rh_start, int &rh_end)
{

    int width = fw->getFrameWidth() - 1; // last usable index

    // check lower bound:
    if(lh_start < 0)
        lh_start = 0;
    if(lh_end < 0)
        lh_end = 0;
    if(cent_start < 0)
        cent_start = 0;
    if(cent_end < 0)
        cent_end = 0;
    if(rh_start < 0)
        rh_start = 0;
    if(rh_end < 0)
        rh_end = 0;

    // check upper bound:
    if(lh_start > width)
        lh_start = width;
    if(lh_end > width)
        lh_end = width;
    if(cent_start > width)
        cent_start = width;
    if(cent_end > width)
        cent_end = width;
    if(rh_start > width)
        rh_start = width;
    if(rh_end > width)
        rh_end = width;

}

void ControlsBox::fft_slider_enable(bool toggled)
{
    bool enable = fw->crosshair_x != -1 && toggled;
    lines_slider->setEnabled(enable);
    line_average_edit->setEnabled(enable);
}

void ControlsBox::waterfallControls(bool enabled)
{
    red_slider.setEnabled(enabled);
    red_slider.setVisible(enabled);
    green_slider.setEnabled(enabled);
    green_slider.setVisible(enabled);
    blue_slider.setEnabled(enabled);
    blue_slider.setVisible(enabled);
    wflength_slider.setEnabled(enabled);
    wflength_slider.setVisible(enabled);

    red_label.setVisible(enabled);
    green_label.setVisible(enabled);
    blue_label.setVisible(enabled);

    redSpin.setVisible(enabled);
    greenSpin.setVisible(enabled);
    blueSpin.setVisible(enabled);

    wflength_label.setVisible(enabled);
    rgbPresetCombo.setVisible(enabled);
    saveRGBPresetButton.setVisible(enabled);
    diskSpaceBar.setVisible(enabled);
    diskSpaceLabel.setVisible(enabled);
}

void ControlsBox::overlayControls(bool see_it)
{
    overlay_lh_width->setEnabled(see_it);
    overlay_lh_width->setVisible(see_it);
    overlay_lh_width_label->setVisible(see_it);
    overlay_lh_width_label->setEnabled(see_it);
    overlay_lh_width_spin->setVisible(see_it);
    overlay_lh_width_spin->setEnabled(see_it);

    overlay_cent_width->setEnabled(see_it);
    overlay_cent_width->setVisible(see_it);
    overlay_cent_width_label->setVisible(see_it);
    overlay_cent_width_label->setEnabled(see_it);
    overlay_cent_width_spin->setVisible(see_it);
    overlay_cent_width_spin->setEnabled(see_it);

    overlay_rh_width->setEnabled(see_it);
    overlay_rh_width->setVisible(see_it);
    overlay_rh_width_label->setVisible(see_it);
    overlay_rh_width_label->setEnabled(see_it);
    overlay_rh_width_spin->setVisible(see_it);
    overlay_rh_width_spin->setEnabled(see_it);
}

void ControlsBox::setRGBWaterfall(int value)
{
    (void)value;
    emit updateRGB(red_slider.value(), green_slider.value(),
                   blue_slider.value());

}

void ControlsBox::debugThis()
{
    qDebug() << "Debug button function in controlsbox reached";

    qDebug() << "wflength slider max: " << wflength_slider.maximum() << "wf slider min: " << wflength_slider.minimum() << "wf slider value: " << wflength_slider.value();

    qDebug() << "--- PREFS debug output: ---";
    qDebug() << "Levels: ";
    qDebug() << "fv ceiling: " << prefs.frameViewCeiling;
    qDebug() << "fv floor: " << prefs.frameViewFloor;
    qDebug() << "dsf ceiling: " << prefs.dsfCeiling;
    qDebug() << "dsf floor: " << prefs.dsfFloor;
    qDebug() << "fft ceiling: " << prefs.fftCeiling;
    qDebug() << "fft floor: " << prefs.fftFloor;
    qDebug() << "stddevCeiling: " << prefs.stddevCeiling;
    qDebug() << "stddevFloor: " << prefs.stddevFloor;
    qDebug() << "--- END PREFS debug output ---";

    emit debugSignal();
}
