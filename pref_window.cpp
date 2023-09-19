#include <QHBoxLayout>
#include <QVBoxLayout>

#include "profile_widget.h"
#include "pref_window.h"

preferenceWindow::preferenceWindow(frameWorker *fw, QTabWidget *qtw, settingsT prefs, QWidget *parent) :
    QWidget(parent)
{   
    /*! \brief Generates the preferenceWindow layout and tabs.
     * \paragraph
     *
     * We need a copy of the QTabWidget from the main window so that we can determine the current view window. Certain hardware controls
     * are only applicable to certain tabs in the main application and may cause issues if activated in the wrong tab.
     * \param qtw Used to create pointers to the current viewing widget.
     * \author Jackie Ryan
     */
    this->fw = fw;
    this->mainWinTab = qtw;
    this->frHeight = fw->getFrameHeight();
    this->frWidth = fw->getFrameWidth();

    closeButton = new QPushButton(tr("&Close"));
    saveSettingsBtn = new QPushButton(tr("&Save Settings"));
    //createLogFileTab();
    createRenderingTab();

    connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));
    connect(saveSettingsBtn, SIGNAL(clicked()), this, SLOT(saveSettingsNow()));

    QTabWidget *tabs = new QTabWidget();
    tabs->addTab(renderingTab, "Rendering");
    if(logFileTab != NULL)
        tabs->addTab(logFileTab, "Log Files");

    enableControls(index = mainWinTab->currentIndex());
    connect(mainWinTab, SIGNAL(currentChanged(int)), this, SLOT(enableControls(int)));

    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(tabs);
    bottomLayout = new QHBoxLayout();
    bottomLayout->addWidget(closeButton);
    bottomLayout->addWidget(saveSettingsBtn);
    layout->addLayout(bottomLayout);

    this->setLayout(layout);
    this->setWindowTitle("Preferences");
    this->preferences = prefs;
    processPreferences();
}
void preferenceWindow::createLogFileTab()
{
    /*! \brief Generate the Log File browser.
     * \paragraph
     *
     * Currently unused. May be fully implemented for testing at a later date.
     */
    logFileTab = new QWidget();

    browseButton = new QPushButton(tr("&Browse..."));
    filePath = new QLineEdit();
    QLabel *logFilePrompt = new QLabel(tr("Log File Directory:"));

    filePath->setReadOnly(true);

    connect(browseButton, SIGNAL(clicked()), this, SLOT(getFilePath()));

    QHBoxLayout *layout = new QHBoxLayout();
    layout->addWidget(logFilePrompt);
    layout->addWidget(filePath);
    layout->addWidget(browseButton);

    logFileTab->setLayout(layout);
}
void preferenceWindow::createRenderingTab()
{
    /*! \brief Generate the controls and layout for the main preference window tab. */
    renderingTab = new QWidget();

    QLabel* dataRangePrompt = new QLabel(tr("Display data from:"));
    QLabel* to = new QLabel(tr("to"));

    paraPixCheck   = new QCheckBox(tr("Enable 2s compliment filter"));
    paraPixCheck->setToolTip("Enable or Disable 2s compliment filter on input data stream. See console output for initial filter state.");
    paraPixCheck->setChecked(true);

    setDarkStatusInFrameCheck = new QCheckBox("Write Dark Status to Frame Data");
    setDarkStatusInFrameCheck->setToolTip("Writes the dark collection status to the frame header");
    setDarkStatusInFrameCheck->setChecked(false);

    ignoreFirstCheck = new QCheckBox(tr("Ignore First Row Data"));
    ignoreLastCheck  = new QCheckBox(tr("Ignore Last Row Data"));
    ignoreFirstCheck->setToolTip("Check if first row contains metadata. Only available on certain liveview tabs (profile and FFT)");
    ignoreLastCheck->setToolTip("Check if last row contains metadata. Only available on certain liveview tabs (profile and FFT)");

    nativeScaleButton = new QRadioButton(tr("Native Scale"));
    invert16bitButton = new QRadioButton(tr("16-bit Bright-Dark Swap"));
    invert14bitButton = new QRadioButton(tr("14-bit Bright-Dark Swap"));

    nativeScaleButton->setChecked(true);

    camera_label = new QLabel;
    camera_t camera = fw->camera_type();
    if (camera == CL_6604A) {
        camera_label->setText(tr("Camera type: 6604A       Frame resolution:  %1 x %2").arg(frWidth).arg(frHeight));
    } else if (camera == CL_6604B) {
        paraPixCheck->setChecked(true);
        camera_label->setText(tr("Camera type: 6604B[CHROMA]       Frame resolution: %1 x %2").arg(frWidth).arg(frHeight));
    } else {
        camera_label->setText(tr("Camera type: Custom      Frame resolution: %1 x %2").arg(frWidth).arg(frHeight));
    }
    leftBound =  new QLineEdit();
    rightBound = new QLineEdit();
    leftBound->setText("0");
    base_scale = max_val[camera];
    rightBound->setText(tr("%1").arg(base_scale));

    ColorLabel = new QLabel();
    ColorLabel->setText("Image Color Scheme:");

    ColorScalePicker = new QComboBox();
    ColorScalePicker->addItem("Jet");
    ColorScalePicker->addItem("Grayscale");
    ColorScalePicker->addItem("Thermal");
    ColorScalePicker->addItem("Hues");
    ColorScalePicker->addItem("Polar");
    ColorScalePicker->addItem("Hot");
    ColorScalePicker->addItem("Cold");
    ColorScalePicker->addItem("Night");
    ColorScalePicker->addItem("Ion");
    ColorScalePicker->addItem("Spectrometer Candy");
    ColorScalePicker->addItem("Geography");
    ColorScalePicker->addItem("Gray with Red Top");

    ColorScalePicker->setToolTip("So many to choose from! Use the scroll wheel while hovering over the combo box");

    darkThemeCheck = new QCheckBox("Use dark theme");
    darkThemeCheck->setToolTip("Select this for a darker UI theme");

    /* If the data range needs to be adjustable later, uncomment this and the #include and definition line in the header
     * and commment out the setReadOnly lines below.
    valid = new QIntValidator();
    valid->setRange(0, 65535);
    leftBound->setValidator(valid);
    rightBound->setValidator(valid); */

    rightBound->setReadOnly(true);
    leftBound ->setReadOnly(true);

    connect(ignoreLastCheck, SIGNAL(clicked(bool)), this, SLOT(ignoreLastRow(bool)));
    connect(ignoreFirstCheck, SIGNAL(clicked(bool)), this, SLOT(ignoreFirstRow(bool)));
    connect(invert16bitButton, SIGNAL(clicked()), this, SLOT(invertRange()));
    connect(invert14bitButton, SIGNAL(clicked()), this, SLOT(invertRange()));
    connect(nativeScaleButton, SIGNAL(clicked()), this, SLOT(invertRange()));
    connect(paraPixCheck, SIGNAL(clicked(bool)), this, SLOT(enableParaPixMap(bool)));
    connect(setDarkStatusInFrameCheck, SIGNAL(clicked(bool)), this, SLOT(dsInFrameSlot(bool)));
    connect(ColorScalePicker, SIGNAL(activated(int)), this, SLOT(setColorScheme(int)));
    connect(darkThemeCheck, SIGNAL(clicked(bool)), this, SLOT(setDarkTheme(bool)));


    QGridLayout *layout = new QGridLayout();
    layout->addWidget(camera_label, 0, 0, 1, 4);
    layout->addWidget(paraPixCheck, 1, 0, 1, 4);
    layout->addWidget(setDarkStatusInFrameCheck, 1, 2, 1, 2);
    layout->addWidget(dataRangePrompt, 2, 0);
    layout->addWidget(leftBound, 2, 1);
    layout->addWidget(to, 2, 2);
    layout->addWidget(rightBound, 2, 3);
    layout->addWidget(nativeScaleButton, 3, 0, 1, 1);
    layout->addWidget(invert16bitButton, 3, 1, 1, 1);
    layout->addWidget(invert14bitButton, 3, 2, 1, 1);
    layout->addWidget(ignoreFirstCheck, 4, 0, 1, 2);
    layout->addWidget(ignoreLastCheck, 4, 2, 1, 2);
    layout->addWidget(ColorLabel, 5, 0);
    layout->addWidget(ColorScalePicker, 5,1);
    layout->addWidget(darkThemeCheck, 5, 2);

    renderingTab->setLayout(layout);
    //enableControls(mainWinTab->currentIndex());
}
void preferenceWindow::getFilePath()
{
    /*! \brief Test slot for saving log files. */
    QString directory = QFileDialog::getExistingDirectory(this, tr("Pick a directory"));
    filePath->setText(directory);
}
void preferenceWindow::enableControls(int ndx)
{
    /*! \brief Changes the active controls based on the current view widget.
     * \paragraph
     * Profile widgets and FFT widgets make use of the ignore first row and ignore last row check boxes while other widgets do not.
     */
    index = ndx;
    if(qobject_cast<profile_widget*>(mainWinTab->widget(index)) || qobject_cast<fft_widget*>(mainWinTab->widget(index))) {
        ignoreFirstCheck->setEnabled(true);
        ignoreLastCheck->setEnabled(true);
    } else {
        ignoreFirstCheck->setEnabled(false);
        ignoreLastCheck->setEnabled(false);
    }
}

void preferenceWindow::processPreferences()
{
    // Initial run.
    // Set the GUI controls to the preference values
    // and trigger any needed actions

    invert14bitButton->setChecked(preferences.brightSwap14);
    if(preferences.brightSwap14)
        invert14bitButton->click();

    invert16bitButton->setChecked(preferences.brightSwap16);
    if(preferences.brightSwap16)
        invert16bitButton->click();

    ignoreLastCheck->setChecked(preferences.skipLastRow);
    ignoreLastCheck->clicked(preferences.skipLastRow);

    ignoreFirstCheck->setChecked(preferences.skipFirstRow);
    ignoreFirstCheck->clicked(preferences.skipFirstRow);

    nativeScaleButton->setChecked(preferences.nativeScale);
    nativeScaleButton->clicked(preferences.nativeScale);

    paraPixCheck->setChecked(preferences.use2sComp);
    paraPixCheck->clicked(preferences.use2sComp);

    setDarkStatusInFrameCheck->setChecked(preferences.setDarkStatusInFrame);
    setDarkStatusInFrameCheck->clicked(preferences.setDarkStatusInFrame);

    ColorScalePicker->setCurrentIndex(preferences.frameColorScheme);
    ColorScalePicker->activated(preferences.frameColorScheme);

    darkThemeCheck->setChecked(preferences.useDarkTheme);
    darkThemeCheck->clicked(preferences.useDarkTheme);
}


void preferenceWindow::invertRange()
{
    /*! \brief Set the inversion factor of the image and communicate the value to the backend. */
    uint factor = 65535; // (2^16) - 1;

    preferences.brightSwap14 = invert14bitButton->isChecked();
    preferences.brightSwap16 = invert16bitButton->isChecked();
    preferences.nativeScale = nativeScaleButton->isChecked();

    if( invert14bitButton->isChecked() )
    {
        factor = 16383; // (2^14) - 1;
        rightBound->setText(tr("16383"));
        fw->to.setInversion(true, factor);
    }
    else if( invert16bitButton->isChecked() )
    {
        rightBound->setText(tr("65535"));
        fw->to.setInversion(true, factor);

    }
    else
    {
        rightBound->setText(tr("%1").arg(base_scale));
        fw->to.setInversion(false, factor);
    }
}
void preferenceWindow::ignoreFirstRow(bool checked)
{
    /*! \brief Ignore first row data for the purpose of averaging. */
    fw->skipFirstRow(checked);
    preferences.skipFirstRow = checked;
}
void preferenceWindow::ignoreLastRow(bool checked)
{
    /*! \brief Ignore last row data for the purposes of averaging. */
    fw->skipLastRow(checked);
    preferences.skipLastRow = checked;
}
void preferenceWindow::enableParaPixMap(bool checked)
{
    /*! \brief Enables or Diables the Parallel Pixel Mapping based on the check box in the Rendering Tab */
    fw->to.paraPixRemap(checked);
    preferences.use2sComp = checked;
    makeStatusMessage(QString("2s Compliment Filter: %1").arg(checked?"Enabled":"Disabled"));
}

void preferenceWindow::dsInFrameSlot(bool checked) {
    // Set dark status pixels in the frame data.
    // Do not check if geometry is < 159 pixels!
    fw->to.enableDarkStatusPixelWrite(checked);
    preferences.setDarkStatusInFrame = checked;
    makeStatusMessage(QString("Mark frames with dark status: %1").arg(checked?"Enabled":"Disabled"));
}

void preferenceWindow::setColorScheme(int index)
{
    //fw->color_scheme = index;
    fw->setColorScheme(index, preferences.useDarkTheme);
    preferences.frameColorScheme = index;
    //std::cerr << "selected index: " << index << std::endl;
}

void preferenceWindow::setDarkTheme(bool useDarkChecked)
{
    preferences.useDarkTheme = useDarkChecked;
    fw->setColorScheme(preferences.frameColorScheme, preferences.useDarkTheme);
}

void preferenceWindow::saveSettingsNow()
{
    emit saveSettings();
}

settingsT preferenceWindow::getPrefs()
{
    return this->preferences;
}

void preferenceWindow::makeStatusMessage(QString internalMessage)
{
    emit statusMessage(QString("[preferenceWindow]: %1").arg(internalMessage));
}
