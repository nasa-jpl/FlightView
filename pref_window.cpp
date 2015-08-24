#include <QHBoxLayout>
#include <QVBoxLayout>

#include "profile_widget.h"
#include "pref_window.h"

preferenceWindow::preferenceWindow(frameWorker *fw, QTabWidget *qtw, QWidget *parent) :
    QWidget(parent)
{   
    /*! \brief Generates the preferenceWindow layout and tabs.
     * \paragraph
     *
     * We need a copy of the QTabWidget from the main window so that we can determine the current view window. Certain hardware controls
     * are only applicable to certain tabs in the main application and may cause issues if activated in the wrong tab.
     * \param qtw Used to create pointers to the current viewing widget.
     * \author JP Ryan
     */
    this->fw = fw;
    this->mainWinTab = qtw;
    this->frHeight = fw->getFrameHeight();
    this->frWidth = fw->getFrameWidth();

    closeButton = new QPushButton(tr("&Close"));
    createLogFileTab();
    createRenderingTab();

    connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));

    QTabWidget *tabs = new QTabWidget();
    tabs->addTab(renderingTab, "Rendering");
    tabs->addTab(logFileTab, "Log Files");

    enableControls(index = mainWinTab->currentIndex());
    connect(mainWinTab, SIGNAL(currentChanged(int)), this, SLOT(enableControls(int)));

    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(tabs);
    layout->addWidget(closeButton);
    this->setLayout(layout);
    this->setWindowTitle("Preferences");
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

    paraPixCheck   = new QCheckBox(tr("Enable Parallel Pixel Mapping"));
    ignoreFirstCheck = new QCheckBox(tr("Ignore First Row Data"));
    ignoreLastCheck  = new QCheckBox(tr("Ignore Last Row Data"));

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

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(camera_label, 0, 0, 1, 4);
    layout->addWidget(paraPixCheck, 1, 0, 1, 4);
    layout->addWidget(dataRangePrompt, 2, 0);
    layout->addWidget(leftBound, 2, 1);
    layout->addWidget(to, 2, 2);
    layout->addWidget(rightBound, 2, 3);
    layout->addWidget(nativeScaleButton, 3, 0, 1, 1);
    layout->addWidget(invert16bitButton, 3, 1, 1, 1);
    layout->addWidget(invert14bitButton, 3, 2, 1, 1);
    layout->addWidget(ignoreFirstCheck, 4, 0, 1, 2);
    layout->addWidget(ignoreLastCheck, 4, 2, 1, 2);

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
void preferenceWindow::invertRange()
{
    /*! \brief Set the inversion factor of the image and communicate the value to the backend. */
    uint factor = 65535; // (2^16) - 1;
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
}
void preferenceWindow::ignoreLastRow(bool checked)
{
    /*! \brief Ignore last row data for the purposes of averaging. */
    fw->skipLastRow(checked);
}
void preferenceWindow::enableParaPixMap(bool checked)
{
    /*! \brief Enables or Diables the Parallel Pixel Mapping based on the check box in the Rendering Tab */
    fw->to.paraPixRemap(checked);
}
