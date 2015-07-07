#include <QHBoxLayout>
#include <QVBoxLayout>

#include "profile_widget.h"
#include "pref_window.h"

preferenceWindow::preferenceWindow(frameWorker* fw, QTabWidget* qtw, QWidget* parent) : QWidget(parent)
{   
    this->fw = fw;
    this->mainWinTab = qtw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();

    closeButton = new QPushButton(tr("&Close"));
    createLogFileTab();
    createRenderingTab();

    connect(closeButton,SIGNAL(clicked()),this,SLOT(close()));

    QTabWidget* tabs = new QTabWidget;
    tabs->addTab(renderingTab,"Rendering");
    tabs->addTab(logFileTab,"Log Files");

    enableControls( index = mainWinTab->currentIndex() );
    connect(mainWinTab, SIGNAL(currentChanged(int)), this, SLOT(enableControls(int)));

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(tabs);
    layout->addWidget(closeButton);
    this->setLayout(layout);
    this->setWindowTitle("Preferences");
}
void preferenceWindow::createLogFileTab()
{
    logFileTab = new QWidget;

    browseButton = new QPushButton( tr("&Browse...") );
    filePath = new QLineEdit;
    QLabel* logFilePrompt = new QLabel( tr("Log File Directory:") );

    filePath->setReadOnly(true);

    connect(browseButton,SIGNAL(clicked()),this,SLOT(getFilePath()));

    QHBoxLayout* layout = new QHBoxLayout;
    layout->addWidget(logFilePrompt);
    layout->addWidget(filePath);
    layout->addWidget(browseButton);

    logFileTab->setLayout(layout);
}
void preferenceWindow::createRenderingTab()
{
    renderingTab = new QWidget;

    QLabel* dataRangePrompt = new QLabel(tr("Display data from:"));
    QLabel* to = new QLabel(tr("to"));

    chromaPixCheck   = new QCheckBox( tr("Enable Chroma Pixel Mapping") );
    ignoreFirstCheck = new QCheckBox( tr("Ignore First Row Data")       );
    ignoreLastCheck  = new QCheckBox( tr("Ignore Last Row Data")        );

    nativeScaleButton = new QRadioButton( tr("Native Scale")            );
    invert16bitButton = new QRadioButton( tr("16-bit Bright-Dark Swap") );
    invert14bitButton = new QRadioButton( tr("14-bit Bright-Dark Swap") );
    nativeScaleButton->setChecked( true );

    camera_label = new QLabel;
    camera_t camera = fw->camera_type();
    if( camera == CL_6604A )
    {
        camera_label->setText( tr("Camera type: 6604A       Frame resolution:  %1 x %2").arg(frWidth).arg(frHeight) );
    }
    else if( camera == CL_6604B )
    {
        chromaPixCheck->setChecked( true );
        camera_label->setText( tr("Camera type: 6604B[Chroma]       Frame resolution: %1 x %2").arg(frWidth).arg(frHeight) );
    }
    else
    {
        camera_label->setText( tr("Camera type: Custom      Frame resolution: %1 x %2").arg(frWidth).arg(frHeight) );
    }
    leftBound =  new QLineEdit;
    rightBound = new QLineEdit;
    leftBound->setText("0");
    if( fw->to.cam_type == CL_6604A )
        base_scale = 16383;
    else
        base_scale = 65535;
    rightBound->setText(tr("%1").arg(base_scale));

    /* If the data range needs to be adjustable later, uncomment this and the #include and definition line in the header
     * and commment out the setReadOnly lines below.
    valid = new QIntValidator;
    valid->setRange(0,65535);
    leftBound->setValidator(valid);
    rightBound->setValidator(valid); */

    rightBound->setReadOnly( true );
    leftBound ->setReadOnly( true );

    connect(   ignoreLastCheck, SIGNAL(clicked(bool)), this, SLOT(ignoreLastRow(bool))      );
    connect(  ignoreFirstCheck, SIGNAL(clicked(bool)), this, SLOT(ignoreFirstRow(bool))     );
    connect( invert16bitButton, SIGNAL(clicked()),     this, SLOT(invertRange())            );
    connect( invert14bitButton, SIGNAL(clicked()),     this, SLOT(invertRange())            );
    connect( nativeScaleButton, SIGNAL(clicked()),     this, SLOT(invertRange())            );
    connect(    chromaPixCheck, SIGNAL(clicked(bool)), this, SLOT(enableChromaPixMap(bool)) );

    QGridLayout* layout = new QGridLayout;
    layout->addWidget(camera_label,0,0,1,4);
    layout->addWidget(chromaPixCheck,1,0,1,4);
    layout->addWidget(dataRangePrompt,2,0);
    layout->addWidget(leftBound,2,1);
    layout->addWidget(to,2,2);
    layout->addWidget(rightBound,2,3);
    layout->addWidget(nativeScaleButton,3,0,1,1);
    layout->addWidget(invert16bitButton,3,1,1,1);
    layout->addWidget(invert14bitButton,3,2,1,1);
    layout->addWidget(ignoreFirstCheck,4,0,1,2);
    layout->addWidget(ignoreLastCheck,4,2,1,2);

    renderingTab->setLayout(layout);
    //enableControls(mainWinTab->currentIndex());
}
void preferenceWindow::getFilePath()
{
    QString directory = QFileDialog::getExistingDirectory(this,tr("Pick a directory"),"/home/jryan/NGIS_DATA/");
    filePath->setText(directory);
}
void preferenceWindow::enableControls( int ndx )
{
    ppw = NULL;
    ffw = NULL;
    index = ndx;
    if( (ppw = qobject_cast<profile_widget*>( mainWinTab->widget(index))) || (ffw = qobject_cast<fft_widget*>(mainWinTab->widget(index))) )
    {
        ignoreFirstCheck->setEnabled( true );
        ignoreFirstRow( ignoreFirstCheck->isChecked() );
        ignoreLastCheck->setEnabled( true );
        ignoreLastRow( ignoreLastCheck->isChecked() );
    }
    else
    {
        ignoreFirstCheck->setEnabled( false );
        ignoreFirstCheck->setChecked( false );
        //ignoreFirstRow( false );
        ignoreLastCheck->setEnabled( false );
        ignoreLastCheck->setChecked( false );
        //ignoreLastRow( false );
    }
}
void preferenceWindow::invertRange()
{
    uint factor = 65535; // (2^16) - 1;
    if( invert14bitButton->isChecked() )
    {
        factor = 16383; // (2^14) - 1;
        rightBound->setText( tr("16383") );
        fw->to.setInversion( true, factor );
    }
    else if( invert16bitButton->isChecked() )
    {
        rightBound->setText( tr("65535") );
        fw->to.setInversion( true, factor );
    }
    else
    {
        rightBound->setText( tr("%1").arg(base_scale) );
        fw->to.setInversion( false, factor );
    }
}
void preferenceWindow::ignoreFirstRow( bool checked )
{
    if ( ppw )
    {
        if( checked )
            ppw->updateStartRow( rowsToSkip );
        else
            ppw->updateStartRow( 0 );
    }
    else if( ffw )
    {
        if( checked )
            ffw->fw->to.update_start_row( rowsToSkip );
        else
            ffw->fw->to.update_start_row( 0 );
    }
}
void preferenceWindow::ignoreLastRow( bool checked )
{
    if( ppw )
    {
        if( checked )
            ppw->updateEndRow( frHeight - rowsToSkip );
    else
            ppw->updateEndRow( frHeight );
    }
    else if( ffw )
    {
        if( checked )
            ffw->fw->to.update_end_row( frHeight - rowsToSkip );
        else
            ffw->fw->to.update_end_row( frHeight );
    }

}
void preferenceWindow::enableChromaPixMap( bool checked )
{
    fw->to.chromaPixRemap( checked );
}
