#include "waterfallviewerwindow.h"
#include "ui_waterfallviewerwindow.h"

waterfallViewerWindow::waterfallViewerWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::waterfallViewerWindow)
{
    ui->setupUi(this);
    wf = ui->waterfallWidget;

    connect(ui->wflengthSlider, SIGNAL(valueChanged(int)), wf, SLOT(changeWFLength(int)));
    connect(wf, &waterfall::statusMessageOut, [=](const QString message) {
        emit statusMessageOutSig(message);
    });
    wf->setSecondaryWF(true);

    // Boarder goes around the handle square.
    // By setting a background, you fill in the square.
    // The square seems to always span the width of the widget

    // handle::vertical is only be for the vertical slider orientation.
    // groove:vertical, set width for wider handle basically.

    sliderStylesheet = QString("\
                               .QSlider::handle:vertical {\
                                   background: #22B14C;\
                                   border: 5px solid #B5E61D;\
                                   width: 1px;\
                                   height: 10px;\
                               }\
                               .QSlider::groove:vertical {\
                                   border: 1px solid #262626;\
                                   width: 50px;\
                                   background: #393939;\
                                   margin: 0 1px;\
                               }\
                            ");
    ui->wflengthSlider->setStyleSheet(sliderStylesheet);
}

waterfallViewerWindow::~waterfallViewerWindow()
{
    delete ui;
}

void waterfallViewerWindow::setup(frameWorker *fw, int vSize, int hSize, startupOptionsType options) {
    wf->setup(fw, vSize, hSize, true, options);
    wf->changeWFLength(ui->wflengthSlider->value());
}

void waterfallViewerWindow::setSecondaryWF(bool isSecondary) {
    wf->setSecondaryWF(isSecondary);
}

void waterfallViewerWindow::setSpecImage(bool followMe, QImage *specImage) {
    wf->setSpecImage(followMe, specImage);
}

void waterfallViewerWindow::changeRGB(int r, int g, int b) {
    // Row numbers
    wf->changeRGB(r,g,b);
}

void waterfallViewerWindow::setRGBLevels(double r, double g, double b, double gamma, bool reprocess) {
    wf->setRGBLevels(r,g,b,gamma,reprocess);
}

void waterfallViewerWindow::setRGBLevelsAndReprocess(double r, double g, double b, double gamma) {
    wf->setRGBLevelsAndReprocess(r,g,b,gamma);
}

void waterfallViewerWindow::changeWFLength(int length) {
    // We might disable this for independent control
    //if(!independentLengthControl)
    ui->wflengthSlider->blockSignals(true);
    ui->wflengthSlider->setValue(length);
    ui->wflengthSlider->blockSignals(false);
    wf->changeWFLength(length);
}

void waterfallViewerWindow::setSpecOpacity(unsigned char opacity) {
    wf->setSpecOpacity(opacity);
}

void waterfallViewerWindow::updateCeiling(int c) {
    wf->updateCeiling(c);
}

void waterfallViewerWindow::updateFloor(int f) {
    wf->updateFloor(f);
}

void waterfallViewerWindow::setUseDSF(bool useDSF) {
    wf->setUseDSF(useDSF);
}

void waterfallViewerWindow::setRecordWFImage(bool recordImageOn) {
    wf->setRecordWFImage(recordImageOn);
}

void waterfallViewerWindow::debugThis() {
    wf->debugThis();
}

void waterfallViewerWindow::useEntireScreen() {
    ui->showWindowBoarderBtn->setChecked(false);
    on_showWindowBoarderBtn_clicked(false);
    this->showMaximized();
    ui->maximizeBtn->setText("N");
}

void waterfallViewerWindow::on_closeBtn_clicked()
{
    setWindowFlags((Qt::WindowType)(Qt::Window & ~Qt::FramelessWindowHint));
    ui->showWindowBoarderBtn->blockSignals(true);
    ui->showWindowBoarderBtn->setChecked(true);
    ui->showWindowBoarderBtn->blockSignals(false);
    this->close();
}

void waterfallViewerWindow::on_showWindowBoarderBtn_clicked(bool checked)
{
    if(checked) {
        hide();
        setWindowFlags((Qt::WindowType)(Qt::Window & ~Qt::FramelessWindowHint));
        show();
    } else {
        hide();
        setWindowFlags(Qt::Window
                | Qt::FramelessWindowHint);
        show();
        this->raise();
    }
}

void waterfallViewerWindow::on_maximizeBtn_clicked()
{
    if(this->isMaximized()) {
        this->showNormal();
        ui->maximizeBtn->setText("+");
    } else {
        this->showMaximized();
        ui->maximizeBtn->setText("N");
    }
}
