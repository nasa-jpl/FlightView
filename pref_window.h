#ifndef PREF_WINDOW_H
#define PREF_WINDOW_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QRadioButton>
#include <QLineEdit>
#include <QFileDialog>
#include <QTabWidget>
//#include <QIntValidator>

#include "frame_worker.h"
#include "profile_widget.h"
#include "fft_widget.h"

class preferenceWindow : public QWidget
{
    Q_OBJECT

    int index;
    int base_scale;
    int rowsToSkip = 1;
    unsigned int frHeight;
    unsigned int frWidth;

    QTabWidget* mainWinTab;
    frameWorker* fw;

    profile_widget* ppw;
    fft_widget* ffw;

    QWidget* logFileTab;
    QWidget* renderingTab;

    QLabel* camera_label;

    QLineEdit* filePath;
    QLineEdit* leftBound;
    QLineEdit* rightBound;

    //QIntValidator* valid;

    QPushButton* closeButton;
    QPushButton* browseButton;

    QCheckBox* chromaPixCheck;
    QCheckBox* ignoreFirstCheck;
    QCheckBox* ignoreLastCheck;

    QRadioButton* nativeScaleButton;
    QRadioButton* invert16bitButton;
    QRadioButton* invert14bitButton;
public:
    preferenceWindow(frameWorker* fw, QTabWidget* qtw, QWidget *parent = 0);

private:
    void createLogFileTab();
    void createRenderingTab();

private slots:
    void getFilePath();
    void enableControls(int ndx );

    void enableChromaPixMap( bool checked );
    /* Enables / Diables the Chroma Pixel Mapping based on
     * the check box in the Rendering Tab */

    void invertRange();
    /* Inverts the data range as it is displayed and saved by liveview2.
     * Checked inverts the range. This is the same thing as floor and
     * ceiling. */

    void ignoreFirstRow( bool checked );

    void ignoreLastRow( bool checked );
};

#endif
