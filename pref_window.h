#ifndef PREF_WINDOW_H
#define PREF_WINDOW_H

/* Qt includes */
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QRadioButton>
#include <QLineEdit>
#include <QFileDialog>
#include <QTabWidget>
//#include <QIntValidator>

/* Live View includes */
#include "frame_worker.h"
#include "profile_widget.h"
#include "fft_widget.h"

/*! \file
 * \brief Adjusts hardware settings in the backend.
 * \paragraph
 *
 * The preferenceWindow offers control over the hardware conditions of the current camera. Many of the command options
 * in this window directly affect the raw data as it is received at the back end. Parallel Pixel Remapping may be turned on
 * or off for cameras with 6604B geometry (1280x480 resolution). On other cameras, this option is disabled. Options for 14-
 * and 16-bit bright-dark swapping are also included. All data arriving on the data bus will be inverted by the specified
 * factor. As a sanity check, the expected data range is displayed above this option. The assumed camera type and geometry
 * are listed at the top of the window. Additionally, the first or last row data in the raw image may be excluded from the
 * image. This option only applies to linear profiles. Log files are not currently an implemented feature.
 * \author JP Ryan
 */

class preferenceWindow : public QWidget
{
    Q_OBJECT

    int index;
    int base_scale;
    unsigned int frHeight;
    unsigned int frWidth;

    QTabWidget *mainWinTab;
    frameWorker *fw;

    QWidget *logFileTab;
    QWidget *renderingTab;

    QLabel *camera_label;

    QLineEdit *filePath;
    QLineEdit *leftBound;
    QLineEdit *rightBound;

    //QIntValidator *valid;

    QPushButton *closeButton;
    QPushButton *browseButton;

    QCheckBox *paraPixCheck;
    QCheckBox *ignoreFirstCheck;
    QCheckBox *ignoreLastCheck;

    QRadioButton *nativeScaleButton;
    QRadioButton *invert16bitButton;
    QRadioButton *invert14bitButton;
public:
    preferenceWindow(frameWorker *fw, QTabWidget *qtw, QWidget *parent = 0);

private:
    void createLogFileTab();
    void createRenderingTab();

private slots:
    void getFilePath();
    void enableControls(int ndx);

    void enableParaPixMap(bool checked);
    void invertRange();
    void ignoreFirstRow(bool checked);
    void ignoreLastRow(bool checked);
};

#endif
