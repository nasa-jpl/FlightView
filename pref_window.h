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
#include <QComboBox>
#include <QLabel>

//#include <QIntValidator>

/* Live View includes */
#include "frame_worker.h"
#include "profile_widget.h"
#include "fft_widget.h"
#include "preferences.h"

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
 * \author Jackie Ryan
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

    QWidget *logFileTab = NULL;
    QWidget *renderingTab;

    QLabel *camera_label;

    QLineEdit *filePath;
    QLineEdit *leftBound;
    QLineEdit *rightBound;

    //QIntValidator *valid;

    QPushButton *closeButton;
    QPushButton *saveSettingsBtn;
    QPushButton *browseButton;

    QCheckBox *paraPixCheck;
    QCheckBox *setDarkStatusInFrameCheck;
    QCheckBox *ignoreFirstCheck;
    QCheckBox *ignoreLastCheck;

    QRadioButton *nativeScaleButton;
    QRadioButton *invert16bitButton;
    QRadioButton *invert14bitButton;

    QHBoxLayout *bottomLayout;

    QLabel *ColorLabel;
    QComboBox *ColorScalePicker;
    QCheckBox *darkThemeCheck;

    bool havePreferencesLoaded = false;
    settingsT preferences;

public:
    preferenceWindow(frameWorker *fw, QTabWidget *qtw, settingsT prefs, QWidget *parent = 0);

    settingsT getPrefs();

private:
    void createLogFileTab();
    void createRenderingTab();
    void processPreferences();
    void makeStatusMessage(QString internalMessage);

private slots:
    void getFilePath();
    void enableControls(int ndx);

    void enableParaPixMap(bool checked);
    void dsInFrameSlot(bool checked);
    void invertRange();
    void ignoreFirstRow(bool checked);
    void ignoreLastRow(bool checked);
    void setColorScheme(int index);
    void setDarkTheme(bool useDarkChecked);
    void saveSettingsNow();

signals:
    void saveSettings();
    void statusMessage(QString message);
};

#endif
