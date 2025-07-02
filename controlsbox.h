#ifndef CONTROLSBOX_H
#define CONTROLSBOX_H

/* Qt GUI includes */
#include <QButtonGroup>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QSettings>
#include <QProgressBar>

/* standard includes */
#include <stdint.h>

/* boost includes - is this still needed? */
#include "boost/shared_array.hpp"

/* LiveView includes */
#include "fft_widget.h"
#include "frameview_widget.h"
#include "flight_widget.h"
#include "rgbadjustments.h"
#include "histogram_widget.h"
#include "playback_widget.h"
#include "profile_widget.h"
#include "pref_window.h"
#include "preferences.h"
#include "startupOptions.h"
#include "filenamegenerator.h"
#include "initialsetup.h"
#include "fileformats.h"

/*! \file
 *  \brief Widget which contains the GUI elements common to several or all plotting widgets.
 * \paragraph
 *
 * The ControlsBox is a wrapper GUI class which contains the (mostly) static controls between widgets. After establishing the buttons
 * in the constructor, the class will call the function tab_slot_changed(int index) to establish widget-specific controls and settings.
 * For instance, all profile widgets and FFTs make use of the Lines To Average slider rather than the disabled Std Dev N slider. As Qt
 * does not support a pure virtual interface for widgets, each widget must make a connection to its own version of updateCeiling(int c),
 * updateFloor(int f), and any other widget specfic action within its case in tab_slot_changed(int index). The beginning of this function
 * specifies the behavior for when tabs are exited - all connections made must be disconnected.
*/

class ControlsBox : public QGroupBox
{
    Q_OBJECT

    startupOptionsType options;
    fileNameGenerator fnamegen;
    rgbAdjustments* rgbLevels;

public:
    explicit ControlsBox(frameWorker *fw, QTabWidget *tw, startupOptionsType options, QWidget *parent = 0);

    frameWorker *fw;
    preferenceWindow *prefWindow;
    QHBoxLayout controls_layout;
    initialSetup setupUI;

    /* LEFT SIDE BUTTONS (Collections) */
    QGridLayout *collections_layout;
    QWidget CollectionButtonsBox;
    QPushButton collect_dark_frames_button;
    QPushButton stop_dark_collection_button;
    QPushButton showRGBLevelsButton;
    QPushButton load_mask_from_file;
    QPushButton showSecondWFBtn;
    QPushButton pref_button;
    QString fps;
    float fps_float = 0.0;
    QLabel fps_label;
    QLabel server_ip_label;
    QLabel server_port_label;
    QLabel frameNumberLabel;
    QCheckBox pausePlaybackChk;

    /* MIDDLE BUTTONS (Sliders) */
    QGridLayout *sliders_layout;
    QWidget ThresholdingSlidersBox;
    QSlider *std_dev_N_slider;
    QSlider *lines_slider;
    QSlider ceiling_slider;
    QSlider floor_slider;
    QSpinBox *std_dev_N_edit;
    QSpinBox *line_average_edit;
    QSpinBox ceiling_edit;
    QSpinBox floor_edit;
    QSlider red_slider;
    QSlider green_slider;
    QSlider blue_slider;
    QSpinBox redSpin;
    QSpinBox greenSpin;
    QSpinBox blueSpin;
    QComboBox rgbPresetCombo;
    QSlider wflength_slider;
    QLabel red_label;
    QLabel green_label;
    QLabel blue_label;
    QLabel wflength_label;
    QLabel *std_dev_n_label;
    QLabel *lines_label;
    QCheckBox low_increment_cbox;
    QCheckBox use_DSF_cbox;
    QCheckBox show_rgb_lines_cbox;
    QCheckBox useRatioCbox;

    /* RIGHT SIDE BUTTONS (save) */
    QGridLayout *save_layout;
    QPushButton showConsoleLogBtn;
    QPushButton showXioSetupBtn;
    QWidget SaveButtonsBox;
    QPushButton save_finite_button;
    QPushButton start_saving_frames_button;
    QPushButton stop_saving_frames_button;
    QPushButton select_save_location;
    QSpinBox frames_save_num_edit;
    QSpinBox frames_save_num_avgs_edit;
    QLineEdit filename_edit;
    QPushButton set_filename_button;
    QPushButton debugButton;
    QPushButton saveRGBPresetButton;
    QProgressBar diskSpaceBar;
    QLabel diskSpaceLabel;

    // Overlay profile only:
    QSlider * overlay_lh_width;
    QSlider * overlay_cent_width;
    QSlider * overlay_rh_width;
    QLabel * overlay_lh_width_label;
    QLabel * overlay_cent_width_label;
    QLabel * overlay_rh_width_label;
    QSpinBox * overlay_lh_width_spin;
    QSpinBox * overlay_cent_width_spin;
    QSpinBox * overlay_rh_width_spin;

    frameview_widget *p_frameview;
    flight_widget *p_flight;
    histogram_widget *p_histogram;
    profile_widget *p_profile;
    fft_widget *p_fft;
    playback_widget *p_playback;

protected:
    void closeEvent(QCloseEvent *e);

private:
    QTabWidget *qtw;
    QWidget *old_tab;
    QWidget *current_tab;
    int diskWarningCounter = 0;
    int diskErrorCounter = 0;
    int ceiling_maximum;
    int previousNumSaved;
    bool checkForOverwrites = true;
    void waterfallControls(bool enabled);
    void overlayControls(bool enabled);

    QSettings *settings;
    void setDefaultSettings();
    void loadSettings();
    void saveSettings();
    void saveSingleRGBPreset(int index, int r, int g, int b);

    void setLevelToPrefs(bool isCeiling, int val);
    void updateUIToPrefs();

    QString prefsFilename;
    settingsT prefs;
    settingsT defaultPrefs;

    int spatialWidth = 0;
    int spectralWidth = 0;
    int bandRed = 0;
    int bandGreen = 0;
    int bandBlue = 0;

    int previousRGBPresetIndex = 0;

    // Individual tab Dark Subtraction settings:
    bool fpaDSF = false;
    bool monoWFDSF = false;
    bool flightDSF = false;
    bool horizontalMeanDSF = false;
    bool horizontalCrossDSF = false;
    bool verticalMeanDSF = false;
    bool verticalCrossDSF = false;
    bool verticalOverlayDSF = false;
    bool playbackDSF = false;


signals:
    /*! \brief Passes the message to save raw frames at the backend.
     * \paragraph
     *
     * Please note that at the time this message is passed, the file name parameter must be valid or the program
     * will experience a segmentation violation. Very little checking of location validity and permissions is done
     * at the backend. */
    void startSavingFinite(unsigned int length, QString fname, unsigned int navgs);
    void startDataCollection(QString baseFilename);

    /*! \brief Ends the saving loop at the backend. */
    void stopSaving();
    void stopDataCollection();

    /*! \brief Passes the DSF the message to begin averaging dark frames for all live widgets. */
    void startDSFMaskCollection();

    void loadDarkFile(QString filename, fileFormat_t formatSelected);

    /*! \brief Averages the collected frames and loads in the mask. */
    void stopDSFMaskCollection();

    void toggleStdDevCalculation(bool enabled);

    /*! \brief Passes the information needed to generate the dark mask and load it into the DSF in the playback_widget. */
    void mask_selected(QString file_name, unsigned int bytes_to_read, long offset);
    void updateRGB(int r, int g, int b);
    void sendRGBLevels(double r, double g, double b, double gamma, bool reprocess);
    void updateWFLength(int length);
    void showSecondWF();
    void setWFTargetFPS_primary(int targetFPS);
    void setWFTargetFPS_secondary(int targetFPS);
    void setWFTargetFPS_render(int targetFPS);
    void haveReadPreferences(settingsT prefs);

    void setCameraPause(bool isPaused);

    void statusMessage(QString message);
    void warningMessage(QString message);
    void errorMessage(QString message);
    void showConsoleLog();
    void debugSignal();

public slots:
    void tab_changed_slot(int index);
    void setRGBWaterfall(int value);
    void getPrefsExternalTrig();
    void setFrameNumber(int number);

    void handleFloorCeilingChangeFromDisplayWidget(double floor, double ceiling);

    // These are intended to be run from ARTIC
    // in concert with shutter operation.
    void startTakingDarks();
    void stopTakingDarks();
    void stopSavingData();
    void toggleDSFUsage(bool makeuseofDSF);

private slots:
    void increment_slot(bool t);
    void attempt_pointers(QWidget *tab);
    void disconnect_old_tab();
    void display_std_dev_slider();
    void display_lines_slider();
    void update_backend_delta();
    void updateFloor(int f);
    void updateCeiling(int c);
    void updateDiskSpace(quint64 total, quint64 available);
    void showSetup();


    /*! \addtogroup savingfunc Frame saving functions
     * Contains functions which control the processes needed to save frames.
     * @{ */
    void show_save_dialog();
    void save_remote_slot(const QString &unverifiedName, unsigned int nFrames, unsigned int numAvgs);
    void save_finite_button_slot();
    void stop_continous_button_slot();
    void updateSaveFrameNum_slot(unsigned int n);
    int validateFileName(const QString &name);
    /*! @} */

    /*! \addtogroup maskfunc Mask recording functions
     * Contains the functions which control the processes needed to record and use Dark
     * Subtraction Filters
     * @{ */
    void loadDarkFromFile();
    void getMaskFile(); // depreciated
    void start_dark_collection_slot();
    void stop_dark_collection_slot();
    void use_DSF_general(bool checked);
    /*! @} */

    void load_pref_window();
    void updatePenWidth(int penWidth);
    void transmitChange(int linesToAverage);
    void updateOverlayParams(int dummy);
    void validateOverlayParams(int &lh_start, int &lh_end, int &cent_start, int &cent_end, int &rh_start, int &rh_end);

    void fft_slider_enable(bool toggled);

    void triggerSaveSettings();

    void debugThis();

};
Q_DECLARE_METATYPE(enum fileFormat_t)

#endif // CONTROLSBOX_H
