#ifndef FLIGHT_WIDGET_H
#define FLIGHT_WIDGET_H

#include <unistd.h>
#include <cstdio>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/system/error_code.hpp>
#include <boost/filesystem/path.hpp>

#include <QObject>
#include <QWidget>
#include <QSplitter>
#include <QTimer>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>

// include <qcustomplot.h>

#include "gpsmanager.h"

#include "frame_worker.h"
#include "frameview_widget.h"
#include "qledlabel.h"
#include "flightindicators.h"
#include "startupOptions.h"
#include "wfshared.h"
#include "wfengine.h"
#include "waterfall.h"
#include "waterfallviewerwindow.h"
#include "preferences.h"
#include "flightappstatustypes.h"

#include "qfi/qfi_EADI.h"

class flight_widget : public QWidget
{
    Q_OBJECT

    frameWorker *fw;
    flightAppStatus_t *flightStatus = NULL;
    wfengine *wfcomputer = NULL;
    QThread *wfcompThread = NULL;
    waterfall *waterfall_widget = NULL;
    waterfallViewerWindow *secondWF = NULL;
    bool waterfallEngineReady = false;

    //QThread *wfThread = NULL;
    frameview_widget *dsf_widget;
    void setupWFConnections();

    gpsManager *gps;
    QString primaryGPSLogLocation; // directory, filename is automatic for primary log
    QString gpsHostname;
    uint16_t gpsPort;
    bool startedPrimaryGPSLog = false;
    startupOptionsType options;
    fs::space_info diskSpace;
    QTimer *diskCheckerTimer = NULL;
    QTimer *fpsLoggingTimer = NULL;
    QTimer hideRGBTimer;

    QSplitter lrSplitter;
    QSplitter rhSplitter;
    QSplitter *gpsPlotSplitter = NULL;

    void updateLabel(QLabel *label, QString text);

    QGridLayout layout;
    QVBoxLayout rhLayout;
    QGroupBox flightControls; // All plots and controls are inside this
    QGridLayout flightControlLayout; // Labels and buttons
    QGridLayout flightPlotsLayout;
    QGridLayout flightAvionicsLayout;

    flightIndicators *fi = NULL;
    fiUI_t flightDisplayElements;

    // stand-in items for flight controls:
    QLabel diskLEDLabel;
    QLedLabel *diskLED = NULL;
    QLedLabel *cameraLinkLED = NULL;

    QStringList priorGPSErrorMessages;
    QStringList priorGPSWarningMessages;
    QStringList totalGPSStatusMessages;
    QTimer *gpsMessageCycleTimer = NULL;
    QTimer *gpsMessageToLogReporterTimer = NULL;
    QMutex gpsMessageMutex;
    unsigned int messageIndex = 0;
    bool recentlyClearedErrors = false;

    // GPS Widgets:
    bool useAvionicsWidgets = false;
    qfi_EADI *EADI = NULL;
    qfi_EHSI *EHSI = NULL;
    qfi_ASI *ASI = NULL;
    qfi_VSI *VSI = NULL;

    uint16_t redRow;
    uint16_t greenRow;
    uint16_t blueRow;

    bool showRGBp = false;

    settingsT prefs;
    bool havePrefs = false;

    bool stickyFPSError;
    bool stickyDiskFull;
    int FPSErrorCounter;
    void processFPSError();


public:
    explicit flight_widget(frameWorker *fw, startupOptionsType options, flightAppStatus_t *flightStatus, QWidget *parent = nullptr);
    ~flight_widget();
    const unsigned int slider_max = (1<<16) * 1.1;
    bool slider_low_inc = false;
    double getCeiling();
    double getFloor();
    void toggleDisplayCrosshair();

public slots:
    void handleNewFrame();
    void resetFPSError();
    void clearStickyErrors();

    void startGPS(QString gpsHostname, uint16_t gpsPort, QString primaryLogLocation); // connect to GPS and start primary log

    // Notify:
    void setStop(); // call before delete
    void startDataCollection(QString secondaryLogFilename);
    void stopDataCollection();

    void handleNewColorScheme(int scheme, bool useDarkThemeVal);
    void handlePrefs(settingsT prefs);
    void handleGPSConnectionError(int errorNum);
    void handleGPSStatusMessages(QStringList errorMessages,
                                 QStringList warningMessages);
    void cycleGPSStatusMessagesViaTimer();
    void gpsMessageToLogReporterSlot();

    void colorMapScrolledY(const QCPRange &newRange);
    void colorMapScrolledX(const QCPRange &newRange);
    void setScrollX(bool Yenabled);
    void setScrollY(bool Xenabled);
    void updateCeiling(int c);
    void updateFloor(int f);
    void changeRGB(int r, int g, int b);
    void setRGBLevels(double red, double green, double blue, double gamma, bool reprocess);
    void setShowRGBLines(bool showLines);
    void setUseRatioSlot(bool useRatio);
    void changeWFLength(int length);
    void showSecondWF();
    void setWFFPS_render(int target);
    void setWFFPS_primary(int target);
    void setWFFPS_secondary(int target);
    void rescaleRange();
    void setUseDSF(bool useDSF);
    void hideRGB();
    void updateFPS();
    void checkDiskSpace();
    void setCrosshairs(QMouseEvent *event);
    void debugThis();
    // debug text handler:
    void showDebugMessage(QString debugMessage);

private slots:
    void logFPSGPSSlot();

signals:
    void statusMessage(QString);
    void haveGPSErrorWarningMessage(QString);
    void connectToGPS(QString host, int port);
    void beginSecondaryLog(QString filename);
    void stopSecondaryLog();
    void stopWidgets();
    void sendDiskSpaceAvailable(quint64 sizeTotal, quint64 available);
    // For the WF:
    void updateCeilingSignal(int c);
    void updateFloorSignal(int f);
    void updateRGBbandSignal(int r, int g, int b);
    void setRGBLevelsSignal(double r, double g, double b, double gamma, bool);
    void changeWFLengthSignal(int length);
    void setWFFPS_render_sig(int target);

    // For the Controls Box:
    void updateFloorCeilingFromFrameviewChange(double floor, double ceiling);
};

#endif // FLIGHT_WIDGET_H
