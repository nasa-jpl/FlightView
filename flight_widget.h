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

#include <qcustomplot.h>

#include "gpsmanager.h"

#include "frame_worker.h"
#include "frameview_widget.h"
#include "qledlabel.h"
#include "startupOptions.h"
#include "waterfall.h"
#include "preferences.h"

#include "qfi/qfi_EADI.h"

class flight_widget : public QWidget
{
    Q_OBJECT

    frameWorker *fw;

    waterfall *waterfall_widget;
    //frameview_widget *waterfall_widget;
    frameview_widget *dsf_widget;

    gpsManager *gps;
    QString primaryGPSLogLocation; // directory, filename is automatic for primary log
    QString gpsHostname;
    uint16_t gpsPort;
    bool startedPrimaryGPSLog = false;
    startupOptionsType options;
    fs::space_info diskSpace;
    QTimer *diskCheckerTimer;
    QTimer hideRGBTimer;

    QSplitter lrSplitter;
    QSplitter rhSplitter;
    QGridLayout layout;
    QVBoxLayout rhLayout;
    QGroupBox flightControls; // All plots and controls are inside this
    QGridLayout flightControlLayout; // Labels and buttons
    QGridLayout flightPlotsLayout;
    QGridLayout flightAvionicsLayout;

    // stand-in items for flight controls:
    QPushButton resetStickyErrorsBtn;
    QLabel gpsLEDLabel;
    QLedLabel gpsLED;
    QLabel cameraLinkLEDLabel;
    QLedLabel cameraLinkLED;
    QLabel aircraftLbl;
    QLabel diskLEDLabel;
    QLedLabel diskLED;

    // GPS Text Labels:
    QLabel gpsLatText;
    QLabel gpsLatData;
    QLabel gpsLongText;
    QLabel gpsLongData;
    QLabel gpsHeadingText;
    QLabel gpsHeadingData;
    QLabel gpsAltitudeText;
    QLabel gpsAltitudeData;
    QLabel gpsUTCtimeData, gpsUTCdateData, gpsUTCValidityData;
    QLabel gpsUTCtimeText, gpsUTCdateText, gpsUTCValidityText;
    QLabel gpsGroundSpeedData, gpsGroundSpeedText;
    QLabel gpsQualityData, gpsQualityText;


    // GPS Plots:
    QCustomPlot gpsPitchRollPlot;
    QCustomPlot gpsHeadingPlot;

    // GPS Widgets:
    bool useAvionicsWidgets = false;
    qfi_EADI *EADI = NULL;
    qfi_EHSI *EHSI = NULL;
    qfi_ASI *ASI = NULL;
    qfi_VSI *VSI = NULL;

    uint16_t redRow;
    uint16_t greenRow;
    uint16_t blueRow;

    settingsT prefs;
    bool havePrefs = false;

    bool stickyFPSError;
    bool stickyDiskFull;
    int FPSErrorCounter;
    void processFPSError();


public:
    explicit flight_widget(frameWorker *fw, startupOptionsType options, QWidget *parent = nullptr);
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
    void startDataCollection(QString secondaryLogFilename);
    void stopDataCollection();

    void handleNewColorScheme(int scheme, bool useDarkThemeVal);
    void handlePrefs(settingsT prefs);
    void handleGPSConnectionError(int errorNum);
    void colorMapScrolledY(const QCPRange &newRange);
    void colorMapScrolledX(const QCPRange &newRange);
    void setScrollX(bool Yenabled);
    void setScrollY(bool Xenabled);
    void updateCeiling(int c);
    void updateFloor(int f);
    void changeRGB(int r, int g, int b);
    void changeWFLength(int length);
    void rescaleRange();
    void setUseDSF(bool useDSF);
    void hideRGB();
    void updateFPS();
    void checkDiskSpace();
    void setCrosshairs(QMouseEvent *event);
    void debugThis();
    // debug text handler:
    void showDebugMessage(QString debugMessage);

signals:
    void statusMessage(QString);
    void connectToGPS(QString host, int port);
    void beginSecondaryLog(QString filename);
    void stopSecondaryLog();
    void sendDiskSpaceAvailable(quint64 sizeTotal, quint64 available);
};

#endif // FLIGHT_WIDGET_H
