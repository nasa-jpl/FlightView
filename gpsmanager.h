#ifndef GPSMANAGER_H
#define GPSMANAGER_H

#include <QObject>
#include <QLabel>
#include "qcustomplot.h"

#include "filenamegenerator.h"

#include "gpsGUI/qfi/qfi_EADI.h"
#include "gpsGUI/qfi/qfi_EHSI.h"
#include "gpsGUI/qfi/qfi_ASI.h"
#include "gpsGUI/qfi/qfi_VSI.h"

#include "gpsGUI/gpsnetwork.h"
#include "gpsGUI/qledlabel.h"

struct utcTime {
    int hour;
    int minute;
    int second;
    float secondFloat;
    QString UTCstr;
    QString UTCValidityStr;
};



class gpsManager : public QObject
{
    Q_OBJECT

    const QString name = "lv:";

    QThread *gpsThread;
    gpsNetwork *gps;
    gpsMessage m;

    fileNameGenerator filenamegen;

    QString baseSaveDirectory;
    QString gpsRecordingBinaryLogFilename;
    bool createLoggingDirectory();

    QTimer gpsReconnectTimer;
    QString host;
    int port;
    QString gpsBinaryLogFilename;

    // Update Frequency:
    // data come in at 200 Hz
    unsigned char updateLabelsInterval = 10;
    unsigned char updatePlotsInverval = 90;
    unsigned char updateAvionicsWidgetsInterval = 10;


    // Status:
    bool statusGNSSReceptionOk = true; // GNSS Info received on time or way too late
    bool statusGNSSReceptionWarning = false; // GNSS Info received late

    bool statusMessageDecodeOk = true; // message decoded ok
    bool statusConnectedToGPS = true; // Connected at this time to the GPS unit
    bool statusGPSHeartbeatOk = true; // Have received messages recently
    bool statusGPSMessagesDropped = false; // true if messages being dropped

    bool statusLinkStickyError = false; // stays true until cleared
    bool statusTroubleStickyError = false; //

    bool statusJustCleared = false; // True if we just cleared errors.

    // Individual "now" errors which are made sticky by the process function:

    // Algorithm Status 1:
    bool navPhase = false; // Error or warning? Means nav ready, kalman filter ready
    bool gpsReceived = false; // warning
    bool gpsValid = false; // Error
    bool gpsWaiting = true; // warning
    bool gpsRejected = false; // Error
    bool altitudeSaturation = false; // Error
    bool speedSaturation = false; // Error
    bool interpolationMissed = false; // Error

    // Algorithm Status 2:
    bool altitudeRejected = false; // Error
    bool zuptActive = false; // Error
    bool zuptOther = false; // Error, includes Valid, RotationMode, and RoValid

    // Algorithm Status 3:
    bool flashWriteError = false; // Error
    bool flashEraseError = false; // Error

    // INS System Status 1:
    bool outputAFull = false; // Error
    bool outputBFull = false; // Error

    // INS System Status 2:
    bool gpsDetected = false; // warning

    // INS System Status 3:
    bool systemReady = false; // warning

    // Status Bits from the device:
    // INS Algorithm Status
    dword priorAlgorithmStatus1=0;
    dword priorAlgorithmStatus2=0;
    dword priorAlgorithmStatus3=0;
    dword priorAlgorithmStatus4=0;

    // INS System Status
    dword priorSystemStatus1=0;
    dword priorSystemStatus2=0;
    dword priorSystemStatus3=0;

    int consecutiveDecodeErrors = 0;

    gpsQualityKinds gnssQualPrior = gpsQualityInvalid;
    QString gnssQualStr = "";
    QString gnssQualShortStr = "";
    bool gnssAlignmentComplete = false;
    QString gnssAlignmentPhase = "";

    // Record 1 out of every 40 points for 5 Hz updates to plots
    // therefore, for 90 seconds of data, we need 90*5 = 450 point vectors
    uint16_t vecSize = 450;
    uint16_t vecPosAltHeading = 0;
    uint16_t vecPosPosition = 0;
    uint16_t vecPosSpeed = 0;

    void prepareVectors(); // resize
    void prepareLEDs(); // initial state
    void prepareLabels(); // initial state
    void prepareGPS(); // thread connections
    unsigned char getBit(dword d, unsigned char bit);

    uint16_t msgsReceivedCount = 0;

    // Heading
    QVector<double> headingsMagnetic;
    QVector<double> headingsCourse;

    // Roll and Pitch:
    QVector<double> rolls;
    QVector<double> pitches;

    // Position
    QVector<double> lats;
    QVector<double> longs;
    QVector<double> alts;

    // Speed
    QVector<double> nVelos;
    QVector<double> eVelos;
    QVector<double> upVelos;

    // Time axis:
    QVector<double> timeAxis;

    void updateUIelements();
    void preparePlots();
    void updatePlots();
    void setTimeAxis(QCPAxis *x);
    void setPlotTitle(QCustomPlot *p, QCPPlotTitle *t, QString title);

    void setPlotColors(QCustomPlot *p, bool dark);

    QTimer gpsMessageHeartbeat;
    unsigned int hbErrorCount=0;

    QElapsedTimer gnssStatusTime;
    bool firstMessage;

    void showStatusMessage(QString);
    void processStatus();

    utcTime processUTCstamp(uint64_t t);
    utcTime currentTime;
    utcTime validityTime;

    void updateLabel(QLabel *label, QString text);
    void updateLED(QLedLabel *led, QLedLabel::State s);

    // UI elements (set to NULL if unused):
    QLedLabel *gpsLinkLED = NULL;
    QLedLabel *gpsTroubleLED = NULL;

    QCustomPlot *plotRollPitch = NULL;
    QCustomPlot *plotHeading = NULL;
    QCPPlotTitle *titleHeading = NULL;

    bool usePlotTitle = false;
    QCPPlotTitle *titleRollPitch = NULL;
    QLabel *gpsLat = NULL;
    QLabel *gpsLong = NULL;
    QLabel *gpsAltitude = NULL;
    QLabel *gpsUTCtime = NULL;
    QLabel *gpsUTCdate = NULL;
    QLabel *gpsUTCValidity = NULL;
    QLabel *gpsGroundSpeed = NULL;
    QLabel *gpsRateClimb = NULL;
    QLabel *gpsHeading = NULL;
    QLabel *gpsRoll = NULL;
    QLabel *gpsPitch = NULL;
    QLabel *gpsQuality = NULL;
    QLabel *gpsAlignment = NULL;

    // Avionics Widgets:
    qfi_ASI *asi;
    qfi_VSI *vsi;
    qfi_EADI *eadi;
    qfi_EHSI *ehsi;

public:
    gpsManager();
    ~gpsManager();

    void insertLEDs(QLedLabel *gpsLinkLED, QLedLabel *gpsTroubleLED);
    void insertPlots(QCustomPlot *gpsRollPitchPlot, QCustomPlot *gpsHeadingPlot);
    void insertLabels(QLabel *gpsLat, QLabel *gpsLong, QLabel *gpsAltitude,
                      QLabel *gpsUTCtime, QLabel *gpsUTCdate, QLabel *gpsUTCValidity,
                      QLabel *gpsGroundSpeed,
                      QLabel *gpsHeading, QLabel *gpsRoll, QLabel *gpsPitch,
                      QLabel *gpsQuality, QLabel *gpsAlignment,
                      QLabel *gpsRateClimb);
    void insertAvionicsWidgets(qfi_ASI *asi, qfi_VSI *vsi,
                               qfi_EADI *eadi, qfi_EHSI *ehsi);

    void prepareElements();

public slots:
    void initiateGPSConnection(QString host, int port, QString gpsBinaryLogFilename);
    void initiateGPSDisconnect();
    void handleGPSReconnectTimer();
    void handleStartsecondaryLog(QString filename);
    void handleStopSecondaryLog();
    void receiveGPSMessage(gpsMessage m); // from GPS network thread
    void clearStickyError();
    void setPlotTheme(bool isDark);

signals:
    void connectToGPS(QString host, int port, QString gpsBinaryLogFilename);
    void startSecondaryLog(QString filename);
    void gpsStatusMessage(QString statusMessage);
    void gpsConnectionError(int errorNumber);
    void stopSecondaryLog();
    void disconnectFromGPS();
    void getDebugInfo();

private slots:
    void handleGPSStatusMessage(QString message);
    void handleGPSDataString(QString gpsString);
    void handleGPSConnectionError(int error);
    void handleGPSConnectionGood();
    void handleGPSTimeout();


};

#endif // GPSMANAGER_H
