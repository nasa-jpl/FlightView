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
    bool statusUTCok = true; // UTC received on time
    bool statusMessageDecodeOk = true; // message decoded ok
    bool statusGPSConnectionNoErrors = true; // Have not received any errors from the network socket
    bool statusConnectedToGPS = true; // Connected at this time to the GPS unit
    bool statusGPSHeartbeatOk = true; // Have received messages recently
    bool statusGPSMessagesDropped = false; // true if messages being dropped
    bool statusStickyError = false; // stays true until cleared
    bool statusJustCleared = false; // True if we just cleared errors.

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

    uint16_t msgsReceivedCount = 0;

    // alt and heading
    QVector<double> headings;
    QVector<double> rolls;
    QVector<double> pitches;

    // position
    QVector<double> lats;
    QVector<double> longs;
    QVector<double> alts;

    // speed
    QVector<double> nVelos;
    QVector<double> eVelos;
    QVector<double> upVelos;

    // time axis:
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

    // UI elements (set to NULL if unused):
    QLedLabel *gpsOkLED = NULL;
    QCustomPlot *plotRollPitch = NULL;
    QCPPlotTitle *titleLatLong = NULL;
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

    // Avionics Widgets:
    qfi_ASI *asi;
    qfi_VSI *vsi;
    qfi_EADI *eadi;
    qfi_EHSI *ehsi;

public:
    gpsManager();
    ~gpsManager();

    void insertLEDs(QLedLabel *gpsOkLED);
    void insertPlots(QCustomPlot *gpsRollPitchPlot);
    void insertLabels(QLabel *gpsLat, QLabel *gpsLong, QLabel *gpsAltitude,
                      QLabel *gpsUTCtime, QLabel *gpsUTCdate, QLabel *gpsUTCValidity,
                      QLabel *gpsGroundSpeed,
                      QLabel *gpsHeading, QLabel *gpsRoll, QLabel *gpsPitch,
                      QLabel *gpsQuality,
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
