#ifndef SAVESERVER_H
#define SAVESERVER_H

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QDataStream>
#include <QString>
#include <sstream>
#include <QMutex>

#include "frame_worker.h"
#include "flightappstatustypes.h"

const quint16 CMD_START_SAVING = 2;
const quint16 CMD_STATUS = 3;
const quint16 CMD_STATUS_EXTENDED = 4;
const quint16 CMD_START_DARKSUB = 5;
const quint16 CMD_STOP_DARKSUB = 6;
const quint16 CMD_START_FLIGHT_SAVING = 7;
const quint16 CMD_STOP_SAVING = 8;
const quint16 CMD_STATUS_FLIGHT = 9;

/*! \file
 *  \brief Establishes a server which can accept remote frame saving commands.
 *  \paragraph
 *
 *  The saveServer class processes incoming QTcpSockets and accepts messages in Tcp format as a QByteStream.
 *  There are message codes defined as macros in this header, where the command to start saving a finite number
 *  of frames is specified by the message type 2. I didn't use an enum because I wanted the software to be sufficiently
 *  modular that if the client was not aware of these macros they could still send messages using the code. The quint16
 *  type was chosen because Qt offers cross-platform support for their internal types, allowing greater portability of any
 *  potential client software. */

class saveServer : public QTcpServer
{
    Q_OBJECT

    frameWorker *reference;

    QTcpSocket *clientConnection;
    uint16_t blockSize;
    uint16_t commandType;

    uint16_t framesToSave = 0;
    QString fname = "";
    uint16_t navgs = 0;

    flightAppStatus_t *flightStatus = NULL;

    bool stat_diskOk = true;
    bool stat_gpsLinkOk = true;
    bool stat_gpsReady = true;
    bool stat_cameraOk = true;
    bool stat_headerOk = false;
    quint16 stat_framesCaptured = 0;

    QMutex readMutex;
    bool signalConnected = false;
    void reconnectSignal();
    void disconnectSignal();

    bool checkValues(uint16_t framesToSaveCount,
                     QString filename,
                     uint16_t naverages);
    void genErrorMessage(QString errorMessage);
    void genStatusMessage(QString statusMessage);
    void printHex(QByteArray *b);

public:
    saveServer(frameWorker *fw, flightAppStatus_t *flightStatus, QObject *parent = 0);

    QHostAddress ipAddress;
    int port;

protected:
    void incomingConnection(qintptr socketDescriptor);

signals:
    void startSavingRemote(const QString &unverifiedName, unsigned int nFrames, unsigned int numAvgs); // not used
    void startSavingFlightData();
    void stopSavingData();
    void startTakingDarks();
    void stopTakingDarks();
    void sigMessage(QString message);

//public slots:
//    void sendFullStatus(bool stat_diskOk, bool stat_gpsLinkOk,
//                        bool stat_gpsReady, bool stat_cameraOk,
//                        bool stat_headerOk, quint16 stat_framesCaptured);

private slots:
    void readCommand();
    void connected_to_client();
    void new_conn_slot();
    void handleDisconnect();


};

#endif // SAVESERVER_H
