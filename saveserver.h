#ifndef SAVESERVER_H
#define SAVESERVER_H

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QDataStream>
#include <QString>
#include <sstream>
#include <QMutex>

#include "frame_worker.h"

const quint16 CMD_START_SAVING = 2;
const quint16 CMD_STATUS = 3;
const quint16 CMD_STATUS_EXTENDED = 4;
const quint16 CMD_START_DARKSUB = 5;
const quint16 CMD_STOP_DARKSUB = 6;

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

    uint16_t framesToSave;
    QString fname;
    uint16_t navgs;

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
    saveServer(frameWorker *fw, QObject *parent = 0);

    QHostAddress ipAddress;
    int port;

protected:
    void incomingConnection(qintptr socketDescriptor);

signals:
    void startSavingRemote(const QString &unverifiedName, unsigned int nFrames, unsigned int numAvgs); // not used
    void startSavingDarks();
    void stopSavingDarks();
    void sigMessage(QString message);

private slots:
    void readCommand();
    void connected_to_client();
    void new_conn_slot();
    void handleDisconnect();


};

#endif // SAVESERVER_H
