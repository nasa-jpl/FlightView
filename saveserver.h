#ifndef SAVESERVER_H
#define SAVESERVER_H

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>

#include "frame_worker.h"

const quint16 START_SAVING = 2;
const quint16 STATUS = 3;

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

public:
    saveServer(frameWorker *fw, QObject *parent = 0);

    uint16_t framesToSave;
    QString fname;
    QHostAddress ipAddress;
    int port;
    frameWorker *reference;

protected:
    void incomingConnection(int socketDescriptor);

private slots:
    void readCommand();

private:
    QTcpSocket *clientConnection;
    uint16_t blockSize;
    uint16_t commandType;
};

#endif // SAVESERVER_H
