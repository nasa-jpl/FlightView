#ifndef SAVESERVER_H
#define SAVESERVER_H

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>

#include "frame_worker.h"

const quint16 START_SAVING = 2;
const quint16 STATUS = 3;

class saveServer : public QTcpServer
{
    Q_OBJECT

public:
    saveServer(frameWorker* fw, QObject* parent = 0);

    quint16 framesToSave;
    QString fname;
    QString ipAddress;
    int port;
    frameWorker* reference;

protected:
    void incomingConnection(int socketDescriptor);

private slots:
    void readCommand();

private:
    QTcpSocket* clientConnection;
    quint16 blockSize;
    quint16 commandType;
};

#endif // SAVESERVER_H
