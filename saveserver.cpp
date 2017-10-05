#include "saveserver.h"

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QNetworkInterface>

saveServer::saveServer(frameWorker *fw, QObject *parent ) :
    QTcpServer(parent)
{
    this->reference = fw;
    port = 65000; // we'll hardcode the port number for now
    clientConnection = new QTcpSocket();

    QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();
    // use the first non-localhost IPv4 address
    for (int i = 0; i < ipAddressesList.size(); ++i) {
        if (ipAddressesList.at(i) != QHostAddress::LocalHost && ipAddressesList.at(i).toIPv4Address()) {
            ipAddress = ipAddressesList.at(i);
            break;
        }
    }

    // if we did not find one, use IPv4 localhost
    if (ipAddress.isNull())
        ipAddress = QHostAddress::LocalHost;

    connect(clientConnection, SIGNAL(readyRead()), this, SLOT(readCommand()));
    connect(this, SIGNAL(newConnection()), this, SLOT(new_conn_slot())  );
    listen(QHostAddress::Any, port); // This will automatically connect on the IP Address of aviris-cal
}

void saveServer::incomingConnection(qintptr socketDescriptor)
{
    std::cout << "incomingConnection() called!\n";
    if(!clientConnection->setSocketDescriptor(socketDescriptor)) {
        std::cout << "Client Connection refused by host! :(" << std::endl;
        return;
    }
}

void saveServer::connected_to_client()
{
    std::cout << "called connected()\n";
}

void saveServer::new_conn_slot()
{
    std::cout << "new_conn_slot(): A new TCP connection exists.\n";
    // does not work:
    // incomingConnection(this->socketDescriptor());

    // Gets to readCommand(), pretty cool, seg faults after that.
    // Because, clientConnection is set to 0x0
    // clientConnection = nextPendingConnection();
    // clientConnection->setSocketDescriptor(socketDescriptor());
}


void saveServer::readCommand()
{
    std::cout << "in readCommand()" << std::endl;

    QDataStream in(clientConnection);
    in.setVersion(QDataStream::Qt_4_0);
    blockSize = 0;
    if (blockSize == 0) {
        if (clientConnection->bytesAvailable() < (qint64) sizeof(quint16)) {
            clientConnection->disconnectFromHost();
            std::cout << "No data received..." << std::endl;
            return;
        }
        in >> blockSize;
    }

    if (clientConnection->bytesAvailable() < blockSize) {
        std::cout << "Only " << clientConnection->bytesAvailable() << " bytes to read compared with block of " << blockSize << " bytes." << std::endl;
        clientConnection->disconnectFromHost();
        return;
    }

    in >> commandType;
    std::cout << "Command received!" << std::endl;

    switch (commandType) {
    case START_SAVING:
        in >> framesToSave;
        in >> fname;
        in >> navgs;
        reference->navgs = navgs;
        reference->startSavingRawData(framesToSave, fname, navgs);
        break;
    case STATUS:
        std::cout << "Sending back information!" << std::endl;
        QByteArray block;
        QDataStream out( &block, QIODevice::WriteOnly );
        out.setVersion(QDataStream::Qt_4_0);
        out << (uint16_t)0;
        out << (uint16_t)reference->to.save_framenum; // send the number of frames left to save, and
        out << (uint16_t)reference->delta;            // send the frames per second
        out << (uint16_t)reference->navgs;                       // number of averages, new code, bogus
        out.device()->seek(0);
        out << (uint16_t)(block.size() - sizeof(quint16));

        clientConnection->write(block);
        break;
    }
    blockSize = 0;
    clientConnection->disconnectFromHost();
}
