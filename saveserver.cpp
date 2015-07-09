#include "saveserver.h"

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QNetworkInterface>
/*#define QDEBUG */

saveServer::saveServer(frameWorker* fw, QObject* parent ) : QTcpServer(parent)
{
    port = 65000; // we'll hardcode the port number for now
    clientConnection = new QTcpSocket();

    QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();
    // use the first non-localhost IPv4 address
    for( int i = 0; i < ipAddressesList.size(); ++i )
    {
        if( ipAddressesList.at(i) != QHostAddress::LocalHost &&
            ipAddressesList.at(i).toIPv4Address() )
        {
            ipAddress = ipAddressesList.at(i).toString();
            break;
        }
    }

    // if we did not find one, use IPv4 localhost
    if( ipAddress.isEmpty() )
        ipAddress = QHostAddress( QHostAddress::LocalHost ).toString();

    connect(clientConnection,SIGNAL(readyRead()),this,SLOT(readCommand()));

    this->reference = fw;

    listen( QHostAddress::Any, port ); // This will automatically connect on the IP Address of aviris-cal
}

void saveServer::incomingConnection( int socketDescriptor )
{
    if( !clientConnection->setSocketDescriptor( socketDescriptor ) )
    {
#ifdef QDEBUG
        std::cout << "Client Connection refused by host! :(" << std::endl;
        return;
#endif
    }
}

void saveServer::readCommand()
{
    QDataStream in(clientConnection);
    in.setVersion(QDataStream::Qt_4_0);

    if( blockSize == 0 )
    {
        if( clientConnection->bytesAvailable() < (int)sizeof(quint16) )
        {
            clientConnection->disconnectFromHost();
#ifdef QDEBUG
            std::cout << "No data received..." << std::endl;
#endif
            return;
        }

        in >> blockSize;
    }

    if( clientConnection->bytesAvailable() < blockSize )
    {
        clientConnection->disconnectFromHost();
        return;
    }

    in >> commandType;
    in >> framesToSave;
    in >> fname;
#ifdef QDEBUG
    std::cout << "Command received!" << std::endl;
#endif

    switch(commandType)
    {
    case START_SAVING:
        reference->to.startSavingRaws( fname.toStdString(), (uint)framesToSave );
        break;
    case STATUS:
        QByteArray block;
        QDataStream out( &block, QIODevice::WriteOnly );
        out << (quint16)0;
        out << (quint16)reference->to.save_framenum; // send the number of frames left to save, and
        out << (quint16)reference->delta;            // send the frames per second
        out << (quint16)(block.size() - sizeof(quint16));
        out.device()->seek(0);
        clientConnection->write(block);
        break;
    }
    blockSize = 0;
    clientConnection->disconnectFromHost();
}
