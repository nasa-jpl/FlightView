#include "saveserver.h"

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QNetworkInterface>

saveServer::saveServer(frameWorker *fw, QObject *parent ) :
    QTcpServer(parent)
{
    this->reference = fw;
    port = 65000; // we'll hardcode the port number for now
    clientConnection = new QTcpSocket();
    clientConnection->setObjectName("lv:saveconn");

   // QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();
    // use the first non-localhost IPv4 address
//    for (int i = 0; i < ipAddressesList.size(); ++i) {
//        if (ipAddressesList.at(i) != QHostAddress::LocalHost && ipAddressesList.at(i).toIPv4Address()) {
//            ipAddress = ipAddressesList.at(i);
//            break;
//        }
//    }

    ipAddress = QHostAddress::AnyIPv4; // 0.0.0.0

    // if we did not find one, use IPv4 localhost
    if (ipAddress.isNull())
        ipAddress = QHostAddress::Any;

    connect(clientConnection, SIGNAL(readyRead()), this, SLOT(readCommand()));
    connect(this, SIGNAL(newConnection()), this, SLOT(new_conn_slot())  );
    connect(clientConnection, SIGNAL(disconnected()), this, SLOT(handleDisconnect()));
    signalConnected = true;
    listen(QHostAddress::Any, port); // This will automatically connect on the IP Address of the machine
}

void saveServer::reconnectSignal()
{
    if(!signalConnected)
    {
        connect(clientConnection, SIGNAL(readyRead()), this, SLOT(readCommand()));
        signalConnected = true;
    }
}

void saveServer::disconnectSignal()
{
    if(signalConnected)
    {
        disconnect(clientConnection, SIGNAL(readyRead()), this, SLOT(readCommand()));
        signalConnected = false;
    }
}

void saveServer::handleDisconnect()
{
    genStatusMessage("Remote host disconnected.");
}

void saveServer::incomingConnection(qintptr socketDescriptor)
{
    if(!clientConnection->setSocketDescriptor(socketDescriptor)) {
        genErrorMessage("Client connection refused by host.");
        return;
    }
}

void saveServer::connected_to_client()
{
    genStatusMessage("Called connected()");
}

void saveServer::new_conn_slot()
{
    // This indicates a client has connected.
    genStatusMessage("New remote connection is active.");
}

bool saveServer::checkValues(uint16_t framesToSaveCount,
                             QString filename, uint16_t naverages)
{
    bool ok = true;
    if(framesToSaveCount > 60000)
        ok = false;
    if((filename.length() < 4) or (filename.length() > 4096))
        ok = false;
    if( (naverages > 2000) or (naverages > framesToSaveCount))
        ok = false;
    return ok;
}


void saveServer::readCommand()
{
    readMutex.lock();
    QDataStream in(clientConnection);
    in.setVersion(QDataStream::Qt_4_0);
    blockSize = 0;
    bool readData = true;

    while(readData)
    {
        if (clientConnection->bytesAvailable() < (qint64) sizeof(quint16)) {
            //clientConnection->disconnectFromHost();
            //genStatusMessage("No further data available over tcp/ip.");
            readMutex.unlock();
            readData = false;
            return;
        }
        in >> blockSize;
        //qInfo() << "Block size: " << blockSize;

        if (clientConnection->bytesAvailable() < blockSize) {
            std::stringstream msg;
            msg << "Only " << clientConnection->bytesAvailable() << " bytes to read compared with block of " << blockSize << " bytes." << std::endl;
            genErrorMessage(QString::fromStdString(msg.str()));
            //clientConnection->disconnectFromHost();
            readMutex.unlock();
            return;
        }

        in >> commandType;
        // genStatusMessage("Command received."); // DEBUG message, disable on release build

        switch (commandType) {
        case CMD_START_SAVING:
            genStatusMessage("SAVE command received.");
            if(reference->to.save_framenum == 0)
            {
                in >> framesToSave;
                in >> fname;
                in >> navgs;
                if(checkValues(framesToSave, fname, navgs))
                {
                    reference->navgs = navgs;
                    reference->startSavingRawData(framesToSave, fname, navgs);
                }
            } else {
                genErrorMessage("Received SAVE command while already saving.");
            }
            break;
        case CMD_START_FLIGHT_SAVING:
        {
            genStatusMessage("START_FLIGHT_SAVING command received.");
            // No arguments expected.
            // emit signal to MainWindow which will trigger the automatic recording of data.
            emit startSavingFlightData();
        }
            break;
        case CMD_STATUS:
        {
            genStatusMessage("Sending STATUS information back.");
            QByteArray block;
            QDataStream out( &block, QIODevice::WriteOnly );
            out.setVersion(QDataStream::Qt_4_0);
            out << (uint16_t)0;
            out << (uint16_t)reference->to.save_framenum; // send the number of frames left to save, and
            out << (uint16_t)reference->delta;            // send the frames per second
            out << (uint16_t)reference->navgs;                       // number of averages, new code, bogus
            out.device()->seek(0);
            out << (uint16_t)(block.size() - sizeof(quint16));
            //printHex(&block);
            clientConnection->write(block);
            break;
        }
        case CMD_STOP_SAVING:
        {
            genStatusMessage("Received STOP_SAVING command");
            emit stopSavingData();
            break;
        }
        case CMD_STATUS_EXTENDED:
        {
            genStatusMessage("Sending STATUS_EXTENDED information back.");
            QByteArray block;
            QDataStream out( &block, QIODevice::WriteOnly );
            out.setVersion(QDataStream::Qt_4_0);
            out << (uint16_t)0; // will be changed to the size of the message later.
            out << (uint16_t)CMD_STATUS_EXTENDED; // new addition
            // NOTE: When the total number of frames to be saved is undefined (start/stop recording mode),
            // the value returned is bogus and should not be used.
            out << (uint16_t)reference->to.save_framenum; // send the number of frames left to save, and
            out << (uint16_t)reference->delta;            // send the frames per second (as a uint)
            out << (uint16_t)reference->navgs;            // number of averages, new code, bogus
            if(fname.isEmpty())
                out << QString("");
            else
                out << fname;
            out.device()->seek(0);
            out << (uint16_t)(block.size() - sizeof(quint16));
            //printHex(&block);
            clientConnection->write(block);
            break;
        }
        case CMD_START_DARKSUB:
        {
            genStatusMessage("Client requested CMD_START_DARKSUB, starting dark collection.");
            emit startTakingDarks();
            break;
        }
        case CMD_STOP_DARKSUB:
        {
            genStatusMessage("Client requested CMD_STOP_DARKSUB, stopping dark collection.");
            emit stopTakingDarks();
            break;
        }
        default:
            genErrorMessage("Unknown command received: " + QString("0x%1").arg(commandType, 2, 16, QChar('0')));
            genErrorMessage("Disconnecting remote host now.");
            clientConnection->disconnectFromHost();
            break;
        }
    }
    readMutex.unlock();
}

void saveServer::genErrorMessage(QString errorMessage)
{
    QString msg = "[saveServer]: ERROR: ";
    msg.append(errorMessage);
    std::cout << msg.toStdString() << std::endl;
    emit sigMessage(msg);
}

void saveServer::genStatusMessage(QString statusMessage)
{
    QString msg = "[saveServer]: Status: ";
    msg.append(statusMessage);
    std::cout << msg.toStdString() << std::endl;
    emit sigMessage(msg);
}

void saveServer::printHex(QByteArray *b)
{
    qDebug() << "Begin byte array debug: ";
    QString i;
    QString h;
    for(int p=0; p < b->length(); p++)
    {
        i.append(QString("[%1]").arg(p,2,10,QChar('0')));
        h.append(QString("[%1]").arg((unsigned char)b->at(p), 2, 16, QChar('0')));
    }
    qDebug() << i;
    qDebug() << h;
    qDebug() << "End byte array debug.";
}


