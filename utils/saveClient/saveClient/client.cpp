
#include <QGridLayout>
#include <QMessageBox>
#include <QtNetwork>

#include "client.h"

Client::Client(QWidget *parent) : QDialog(parent), networkSession(0)
{
    fileLabel = new QLabel(tr("&Save Location:"));
    filenameEntry = new QLineEdit;
    filenameEntry->setText( "/" );
    fileLabel->setBuddy(filenameEntry);

    statusLabel = new QLabel(tr("This example requires that you run "
                                "Live View2 as well."));

    connectButton = new QPushButton(tr("Connect"));
    connectButton->setDefault(true);
    connectButton->setEnabled(true);

    quitButton = new QPushButton(tr("Quit"));

    buttonBox = new QDialogButtonBox;
    buttonBox->addButton(connectButton, QDialogButtonBox::ActionRole);
    buttonBox->addButton(quitButton, QDialogButtonBox::RejectRole);

    tcpSocket = new QTcpSocket(this);

    connect(filenameEntry, SIGNAL(textChanged(QString)),
            this, SLOT(enableConnectButton()));
    connect(connectButton, SIGNAL(clicked()),
            this, SLOT(sendMessage()));
    connect(quitButton, SIGNAL(clicked()), this, SLOT(close()));
    connect(tcpSocket, SIGNAL(error(QAbstractSocket::SocketError)),
            this, SLOT(displayError(QAbstractSocket::SocketError)));

    QGridLayout* mainLayout = new QGridLayout;
    mainLayout->addWidget(fileLabel, 0, 0);
    mainLayout->addWidget(filenameEntry, 0, 1);
    mainLayout->addWidget(statusLabel, 1, 0, 1, 2);
    mainLayout->addWidget(buttonBox, 2, 0, 1, 2);
    setLayout(mainLayout);

    setWindowTitle(tr("Frame Capture Client"));
    filenameEntry->setFocus();

    QNetworkConfigurationManager manager;
    if (manager.capabilities() & QNetworkConfigurationManager::NetworkSessionRequired) {
        // Get saved network configuration
        QSettings settings(QSettings::UserScope, QLatin1String("QtProject"));
        settings.beginGroup(QLatin1String("QtNetwork"));
        const QString id = settings.value(QLatin1String("DefaultNetworkConfiguration")).toString();
        settings.endGroup();

        // If the saved network configuration is not currently discovered use the system default
        QNetworkConfiguration config = manager.configurationFromIdentifier(id);
        if ((config.state() & QNetworkConfiguration::Discovered) !=
            QNetworkConfiguration::Discovered) {
            config = manager.defaultConfiguration();
        }

        networkSession = new QNetworkSession(config, this);
        connect(networkSession, SIGNAL(opened()), this, SLOT(sessionOpened()));

        connectButton->setEnabled(false);
        statusLabel->setText(tr("Opening network session."));
        networkSession->open();
    }
}
void Client::sendMessage()
{
    connectButton->setEnabled(false);
    QByteArray block;
    QDataStream out(&block, QIODevice::WriteOnly );
    out.setVersion(QDataStream::Qt_4_0);

    fname = filenameEntry->text();

    quint16 messageType = 2; // start saving raws type
    out << (quint16)0;
    out << messageType;
    out << (quint16)100;
    out << fname;
    out.device()->seek(0);
    out << (quint16)(block.size() - sizeof(quint16));

    // hardcoding
    QHostAddress our_host;
    our_host.setAddress( "10.0.0.1" ); // replace with the ip address of the computer running liveview
    int portNumber = 65000;

    tcpSocket->connectToHost( aviris_cal, portNumber );
    tcpSocket->waitForConnected(10);
    tcpSocket->write(block);
}
void Client::displayError(QAbstractSocket::SocketError socketError)
{
    switch (socketError) {
    case QAbstractSocket::RemoteHostClosedError:
        break;
    case QAbstractSocket::HostNotFoundError:
        QMessageBox::information(this, tr("Frame Capture Client"),
                                 tr("The host was not found. Please check the "
                                    "host name and port settings."));
        break;
    case QAbstractSocket::ConnectionRefusedError:
        QMessageBox::information(this, tr("Frame Capture Client"),
                                 tr("The connection was refused by the peer. "
                                    "Make sure Live View2 is running, "
                                    "and check that the host name and port "
                                    "settings are correct."));
        break;
    default:
        QMessageBox::information(this, tr("Frame Capture Client"),
                                 tr("The following error occurred: %1.")
                                 .arg(tcpSocket->errorString()));
    }

    connectButton->setEnabled(true);
}
void Client::enableConnectButton()
{
    connectButton->setEnabled((!networkSession || networkSession->isOpen()) && !filenameEntry->text().isEmpty() );
}
void Client::sessionOpened()
{
    // Save the used configuration
    QNetworkConfiguration config = networkSession->configuration();
    QString id;
    if (config.type() == QNetworkConfiguration::UserChoice)
        id = networkSession->sessionProperty(QLatin1String("UserChoiceConfiguration")).toString();
    else
        id = config.identifier();

    QSettings settings(QSettings::UserScope, QLatin1String("QtProject"));
    settings.beginGroup(QLatin1String("QtNetwork"));
    settings.setValue(QLatin1String("DefaultNetworkConfiguration"), id);
    settings.endGroup();

    statusLabel->setText(tr("This example requires that you run "
                            "Live View2 as well."));

    enableConnectButton();
}
