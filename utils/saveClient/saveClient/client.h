#ifndef CLIENT_H
#define CLIENT_H

#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QtNetwork/QNetworkSession>
#include <QtNetwork/QTcpSocket>

class Client : public QDialog
{
    Q_OBJECT

public:
    Client(QWidget *parent = 0);

private slots:
    void sendMessage();
    void displayError(QAbstractSocket::SocketError socketError);
    void enableConnectButton();
    void sessionOpened();
    void genStatusMessage(QString msg);
    void genErrorMessage(QString msg);
    void printHex(QByteArray *d);

private:
    QLabel* statusLabel;
    QLabel* fileLabel;
    QLineEdit* filenameEntry;
    QDialogButtonBox* buttonBox;
    QPushButton* quitButton;
    QPushButton* connectButton;

    QTcpSocket* tcpSocket;
    QString fname;
    quint16 blockSize;

    QNetworkSession *networkSession;
};

#endif
