#ifndef CONSOLELOG_H
#define CONSOLELOG_H

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QLineEdit>
#include <QString>
#include <QDateTime>
#include <QDir>

#include <fstream>
#include <iostream>
#include <ostream>
#include <sys/utsname.h>
#include <stdlib.h>
#include <thread>

#include "startupOptions.h"
#include "udpbinarylogger.h"
#include "linebuffer.h"

class consoleLog : public QWidget
{
    Q_OBJECT
public:
    explicit consoleLog(startupOptionsType options, QWidget *parent = nullptr);
    explicit consoleLog(startupOptionsType options, QString logFileName, bool enableFlightMode, QWidget *parent = nullptr);
    ~consoleLog();

public slots:
    void insertText(QString text);
    void insertTextNoTagging(QString text);

signals:
    void haveLogText(QString completeLogText);

private slots:
    void onClearBtnPushed();
    void onAnnotateBtnPushed();

private:
    void createUI();
    void destroyUI();
    void makeConnections();
    void writeToFile(QString textOut);
    void openFile(QString filename);
    void closeFile();
    void makeDirectory(QString directory);
    void logSystemConfig();
    std::thread udpThread;
    bool usingUDP = false;
    bool udpThreadRunning = false;
    lineBuffer* buffer = NULL;
    udpbinarylogger *udp = NULL;
    bool fileIsOpen = false;
    bool enableLogToFile = false;
    bool enableUDPLogging = true;
    bool flightMode = false;
    std::ofstream outfile;
    startupOptionsType options;
    QString logFileName = "";
    QString createTimeStamp();
    QString createFilenameFromDirectory(QString directoryName);
    QPlainTextEdit logView;
    QPushButton clearBtn;
    QPushButton annotateBtn;
    QLineEdit annotateText;
    QVBoxLayout layout;
    QHBoxLayout hLayout;
    void logToUDPBuffer(QString text);
    void handleOwnText(QString message);
    void handleError(QString errorText);
    void handleNote(QString noteText);
    void handleWarning(QString warningText);
    const char msgInitSequence[4] =   {0x4a,0x50,0x4c,0x00};
    const char msgReplySequence[37] = {0x48,0x65,0x6c,0x6c,
                                       0x6f,0x20,0x66,0x72,
                                       0x6f,0x6d,0x20,0x74,
                                       0x68,0x65,0x20,0x41,
                                       0x56,0x49,0x52,0x49,
                                       0x53,0x20,0x4c,0x61,
                                       0x62,0x20,0x61,0x74,
                                       0x20,0x4a,0x50,0x4c,
                                       0x21,0x00,0x00,0x00};
};

#endif // CONSOLELOG_H
