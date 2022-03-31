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

#include <fstream>
#include <iostream>

class consoleLog : public QWidget
{
    Q_OBJECT
public:
    explicit consoleLog(QWidget *parent = nullptr);
    explicit consoleLog(QString logFileName, QWidget *parent = nullptr);
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
    bool fileIsOpen = false;
    bool enableLogToFile = false;
    std::ofstream outfile;
    QString logFileName = "";
    QString createTimeStamp();
    QString createFilenameFromDirectory(QString directoryName);
    QPlainTextEdit logView;
    QPushButton clearBtn;
    QPushButton annotateBtn;
    QLineEdit annotateText;
    QVBoxLayout layout;
    QHBoxLayout hLayout;
    void handleError(QString errorText);
};

#endif // CONSOLELOG_H
