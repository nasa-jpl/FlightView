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

class consoleLog : public QWidget
{
    Q_OBJECT
public:
    explicit consoleLog(QWidget *parent = nullptr);

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
    QString createTimeStamp();
    QPlainTextEdit logView;
    QPushButton clearBtn;
    QPushButton annotateBtn;
    QLineEdit annotateText;
    QVBoxLayout layout;
    QHBoxLayout hLayout;
};

#endif // CONSOLELOG_H
