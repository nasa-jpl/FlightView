#include "consolelog.h"

consoleLog::consoleLog(QWidget *parent) : QWidget(parent)
{
    createUI();
    this->setWindowTitle("FlightView Console Log");
    this->resize(500, 200);
    connect(&annotateBtn, SIGNAL(pressed()), this, SLOT(onAnnotateBtnPushed()));
    connect(&clearBtn, SIGNAL(pressed()), this, SLOT(onClearBtnPushed()));
}

void consoleLog::createUI()
{
    logView.setMinimumHeight(200);
    logView.setMinimumWidth(230);
    layout.addWidget(&logView);
    layout.addItem(&hLayout);
    annotateText.setMinimumWidth(200);
    annotateBtn.setMaximumWidth(75);
    annotateBtn.setText("Annotate");
    annotateBtn.setToolTip("Press to annotate the log");
    hLayout.addWidget(&annotateText);
    hLayout.addWidget(&annotateBtn);
    this->setLayout(&layout);
}

void consoleLog::destroyUI()
{

}

void consoleLog::onClearBtnPushed()
{
    logView.clear();
}

void consoleLog::onAnnotateBtnPushed()
{
    if(annotateText.text().isEmpty())
        return;
    insertText(annotateText.text());
    annotateText.clear();
}

void consoleLog::insertText(QString text)
{
    insertTextNoTagging(createTimeStamp() + text);
}

void consoleLog::insertTextNoTagging(QString text)
{
    logView.appendPlainText(text);
    /*
     * Aparently, the line edit already does this...
    if (text.endsWith("\n"))
    {
        logView.appendPlainText(text);
    } else {
        logView.appendPlainText(text.append("\n"));
    }
    */
}

QString consoleLog::createTimeStamp()
{
    QDateTime now = QDateTime::currentDateTimeUtc();
    QString dateString = now.toString("[yyyy-MM-dd hh:mm:ss]: ");

    return dateString;
}
