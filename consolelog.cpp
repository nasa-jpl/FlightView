#include "consolelog.h"

consoleLog::consoleLog(QWidget *parent) : QWidget(parent)
{
    this->logFileName = "";
    this->enableLogToFile = false;

    this->createUI();
    this->makeConnections();
 }

consoleLog::consoleLog(QString logFileName, QWidget *parent) : QWidget(parent)
{
    this->logFileName = createFilenameFromDirectory(logFileName);
    this->enableLogToFile = true;

    this->createUI();
    this->makeConnections();

    openFile(this->logFileName);
}

consoleLog::~consoleLog()
{
    this->destroyUI();
    this->closeFile();
}

void consoleLog::createUI()
{
    logView.setMinimumHeight(200);
    logView.setMinimumWidth(230);
    logView.setReadOnly(true);
    layout.addWidget(&logView);
    layout.addItem(&hLayout);
    annotateText.setMinimumWidth(200);
    annotateBtn.setMaximumWidth(75);
    annotateBtn.setText("Annotate");
    annotateBtn.setToolTip("Press to annotate the log");
    annotateBtn.setDefault(true);
    annotateBtn.setAutoDefault(true);
    hLayout.addWidget(&annotateText);
    hLayout.addWidget(&annotateBtn);
    this->setLayout(&layout);

    this->setWindowTitle("FlightView Console Log");
    this->resize(500, 200);

    logView.scroll(0,200);
}

void consoleLog::makeConnections()
{
    connect(&annotateBtn, SIGNAL(pressed()), this, SLOT(onAnnotateBtnPushed()));
    connect(&clearBtn, SIGNAL(pressed()), this, SLOT(onClearBtnPushed()));
    connect(&annotateText, SIGNAL(returnPressed()), this, SLOT(onAnnotateBtnPushed()));
}

void consoleLog::destroyUI()
{

}

void consoleLog::writeToFile(QString textOut)
{
    if(fileIsOpen)
    {
        textOut.append("\n");
        outfile.write(textOut.toLocal8Bit().data(), textOut.length());
        if ( (outfile.rdstate() & std::ifstream::failbit ) != 0 )
        {
            handleError("ERROR: Could not write text to file!");
        }
        outfile.flush();
    } else {
        handleError("Log file is not open, text not being logged.");
    }
    return;
}

void consoleLog::openFile(QString filename)
{
    if(fileIsOpen)
    {
        // indicate error because file is already open
        handleError("ConsoleLog was already opened, tried to open again.");
        return;
    } else {
        if(filename.isEmpty())
        {
            handleError("ConsoleLog filename is empty. Cannot log to file.");
            return;
        }
        this->logFileName = filename;
        outfile.open(filename.toLocal8Bit().data(),  std::ios_base::app); // append
        if ( (outfile.rdstate() & std::ifstream::failbit ) != 0 )
        {
            handleError(QString("ConsoleLog file [%1] could not be opened for appending.").arg(filename));
            fileIsOpen = false;
            return;
        } else {
            fileIsOpen = true;
            insertText(QString("[ConsoleLog]: Opened file [%1] for logging.").arg(filename));
        }
    }

    return;
}

void consoleLog::closeFile()
{
    if(fileIsOpen)
    {
        outfile.flush();
        outfile.close();
    } else {
        handleError(QString("ConsoleLog file [%1] could not be closed"));
    }
}

void consoleLog::onClearBtnPushed()
{
    logView.clear();
}

void consoleLog::onAnnotateBtnPushed()
{
    if(annotateText.text().isEmpty())
        return;
    insertText("[Operator Annotation]: " + annotateText.text());
    annotateText.clear();
}

void consoleLog::insertText(QString text)
{
    insertTextNoTagging(createTimeStamp() + text);
}

void consoleLog::insertTextNoTagging(QString text)
{
    logView.appendPlainText(text);
    if(enableLogToFile)
        writeToFile(text);
}

QString consoleLog::createTimeStamp()
{
    QDateTime now = QDateTime::currentDateTimeUtc();
    QString dateString = now.toString("[yyyy-MM-dd hh:mm:ss]: ");

    return dateString;
}

QString consoleLog::createFilenameFromDirectory(QString directoryName)
{
    QString filename;
    if(directoryName.isEmpty())
    {
        handleError("ERROR: Directory name supplied for logging is empty. Using /tmp.");
        directoryName = QString("/tmp/");
    }
    if(!directoryName.endsWith("/"))
        directoryName.append("/");

    makeDirectory(directoryName);

    QDateTime now = QDateTime::currentDateTimeUtc();
    QString dateString = now.toString("yyyy-MM-dd_hhmmss");

    filename = directoryName + dateString + "-FlightView.log";
    return filename;
}

void consoleLog::makeDirectory(QString directoryName)
{
    QDir dir(directoryName);
    if (!dir.exists())
    {
        // create directory
        handleNote(QString("Creating directory [%1]").arg(directoryName));
        dir.mkpath(".");
    }

    // Check our work:
    if(!dir.exists())
    {
        handleError(QString("Could not create directory [%1]").arg(directoryName));
    }
    return;
}

void consoleLog::handleError(QString errorText)
{
    // This function is for errors *within* the consoleLog class.
    // These errors are not logged to the file since they are likely
    // caused by not beign able to log errors.

    errorText.prepend("[ConsoleLog ERROR]: ");
    logView.appendPlainText(errorText);
    std::cout << errorText.toLocal8Bit().data() << std::endl;
}

void consoleLog::handleWarning(QString warningText)
{
    // This function is for warnings *within* the consoleLog class.
    // These warnings are not logged to the file since they are likely
    // caused by not beign able to log or setting up the log.

    warningText.prepend("[ConsoleLog WARNING]: ");
    logView.appendPlainText(warningText);
    std::cout << warningText.toLocal8Bit().data() << std::endl;
}

void consoleLog::handleNote(QString noteText)
{
    // This function is for notes *within* the consoleLog class.
    // These notes are not logged to the file since they are likely
    // caused by not beign able to log or setting up the log.

    noteText.prepend("[ConsoleLog NOTE]: ");
    logView.appendPlainText(noteText);
    std::cout << noteText.toLocal8Bit().data() << std::endl;
}
