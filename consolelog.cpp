#include "consolelog.h"

consoleLog::consoleLog(QWidget *parent) : QWidget(parent)
{
    this->logFileName = "";
    this->enableLogToFile = false;

    this->createUI();
    this->makeConnections();
    insertText(QString("[ConsoleLog]: Warning: Not logging text to a file."));

    this->logSystemConfig();
 }

consoleLog::consoleLog(QString logFileName, bool enableFlightMode, QWidget *parent) : QWidget(parent)
{
    this->flightMode = enableFlightMode;
    this->logFileName = createFilenameFromDirectory(logFileName);
    this->enableLogToFile = true;

    this->createUI();
    this->makeConnections();

    openFile(this->logFileName);
    this->logSystemConfig();
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
    QFont font("monospace");
    font.setStyleHint(QFont::Monospace);
    logView.setFont(font);
    layout.addWidget(&logView);
    layout.addItem(&hLayout);
    annotateText.setMinimumWidth(200);
    annotateText.setFont(font);
    annotateBtn.setMaximumWidth(75);
    annotateBtn.setText("Annotate");
    annotateBtn.setToolTip("Press to annotate the log");
    annotateBtn.setDefault(true);
    annotateBtn.setAutoDefault(true);
    hLayout.addWidget(&annotateText);
    hLayout.addWidget(&annotateBtn);
    this->setLayout(&layout);

    this->setWindowTitle("FlightView Console Log");
    this->resize(1156, 512);

    logView.setFocusPolicy(Qt::NoFocus);
    annotateBtn.setFocusPolicy(Qt::NoFocus);
    logView.scroll(0,200);
    annotateText.setFocus();
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
    if(annotateText.text() == QString(msgInitSequence))
        insertText(msgReplySequence);
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
    QString dateString;
    QString namePrefix = "AV3";
    if(flightMode) {
        dateString.append(namePrefix);
        dateString.append(now.toString("yyyyMMdd"));
        dateString.append("t"); // t = "timezone" in datetime, so we must append it this way.
        dateString.append(now.toString("hhmmss"));
    } else {
        dateString = now.toString("yyyy-MM-dd_hhmmss");
    }
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

void consoleLog::logSystemConfig()
{
    struct utsname info;
    int rtnval = 0;
    rtnval = uname(&info);
    if(rtnval)
        return;
    handleOwnText(QString("Sysname: %1").arg(info.sysname));
    handleOwnText(QString("Hostname: %1").arg(info.nodename));
    handleOwnText(QString("Kernel release: %1").arg(info.release));
    handleOwnText(QString("Kernel version: %1").arg(info.version));
    handleOwnText(QString("Machine Type: %1").arg(info.machine));
#ifdef _GNU_SOURCE
    handleOwnText(QString("Domainname: %1").arg(info.domainname));
#endif

    // Distribution name:
    FILE *fp;
    char lsbInfo[1024] = {'\0'};
    QString infoStr;
    fp = popen("/usr/bin/lsb_release -d", "r");
    if(fp==NULL)
    {
        handleOwnText("Could not determine lsb_release");
        return;
    }
    if(fgets(lsbInfo, sizeof(lsbInfo), fp))
    {
        infoStr = QString(lsbInfo);
        infoStr.replace(QString("\t"), QString(" ")).replace("\n", "");
        handleOwnText(QString("Linux LSB %1").arg(infoStr));
    } else {
        handleOwnText("Could not determine lsb_release");
    }
    pclose(fp);

    handleOwnText(QString("Compiled against Qt version: %1").arg(QT_VERSION_STR));
    if(QString(GIT_CURRENT_SHA1_SHORT).isEmpty())
        return;

    handleOwnText(QString("Git short SHA1: %1").arg(GIT_CURRENT_SHA1_SHORT));
    handleOwnText(QString("Git long SHA1:  %1").arg(GIT_CURRENT_SHA1));
    handleOwnText(QString("Link to commit: https://github.com/nasa-jpl/LiveViewLegacy/tree/%1").arg(GIT_CURRENT_SHA1_SHORT));
}

void consoleLog::handleOwnText(QString message)
{
    insertText(QString("[ConsoleLog]: %1").arg(message));
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
