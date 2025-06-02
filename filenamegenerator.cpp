#include "filenamegenerator.h"

fileNameGenerator::fileNameGenerator()
{
    shortFilename = "unset";
    directory = "unset";
    extension = "unset";
}

void fileNameGenerator::setFlightFormat(bool useFlightFormat, QString flightNamePrefix="")
{
    this->flightFormat = useFlightFormat;
    if(flightFormat)
    {
        this->namePrefix = flightNamePrefix;
        //this->extension = "";
    }
}

void fileNameGenerator::setPrefix(QString prefix) {
    this->namePrefix = prefix;
}

void fileNameGenerator::setMainDirectory(QString basedir)
{
    this->directory = basedir;
}

void fileNameGenerator::setFilenameExtension(QString extension)
{
    this->extension = extension;
}

QString fileNameGenerator::getNewFullFilename(QString basedir, QString extension)
{
    // This is what to call when you need a new filename.
    this->directory = basedir;
    this->extension = extension;
    generate();
    return getFullFilename();
}

QString fileNameGenerator::getNewFullFilename(QString basedir,
                                              QString prefix,
                                              QString postfix, QString extension)
{
    // This is what to call when you need a new filename.
    // If you want to generate a second filename usig the existing timestamp,
    // then just call getFullFilename(prefix, postfix, extension) directly.
    this->directory = basedir;
    this->extension = extension;
    generate();
    return getFullFilename(prefix, postfix, extension);
}

QString fileNameGenerator::getNewFullFilename()
{
    generate();
    return getFullFilename();
}

QString fileNameGenerator::getFullFilename()
{
    if(extension.isEmpty())
        return directory + "/" + shortFilename;
    else
        return directory + "/" + shortFilename + "." + extension;
}

QString fileNameGenerator::getFullFilename(QString extension)
{
    if(extension.isEmpty())
        return directory + "/" + shortFilename;
    else
        return directory + "/" + shortFilename + "." + extension;
}

QString fileNameGenerator::getFullFilename(QString prefix, QString postfix, QString extension)
{
    if(extension.isEmpty())
        return directory + "/" + prefix + shortFilename + postfix;
    else
        return directory + "/" + prefix + shortFilename + postfix + "." + extension;
}

QString fileNameGenerator::getShortFilename()
{
    if(extension.isEmpty())
        return shortFilename;
    else
        return shortFilename + extension;
}

QString fileNameGenerator::getFilenameExtension()
{
    return extension;
}

void fileNameGenerator::generate()
{
    shortFilename = getTimeDatestring();
}

QString fileNameGenerator::getTimeDatestring()
{
    // YYYY-MM-DD_HHMMSS UTC
    QDateTime now = QDateTime::currentDateTimeUtc();
    QString dateString;
    if(flightFormat)
    {
        dateString.append(namePrefix);
        dateString.append(now.toString("yyyyMMdd"));
        dateString.append("t"); // t = "timezone" in datetime, so we must append it this way.
        dateString.append(now.toString("hhmmss"));
    } else {
        // Older format:
        dateString = now.toString("yyyy-MM-dd_hhmmss");
    }

    //qDebug() << __PRETTY_FUNCTION__ << "time string: " << dateString;
    return dateString;
}

bool fileNameGenerator::createDirectory(QString directoryName)
{
    if(directoryName.isEmpty())
    {
        qDebug() << "ERROR! GPS Base Directory requested for primary log is blank!";
        return false;
    }
    QDir dir(directoryName);
    if (!dir.exists())
        dir.mkpath(".");

    if(!dir.exists())
    {
        qDebug() << "ERROR! Could not create gps primary log directory " << directoryName;
    }
    return dir.exists();
}

bool fileNameGenerator::createDirectory()
{
    return this->createDirectory(this->directory);
}

