#ifndef FILENAMEGENERATOR_H
#define FILENAMEGENERATOR_H
#include <QString>
#include <QDateTime>
#include <QDir>

#include <QDebug>

class fileNameGenerator
{
public:
    fileNameGenerator();
    void setFlightFormat(bool useFlightFormat, QString flightNamePrefix);
    void setMainDirectory(QString basedir);
    void setFilenameExtension(QString extension);
    void setPrefix(QString prefix);
    void generate(); // call this for new time stamp
    QString getNewFullFilename(QString basedir, QString extension); // auto new filename
    QString getNewFullFilename(QString basedir, QString prefix, QString postfix, QString extension); // auto new filename
    QString getNewFullFilename(); // auto new filename
    QString getFullFilename(); // use existing timestamp
    QString getFullFilename(QString extension); // use existing timestamp
    QString getFullFilename(QString prefix, QString postfix, QString extension);
    QString getShortFilename(); // use existing timestamp
    QString getFilenameExtension();
    bool createDirectory(QString directory);
    bool createDirectory();

private:
    QString getTimeDatestring();
    QString shortFilename;
    QString directory;
    QString extension;

    bool flightFormat = true;
    QString namePrefix = "UNSET";
};

#endif // FILENAMEGENERATOR_H
