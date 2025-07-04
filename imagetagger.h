#ifndef IMAGETAGGER_H
#define IMAGETAGGER_H

#include <string>
#include <cstdio>

#include <QString>
#include <QDateTime>

#include <expat.h>
#include <exiv2/exiv2.hpp>

#include "gpsmanager.h"
#include "gpsGUI/gpsbinaryreader.h"
#include "dms.h"

using namespace Exiv2;

static std::string get100Int(float f) {
    return QString("%1/100").arg( (int)(f*100)).toStdString();
}

static std::string get100Int(double d) {
    return QString("%1/100").arg( (int)(d*100)).toStdString();
}

static std::string get1000Int(float f) {
    return QString("%1/1000").arg( (int)(f*1000)).toStdString();
}

static std::string get1000Int(double d) {
    return QString("%1/1000").arg( (int)(d*1000)).toStdString();
}

enum tagRtnType {
    tagRtnOK,
    tagRtnFilenameInvalid,
    tagRtnCantOpenFile,
    tagRtnStartGPSInvalid,
    tagRtnDestGPSInvalid,
    tagRtnGeneralExceptionOccurred,
    tagRtnUndefined
};

static tagRtnType imageTagger(const char* filename, gpsMessage start, gpsMessage dest, float fps,
                        int redRow, int greenRow, int blueRow, double gamma,
                        int floor, int ceiling, uint16_t recordingID, QString instrumentName) {
    // Check everything out first.
    if(filename==NULL) {
        std::cerr << "Tagger: no filename supplied\n";
        return tagRtnFilenameInvalid;
    }

    if(!start.validDecode) {
        std::cerr << "Tagger: starting gps is not valid decode.\n";
        return tagRtnStartGPSInvalid;
    }

    if(!dest.validDecode) {
        std::cerr << "Tagger: ending gps is not valid decode.\n";
        return tagRtnDestGPSInvalid;
    }

    try {
        auto image = ImageFactory::open(filename);
        if(image.get()) {
            // do things
            image->readMetadata();
            Exiv2::ExifData& exifData = image->exifData();
            // see https://exiftool.org/TagNames/GPS.html

            QDateTime now = QDateTime::currentDateTimeUtc();
            exifData["Exif.Image.DateTime"] = now.toString("yyyy:MM:dd hh:mm:ss").toStdString();

            exifData["Exif.GPSInfo.GPSProcessingMethod"] = "charset=Ascii HYBRID-FIX";
            exifData["Exif.GPSInfo.GPSVersionID"] = "2 2 0 0";
            exifData["Exif.GPSInfo.GPSMapDatum"] = "WGS-84";
            exifData["Exif.GPSInfo.GPSMeasureMode"] = "3"; // 3 = 3D, 2 = 2D
            exifData["Exif.GPSInfo.GPSDifferential"] = "1"; // 1 = differential correction applied, 0 = not applied
            exifData["Exif.Photo.ShutterSpeedValue"] = get100Int(fps); // waterfall FPS relates to downtrack skip

            if(start.havePosition) {
                // lat, lon, altitude, alt ref
                exifData["Exif.GPSInfo.GPSAltitudeRef"] = "0"; // 0 = MSL, 1 = BELOW sea level, for negative?
                exifData["Exif.GPSInfo.GPSAltitude"] = get1000Int(start.altitude); // units are meters
            }

            if(start.haveCourseSpeedGroundData) {
                exifData["Exif.GPSInfo.GPSSpeedRef"] = "N"; // N = knots, M = mph, K = kph
                exifData["Exif.GPSInfo.GPSSpeed"] = get100Int(start.speedOverGround*1.94384); // convert to knots
                exifData["Exif.GPSInfo.GPSTrackRef"] = "M"; // T = True North, M = Magnetic North
                exifData["Exif.GPSInfo.GPSTrack"] = get1000Int(start.courseOverGround);
            }

            // Image Direction, camera facing direction, but for us,
            // this will be the aircraft pointing direction, 'heading'
            if(start.haveAltitudeHeading) {
                exifData["Exif.GPSInfo.GPSImgDirection"] = get1000Int(start.heading);
                exifData["Exif.GPSInfo.GPSImgDirectionRef"] = "M"; // T = True North, M = Magnetic North
            }

            if(start.haveUTC) {
                utcTime t = processUTCstamp(start.UTCdataValidityTime);
                QString tstr = QString("%1/1 %2/1 %3/100").arg(t.hour).arg(t.minute).arg((int)(t.secondFloat*100));
                exifData["Exif.GPSInfo.GPSTimeStamp"] = tstr.toStdString(); // hh/1 mm/1 ss/1 assumed to be UTC
            }

            if(start.haveSystemDateData) {
                QString dstr = QString("%1:%2:%3").arg(start.systemYear).arg(start.systemMonth).arg(start.systemDay);
                exifData["Exif.GPSInfo.GPSDateStamp"] = dstr.toStdString(); // yyyy:mm:dd
            }

            // Start of image
            if(start.havePosition) {
                DMS::combDMS_t cLatStart = DMS::DegreesMinutesSecondsLat(start.latitude);
                if(start.longitude > 180)
                    start.longitude = -360+start.longitude;

                DMS::combDMS_t cLongStart = DMS::DegreesMinutesSecondsLon(start.longitude);
                exifData["Exif.GPSInfo.GPSLatitude"] = cLatStart.s;
                exifData["Exif.GPSInfo.GPSLatitudeRef"] = std::string(1, cLatStart.c);

                exifData["Exif.GPSInfo.GPSLongitude"] = cLongStart.s;
                exifData["Exif.GPSInfo.GPSLongitudeRef"] = std::string(1, cLongStart.c);
            }

            // End of image
            if(dest.havePosition) {
                DMS::combDMS_t cLat = DMS::DegreesMinutesSecondsLat(dest.latitude);
                if(dest.longitude > 180)
                    dest.longitude = -360+dest.longitude;

                DMS::combDMS_t cLong = DMS::DegreesMinutesSecondsLon(dest.longitude);
                exifData["Exif.GPSInfo.GPSDestLatitude"] = cLat.s;
                exifData["Exif.GPSInfo.GPSDestLatitudeRef"] = std::string(1, cLat.c);

                exifData["Exif.GPSInfo.GPSDestLongitude"] = cLong.s;
                exifData["Exif.GPSInfo.GPSDestLongitudeRef"] = std::string(1, cLong.c);
            }
            if(dest.haveAltitudeHeading) {
                exifData["Exif.GPSInfo.GPSDestBearingRef"] = "M"; // T = True North, M = Magnetic North
                exifData["Exif.GPSInfo.GPSDestBearing"] = get1000Int(dest.heading);
            }

            // Floor, Ceiling, RGB Rows, and Gamma:
            exifData["Exif.Photo.LensSpecification"] = QString("%1/1 %2/1 %3/1 1/1").arg(redRow).arg(greenRow).arg(blueRow).toStdString();
            exifData["Exif.Photo.Gamma"] = get100Int(gamma);
            exifData["Exif.Image.BlackLevel"] = QString("%1").arg(floor).toStdString();
            exifData["Exif.Image.WhiteLevel"] = QString("%1").arg(ceiling).toStdString();
            exifData["Exif.Image.WhitePoint"] = QString("%1/1 %2/1").arg(floor).arg(ceiling).toStdString();
            exifData["Exif.Image.Software"] = QString("FlightView %1").arg(GIT_CURRENT_SHA1_SHORT).toStdString();
            exifData["Exif.Photo.UserComment"] = QString("charset=Ascii %1").arg(instrumentName).toStdString();
            exifData["Exif.Image.SubjectLocation"] = (uint16_t)recordingID; // May be called "SubjectArea", datatype is "short" aka 16 bit
            image->writeMetadata();

        } else {
            std::cerr << "Tagger: cannot open image.\n";
            return tagRtnCantOpenFile;
        }
    }

    catch(...) {
        std::cerr << "Exception caught: Error tagging waterfall preview.";
#ifdef QT_DEBUG
        abort(); // crash on purpose if we had an error and it is a debug build.
#endif
        return tagRtnGeneralExceptionOccurred;
    }

    return tagRtnOK;
}


#endif // IMAGETAGGER_H
