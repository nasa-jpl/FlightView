#include <expat.h>
#include <exiv2/exiv2.hpp>

#include "gpsGUI/gpsbinaryreader.h"

using namespace Exiv2;

static bool imageTagger(const char* filename, gpsMessage start, gpsMessage dest, bool &ok) {
    // Check everything out first.
    if(filename==NULL)
        return false;

    if(!start.validDecode)
        return false;

    if(!dest.validDecode)
        return false;

    auto image = ImageFactory::open(filename);
    if(image.get()) {
        // do things
        image->readMetadata();
        Exiv2::ExifData& exifData = image->exifData();
        // see https://exiftool.org/TagNames/GPS.html
        exifData["Exif.GPSInfo.GPSProcessingMethod"] = "charset=Ascii HYBRID-FIX";
        exifData["Exif.GPSInfo.GPSVersionID"] = "2 2 0 0";
        exifData["Exif.GPSInfo.GPSMapDatum"] = "WGS-84";
        exifData["Exif.GPSInfo.GPSMeasureMode"] = "3"; // 3 = 3D, 2 = 2D
        exifData["Exif.GPSInfo.GPSDifferential"] = "1"; // 1 = differential correction applied, 0 = not applied


        if(start.havePosition) {
            // lat, lon, altitude, alt ref
            exifData["Exif.GPSInfo.GPSAltitudeRef"] = "0"; // 0 = MSL, 1 = BELOW sea level, for negative?
            exifData["Exif.GPSInfo.GPSAltitude"] = QString("%1/100").arg((int)(start.altitude*100)).toStdString(); // units are meters
        }

        exifData["Exif.GPSInfo.GPSSpeedRef"] = "N"; // N = knots, M = mph, K = kph
        exifData["Exif.GPSInfo.GPSSpeed"] = "140/1"; // knots

        exifData["Exif.GPSInfo.GPSTrackRef"] = "T"; // T = True North, M = Magnetic North
        exifData["Exif.GPSInfo.GPSTrack"] = "270/1"; // degrees?

        exifData["Exif.GPSInfo.GPSImgDirection"] = "277/1"; // degrees?
        exifData["Exif.GPSInfo.GPSImgDirectionRef"] = "T"; // T = True North, M = Magnetic North

        exifData["Exif.GPSInfo.GPSTimeStamp"] = "04/1 35/1 12/1"; // hh/1 mm/1 ss/1 assumed to be UTC
        exifData["Exif.GPSInfo.GPSDateStamp"] = "2024:01:03"; // yyyy:mm:dd


        // Start of image
        exifData["Exif.GPSInfo.GPSLatitude"] = "45/1 32/1 30020/10000";
        exifData["Exif.GPSInfo.GPSLongitude"] = "35/1 3/1 241/10";
        exifData["Exif.GPSInfo.GPSLatitudeRef"] = "N";
        exifData["Exif.GPSInfo.GPSLongitudeRef"] = "W";

        // End of image
        exifData["Exif.GPSInfo.GPSDestLongitude"] = "31/1 3/1 211/10";
        exifData["Exif.GPSInfo.GPSDestLatitude"] = "41/1 32/1 31020/10000";
        exifData["Exif.GPSInfo.GPSDestLatitudeRef"] = "N";
        exifData["Exif.GPSInfo.GPSDestLongitudeRef"] = "W";
        exifData["Exif.GPSInfo.GPSDestBearingRef"] = "T"; // T = True North, M = Magnetic North
        exifData["Exif.GPSInfo.GPSDestBearing"] = "269/1"; // degrees?



        exifData["Exif.Photo.UserComment"] = "charset=Ascii AVIRIS-III";
        image->writeMetadata();

    } else {
        return false;
    }

    return true;
}
