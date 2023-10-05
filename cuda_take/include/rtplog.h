#ifndef RTPLOG_H
#define RTPLOG_H

#include <iostream>
#include <sstream>

extern "C" {
#include <gst/gstinfo.h>
#include <string.h>
#include <glib/gprintf.h>
}

// NB: This file is based off cudalog.h,
// but with modifications to use the gstreamer log handlers,
// necessary because gstreamer blocks stdout.

// from https://stackoverflow.com/questions/674060/customize-cout

// Lower numbers are lower-level things, which are hidden unless logginglevel is set to a lower number.

#define logginglevel (8)

// If you #define MUTE_LOG, then no output at all.
// define MUTE_LOG

// usage:
// RLOG << "thing to be logged";
// RLL(6) << "Log level 6 issue text";

class RTPLog
{
public:
    bool hideline = false;

    RTPLog(const std::string &funcName)
    {
#ifndef MUTE_LOG
            //std::cerr << funcName << ": ";
            gst_print("%s: ", funcName.c_str());
#endif
    }

    RTPLog(const std::string &funcName, int logLevel)
    {
#ifndef MUTE_LOG
        if(logLevel >= logginglevel)
        {
            //std::cerr << funcName << " [" << logLevel << "]: ";
            gst_print("%s [%d]: ", funcName.c_str(), logLevel);
        } else {
            hideline = true;
        }
#endif
    }

    template <class T>
    RTPLog &operator<<(const T &v)
    {
#ifndef MUTE_LOG
        if(!hideline) {
            std::stringstream ss;
            ss << v;
            gst_print("%s", ss.str().c_str());
        }
#endif
        return *this;
    }

    ~RTPLog()
    {
#ifndef MUTE_LOG
        if(hideline)
        {
            // do nothing
            // std::cerr << std::flush;
        }
        else
        {
            //std::cerr << std::endl << std::flush;
            gst_print("\n");
        }
#endif
    }
};

#define RLOG RTPLog(__PRETTY_FUNCTION__)
#define RLL(x) RTPLog(__PRETTY_FUNCTION__, x)

#endif // RTPLOG_H
