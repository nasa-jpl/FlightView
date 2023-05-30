#ifndef CUDALOG_H
#define CUDALOG_H
#include <iostream>

// from https://stackoverflow.com/questions/674060/customize-cout

// Lower numbers are lower-level things, which are hidden unless logginglevel is set to a lower number.

#define logginglevel (8)

// If you #define MUTE_LOG, then no output at all.
// define MUTE_LOG

// usage:
// LOG << "thing to be logged";
// LL(6) << "Log level 6 issue text";

class Log
{
public:
    bool hideline = false;

    Log(const std::string &funcName)
    {
#ifndef MUTE_LOG
            std::cout << funcName << ": ";
#endif
    }

    Log(const std::string &funcName, int logLevel)
    {
#ifndef MUTE_LOG
        if(logLevel >= logginglevel)
        {
            std::cout << funcName << " [" << logLevel << "]: ";
        } else {
            hideline = true;
        }
#endif
    }

    template <class T>
    Log &operator<<(const T &v)
    {
#ifndef MUTE_LOG
        if(!hideline)
            std::cout << v;
#endif
        return *this;
    }

    ~Log()
    {
#ifndef MUTE_LOG
        if(hideline)
        {
            // do nothing
        }
        else
        {
            std::cout << std::endl;
        }
#endif
    }
};

#define LOG Log(__FUNCTION__)
#define LL(x) Log(__FUNCTION__, x)

#endif // CUDALOG_H
