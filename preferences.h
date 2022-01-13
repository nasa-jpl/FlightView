#ifndef PREFERENCES_H
#define PREFERENCES_H


struct settingsT {
    // Camera:
    bool skipFirstRow = false;
    bool skipLastRow = false;
    bool use2sComp = false;
    bool nativeScale = true;
    bool brightSwap16 = false;
    bool brightSwap14 = false;

    // Interface:
    int frameColorScheme;
    int darkSubLow;
    int darkSubHigh;
    int rawLow;
    int rawHigh;

    // Flight:
    bool hidePlayback = true;

};




#endif // PREFERENCES_H
