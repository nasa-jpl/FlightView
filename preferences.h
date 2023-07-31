#ifndef PREFERENCES_H
#define PREFERENCES_H
#include <QString>

struct settingsT {

    // Runtime-only:
    bool readFile = false;

    // [Camera]:
    bool skipFirstRow = false;
    bool skipLastRow = false;
    bool use2sComp = false;
    bool nativeScale = true;
    bool brightSwap16 = false;
    bool brightSwap14 = false;

    // [Interface]:
    int frameColorScheme;
    bool useDarkTheme;
    int frameViewCeiling;
    int frameViewFloor;

    int dsfCeiling;
    int dsfFloor;

    int fftCeiling;
    int fftFloor;

    int stddevCeiling;
    int stddevFloor;

    int preferredWindowWidth = 1280;
    int preferredWindowHeight = 1024;

    QByteArray windowGeometry;
    QByteArray windowState;

    // [RGB]:
    unsigned int bandRed[10];
    unsigned int bandBlue[10];
    unsigned int bandGreen[10];
    QString presetName[10];
    double gainRed[10] = {1.0};
    double gainBlue[10] = {1.0};
    double gainGreen[10] = {1.0};
    double gamma[10] = {1.0};
    bool gammaEnabled[10] = {false};

    // [Flight]:
    bool hidePlayback = true;
    bool hideFFT = true;
    bool hideVerticalOverlay = true;
    bool hideVertMeanProfile = false;
    bool hideVertCrosshairProfile = false;
    bool hideHorizontalMeanProfile = false;
    bool hideHorizontalCrosshairProfile = false;
    bool hideHistogramView = false;
    bool hideStddeviation = false;
    bool hideWaterfallTab = false;
    int percentDiskWarning = 85;
    int percentDiskStop = 99;
};




#endif // PREFERENCES_H
