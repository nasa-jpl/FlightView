#ifndef SETTINGS_H
#define SETTINGS_H


static const unsigned int FRAME_SKIP_FACTOR = 10; //This means only every frame modulo 10 will be redrawn, this has to do with the slowness of qcustomplot, a lower value will increase the frame rate. A value of 0 will make it attempt to draw every single frame
static const unsigned int TARGET_FRAMERATE = 20;
static const unsigned int FRAME_DISPLAY_PERIOD_MSECS = 1000/TARGET_FRAMERATE;

//#define FRAME_SKIP_FACTOR 10
//On a chroma this seems to have to be 10 for acceptable gui performance, on a 6604A it can be ~4

#endif // SETTINGS_H
