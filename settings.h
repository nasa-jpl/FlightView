#ifndef SETTINGS_H
#define SETTINGS_H

/*! \file
 * \brief Defines frame rate rendering constants for Live View.
 * \paragraph
 *
 * The settings contain three main constants - the frameskip factor, the target framerate and the frame draw period. The backend generates frames
 * at a significantly faster rate than QCustomPlot can handle (100fps for 6604-A, and 118fps for 6604B). Therefore we should only draw every frame
 * modulo 10 to improve graphical performance. The target framerate sets the ideal fps for the frontend display, particularly for the frameview_widget.
 */

static const unsigned int FRAME_SKIP_FACTOR = 10; //This means only every frame modulo 10 will be redrawn, this has to do with the slowness of qcustomplot, a lower value will increase the frame rate. A value of 0 will make it attempt to draw every single frame
static const unsigned int TARGET_FRAMERATE = 60;
static const unsigned int FRAME_DISPLAY_PERIOD_MSECS = 1000 / TARGET_FRAMERATE;

//#define FRAME_SKIP_FACTOR 10
//On a 6604B this seems to have to be 10 for acceptable gui performance, on a 6604A it can be ~4

#endif // SETTINGS_H
