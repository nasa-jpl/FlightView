#ifndef IMAGE_TYPE_H
#define IMAGE_TYPE_H

/*! \file
 * \brief Contains a list of all frontend image type.
 * Several members are deprecated in the current version.
 * \deprecated VERTICAL_CROSS and
 * \deprecated HORIZONTAL_CROSS
 * \deprecated images are no longer used as the profile_widget accounts for crosshair profiles as a more modular form of mean profile.
 * When using image_types in a switch statement, use default: break; to circumvent warnings about missed members. */

enum image_t {BASE, DSF, STD_DEV, STD_DEV_HISTOGRAM, VERTICAL_MEAN, HORIZONTAL_MEAN, FFT_MEAN,\
              VERTICAL_CROSS, HORIZONTAL_CROSS, VERT_OVERLAY};

#endif // IMAGE_TYPE_H
