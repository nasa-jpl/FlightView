#ifndef FRAME_C_META_H
#define FRAME_C_META_H

/*! \file
 * \brief Declares the frame_c data structure as a meta type.
 * \paragraph
 *
 * The frame_c data structure is defined within the frame_c.h header of cuda_take
 * Here the class is declared as a metatype to be used as a QVariant inside of
 * Live View. */

#include "take_object.hpp"
#include "qmetatype.h"
Q_DECLARE_METATYPE(frame_c*)
Q_DECLARE_METATYPE(QVector<double>)
Q_DECLARE_METATYPE(QSharedPointer<QVector<double>>)

#endif // FRAME_C_META_H
