#ifndef WFSHARED_H
#define WFSHARED_H
// This file is for things common to
// the waterfall engine and the waterfall display

#define WF_SPEC_BUF_COUNT 10

#include <QImage>

struct specImageBuff_t {
    bool isValid = false;
    QImage* image[WF_SPEC_BUF_COUNT];
    int lastWrittenImage = 0;
    int currentWritingImage = 0;
};

Q_DECLARE_METATYPE(struct specImageBuff_t)

#endif // WFSHARED_H
