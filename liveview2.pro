#-------------------------------------------------
#
# Project created by QtCreator 2014-05-28T11:16:33
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = liveview2
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    frameview_widget.cpp \
    controlsbox.cpp \
    frame_worker.cpp

HEADERS  += mainwindow.h \
    frameview_widget.h \
    controlsbox.h\
    frame_worker.h

BACKEND_HEADERS += edtinc.h \
    take_object.hpp\
    frame.hpp


HEADERS += BACKEND_HEADERS
OTHER_FILES += \
    aviris-ng-logo.png \
    aviris-logo-transparent.png \
    icon.png \
    liveview2.rc

RESOURCES += \
    images.qrc
#RC_FILE = liveview2.rc

unix:!macx:!symbian: LIBS += -L$$PWD/../../cuda_take/ -lcuda_take -lboost_thread
INCLUDEPATH += $$PWD/../../cuda_take/include\
$$PWD/../../cuda_take/EDT_include
DEPENDPATH += $$PWD/../../cuda_take

unix:!macx:!symbian: PRE_TARGETDEPS += $$PWD/../../cuda_take/libcuda_take.a
