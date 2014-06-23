#-------------------------------------------------
#
# Project created by QtCreator 2014-05-28T11:16:33
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = liveview2
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    frameview_widget.cpp \
    controlsbox.cpp \
    frame_worker.cpp \
    qcustomplot.cpp

HEADERS  += mainwindow.h \
    frameview_widget.h \
    controlsbox.h\
    frame_worker.h \
    image_type.h \
    qcustomplot.h

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

#NOTE! We're now using qcustomplot.cpp, because we're going to be making modifications to QColorMap stuff
# Tell the qcustomplot header that it will be used as library:
#DEFINES += QCUSTOMPLOT_USE_LIBRARY

# Link with debug version of qcustomplot if compiling in debug mode, else with release library:
#CONFIG(debug, release|debug) {
#  win32:QCPLIB = qcustomplotd1
#  else: QCPLIB = qcustomplotd
#} else {
#  win32:QCPLIB = qcustomplot1
#  else: QCPLIB = qcustomplot
#}
#LIBS += -L$$PWD/lib/ -l$$QCPLIB


unix:!macx:!symbian: LIBS += -L$$PWD/../../cuda_take/ -lcuda_take -lboost_thread -lcudart
INCLUDEPATH += $$PWD/../../cuda_take/include\
$$PWD/../../cuda_take/EDT_include
DEPENDPATH += $$PWD/../../cuda_take

unix:!macx:!symbian: PRE_TARGETDEPS += $$PWD/../../cuda_take/libcuda_take.a
