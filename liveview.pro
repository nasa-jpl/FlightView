#-------------------------------------------------
#
# Project created by QtCreator 2014-05-28T11:16:33
#
#-------------------------------------------------

QT       += core gui
QT       += network

greaterThan(QT_MAJOR_VERSION, 4): QT += network widgets printsupport

TARGET = liveview
TEMPLATE = app

SOURCES += main.cpp\
    flight_widget.cpp \
        mainwindow.cpp \
    frameview_widget.cpp \
    controlsbox.cpp \
    frame_worker.cpp \
    qcustomplot.cpp \
    histogram_widget.cpp \
    fft_widget.cpp \
    profile_widget.cpp \
    pref_window.cpp \
    qledlabel.cpp \
    saveserver.cpp \
    playback_widget.cpp \


HEADERS  += mainwindow.h \
    flight_widget.h \
    frameview_widget.h \
    controlsbox.h\
    frame_worker.h \
    image_type.h \
    qcustomplot.h \
    histogram_widget.h \
    fft_widget.h \
    frame_c_meta.h \
    qledlabel.h \
    settings.h \
    profile_widget.h \
    pref_window.h \
    saveserver.h \
    playback_widget.h

OTHER_FILES += \
    aviris-ng-logo.png \
    aviris-logo-transparent.png \
    icon.png \
    liveview.rc

RESOURCES += \
    images.qrc

QMAKE_CXXFLAGS += -O2 -std=c++11 -march=native -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-result
#RC_FILE = liveview.rc

# Used for build tracking:
DEFINES += HOST=\\\"`hostname`\\\" UNAME=\\\"`whoami`\\\"

# qmake will create this directory automatically:
DESTDIR = ./lv_release
# Copy files into DESTDIR for potential releases:
QMAKE_POST_LINK += cp ../LiveViewLegacy/liveview_icon.png $$DESTDIR;
QMAKE_POST_LINK += cp ../LiveViewLegacy/LiveView.desktop $$DESTDIR;


#NOTE! We're now using qcustomplot.cpp, because we're going to be making modifications to QColorMap stuff
# Tell the qcustomplot header that it will be used as library:
# Link with debug version of qcustomplot if compiling in debug mode, else with release library:
#CONFIG(debug, release|debug) {
#  win32:QCPLIB = qcustomplotd1
#  else: QCPLIB = qcustomplotd
#} else {
#  win32:QCPLIB = qcustomplot1
#  else: QCPLIB = qcustomplot
#}
#LIBS += -L$$PWD/lib/ -l$$QCPLIB

#unix:!macx:!symbian: LIBS += -L$$PWD/../cuda_take/ -lcuda_take -lboost_thread -lcudart -lgomp -lboost_system -lokFrontPanel -ldl # -lGL -lQtOpenGL

unix:!macx:!symbian: LIBS += -L$$PWD/cuda_take/ -lcuda_take -lboost_thread -lcudart -lgomp -lboost_system -ldl # -lGL -lQtOpenGL
INCLUDEPATH += $$PWD/cuda_take/include\
/opt/EDTpdv
DEPENDPATH += $$PWD/cuda_take

unix:!macx:!symbian: PRE_TARGETDEPS += $$PWD/cuda_take/libcuda_take.a
