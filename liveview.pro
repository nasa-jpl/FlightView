#-------------------------------------------------
#
# Project created by QtCreator 2014-05-28T11:16:33
#
#-------------------------------------------------

QT       += core gui
QT       += network svg

greaterThan(QT_MAJOR_VERSION, 4): QT += network widgets printsupport
DEFINES += GIT_CURRENT_SHA1="\\\"$(shell git -C $$PWD rev-parse HEAD)\\\""
DEFINES += GIT_CURRENT_SHA1_SHORT="\\\"$(shell git -C $$PWD rev-parse --short HEAD)\\\""

TARGET = liveview
TEMPLATE = app

#CONFIG += console
#CONFIG += warn_on

SOURCES += main.cpp\
    consolelog.cpp \
    cuda_take/src/take_object.cpp \
    filenamegenerator.cpp \
    flight_widget.cpp \
    gpsmanager.cpp \
    mainwindow.cpp \
    frameview_widget.cpp \
    controlsbox.cpp \
    frame_worker.cpp \
    qcustomplot.cpp \
    histogram_widget.cpp \
    fft_widget.cpp \
    profile_widget.cpp \
    pref_window.cpp \
    saveserver.cpp \
    playback_widget.cpp \
    gpsGUI/qledlabel.cpp \
    gpsGUI/gpsnetwork.cpp \
    gpsGUI/gpsbinaryreader.cpp \
    gpsGUI/gpsbinarylogger.cpp \
    waterfall.cpp

HEADERS  += mainwindow.h \
    consolelog.h \
    filenamegenerator.h \
    flight_widget.h \
    frameview_widget.h \
    controlsbox.h\
    frame_worker.h \
    gpsmanager.h \
    image_type.h \
    preferences.h \
    qcustomplot.h \
    histogram_widget.h \
    fft_widget.h \
    frame_c_meta.h \
    rgbline.h \
    settings.h \
    profile_widget.h \
    pref_window.h \
    saveserver.h \
    playback_widget.h \
    gpsGUI/qledlabel.h \
    gpsGUI/gpsnetwork.h \
    gpsGUI/gpsbinaryreader.h \
    gpsGUI/gpsbinarylogger.h \
    startupOptions.h \
    waterfall.h \
    preferences.h

DISTFILES +=    cuda_take/include/take_object.hpp \
                cuda_take/include/chroma_translate_filter.hpp \
                cuda_take/include/std_dev_filter_device_code.cuh \
                cuda_take/include/mean_filter.hpp \
                cuda_take/include/frame_c.hpp \
                cuda_take/include/fft.hpp \
                cuda_take/include/dark_subtraction_filter.hpp \
                cuda_take/include/cuda_utils.hpp \
                cuda_take/include/constants.h \
                cuda_take/include/camera_types.hpp \
                cuda_take/include/xiocamera.h \
                cuda_take/include/osutils.h \
                cuda_take/include/alphanum.hpp \
                cuda_take/include/camera_types.h \
                cuda_take/include/cameramodel.h \
                cuda_take/include/cudalog.h \
                cuda_take/include/takeoptions.h

DISTFILES +=    cuda_take/src/take_object.cpp \
                cuda_take/src/std_dev_filter_device_code.cu \
                cuda_take/src/std_dev_filter.cpp \
                cuda_take/src/mean_filter.cpp \
                cuda_take/src/main.cpp \
                cuda_take/src/fft.cpp \
                cuda_take/src/dark_subtraction_filter.cpp \
                cuda_take/src/chroma_translate_filter.cpp \
                cuda_take/src/xiocamera.cpp



# the following two lines are needed for the QFI widgets:
include(qfi.pri)
INCLUDEPATH += gpsGUI


OTHER_FILES += \
    aviris-ng-logo.png \
    aviris-logo-transparent.png \
    icon.png \
    liveview.rc

RESOURCES += \
    images.qrc

QMAKE_CXXFLAGS += -O2 -std=c++11 -march=native -Wno-class-memaccess -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-result
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

unix:!macx:!symbian: LIBS += -L$$PWD/cuda_take/ -lcuda_take -lboost_thread -lboost_filesystem -L/usr/local/cuda/lib64 -lcudart -lgomp -lboost_system -ldl # -lGL -lQtOpenGL
INCLUDEPATH += $$PWD/cuda_take/include\
/opt/EDTpdv /usr/local/cuda/include
DEPENDPATH += $$PWD/cuda_take

unix:!macx:!symbian: PRE_TARGETDEPS += $$PWD/cuda_take/libcuda_take.a
