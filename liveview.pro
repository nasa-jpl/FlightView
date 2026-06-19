#----------------------------------------------------------
#
# - LiveView  //  FlightView -
#
# https://github.com/nasa-jpl/LiveView
# Copyright 2014-2025 California Institute of Technology.
# Government Sponsorship(s) Acknowledged.
#----------------------------------------------------------

QT       += core gui
QT       += network svg

greaterThan(QT_MAJOR_VERSION, 4): QT += network widgets printsupport
DEFINES += GIT_CURRENT_SHA1="\\\"$(shell git -C \"$$PWD\" rev-parse HEAD)\\\""
DEFINES += GIT_CURRENT_SHA1_SHORT="\\\"$(shell git -C \"$$PWD\" rev-parse --short HEAD)\\\""
DEFINES += GIT_BRANCH="\\\"$(shell git -C \"$$PWD\" symbolic-ref --short HEAD)\\\""
DEFINES += SRC_DIR="\\\"\'$$PWD\'\\\""

# For CameraLink support,
# run qmake with the argument CONFIG+=USE_CAMERALINK
contains(CONFIG, USE_CAMERALINK) {
    message("Compiling with CameraLink support.")
    DEFINES += CAMERALINK
} else {
    message("Compiling without CameraLink support.")
}

# For ZeroMQ support,
# run qmake with the argument CONFIG+=USE_ZMQ
contains(CONFIG, USE_ZMQ) {
    message("Compiling with ZeroMQ support.")
    DEFINES += USE_ZMQ
} else {
    message("Compiling without ZeroMQ support.")
}

CONFIG+=link_pkgconfig
PKGCONFIG+=gstreamer-1.0 gstreamer-app-1.0 glib-2.0 gobject-2.0

TARGET = liveview
TEMPLATE = app

#CONFIG += console
#CONFIG += warn_on

SOURCES += main.cpp\
    consolelog.cpp \
    backend/src/acquire.cpp \
    backend/src/rtpnextgen.cpp \
    filenamegenerator.cpp \
    flight_widget.cpp \
    flightindicators.cpp \
    gpsGUI/zupt.cpp \
    gpsmanager.cpp \
    initialsetup.cpp \
    linebuffer.cpp \
    mainwindow.cpp \
    frameview_widget.cpp \
    controlsbox.cpp \
    frame_worker.cpp \
    qcustomplot.cpp \
    histogram_widget.cpp \
    fft_widget.cpp \
    profile_widget.cpp \
    pref_window.cpp \
    backend/src/safestringset.cpp \
    rgbadjustments.cpp \
    saveserver.cpp \
    playback_widget.cpp \
    gpsGUI/qledlabel.cpp \
    gpsGUI/gpsnetwork.cpp \
    gpsGUI/gpsbinaryreader.cpp \
    gpsGUI/gpsbinarylogger.cpp \
    udpbinarylogger.cpp \
    waterfall.cpp \
    waterfallviewerwindow.cpp \
    wfengine.cpp

contains(CONFIG, USE_ZMQ) {
    SOURCES += zmqclient.cpp
}

HEADERS  += mainwindow.h \
    consolelog.h \
    backend/include/fileformats.h \
    backend/include/rtpnextgen.hpp \
    dms.h \
    filenamegenerator.h \
    flight_widget.h \
    flightappstatustypes.h \
    flightindicators.h \
    frameview_widget.h \
    controlsbox.h\
    frame_worker.h \
    gpsGUI/zupt.h \
    gpsmanager.h \
    imagetagger.h \
    linebuffer.h \
    shm_gps.h \
    image_type.h \
    initialsetup.h \
    preferences.h \
    qcustomplot.h \
    histogram_widget.h \
    fft_widget.h \
    frame_c_meta.h \
    rgbadjustments.h \
    rgbline.h \
    backend/include/safestringset.h \
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
    udpbinarylogger.h \
    waterfall.h \
    preferences.h \
    waterfallviewerwindow.h \
    wfengine.h \
    wfshared.h

contains(CONFIG, USE_ZMQ) {
    HEADERS += zmqclient.h
}

DISTFILES +=    backend/include/acquire.hpp \
                aviris3-logo.png \
                backend/include/chroma_translate_filter.hpp \
                backend/include/std_dev_filter_device_code.cuh \
                backend/include/mean_filter.hpp \
                backend/include/frame_c.hpp \
                backend/include/fft.hpp \
                backend/include/dark_subtraction_filter.hpp \
                backend/include/white_ref_filter.hpp \
                backend/include/cuda_utils.hpp \
                backend/include/constants.h \
                backend/include/camera_types.hpp \
                backend/include/xiocamera.h \
                backend/include/osutils.h \
                backend/include/alphanum.hpp \
                backend/include/camera_types.h \
                backend/include/cameramodel.h \
                backend/include/cudalog.h \
                backend/include/takeoptions.h \
                backend/include/rtpcamera.hpp \
                backend/include/safelist.h \
                backend/include/safebuffer.h

DISTFILES +=    backend/src/acquire.cpp \
                backend/src/std_dev_filter_device_code.cu \
                backend/src/std_dev_filter.cpp \
                backend/src/mean_filter.cpp \
                backend/src/main.cpp \
                backend/src/fft.cpp \
                backend/src/dark_subtraction_filter.cpp \
                backend/src/white_ref_filter.cpp \
                backend/src/chroma_translate_filter.cpp \
                backend/src/xiocamera.cpp \
                backend/src/rtpcamera.cpp \
                backend/src/safelist.cpp



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

# macOS app icon
macx {
    ICON = liveview.icns
}

QMAKE_CXXFLAGS += -std=c++11 -faligned-new

# macOS-specific OpenMP configuration
macx {
    # Detect Homebrew installation location
    exists(/opt/homebrew/bin/brew) {
        HOMEBREW_PREFIX = /opt/homebrew
        message("Using Homebrew from /opt/homebrew (Apple Silicon)")
    } else {
        HOMEBREW_PREFIX = /usr/local
        message("Using Homebrew from /usr/local (Intel)")
    }
    
    # OpenMP support via libomp
    QMAKE_CXXFLAGS += -Xpreprocessor -fopenmp
    QMAKE_LFLAGS += -lomp
    INCLUDEPATH += $$HOMEBREW_PREFIX/include
    INCLUDEPATH += $$HOMEBREW_PREFIX/opt/libomp/include
    LIBS += -L$$HOMEBREW_PREFIX/lib
    LIBS += -L$$HOMEBREW_PREFIX/opt/libomp/lib
}

CONFIG(debug, debug|release) {
    # Debug-only diagnostics (e.g. frame hex dumps) are gated behind this define.
    DEFINES += FV_DEBUG_BUILD
    macx {
        QMAKE_CXXFLAGS += -std=c++11 -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter -Wno-unused-result
    } else {
        QMAKE_CXXFLAGS += -std=c++11 -march=native -mtune=native -fopenmp -Wno-class-memaccess -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-result
    }
}

CONFIG(release, debug|release) {
    macx {
        QMAKE_CXXFLAGS += -O3 -std=c++11 -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter -Wno-unused-result
    } else {
        QMAKE_CXXFLAGS += -O3 -std=c++11 -march=native -mtune=native -fopenmp -Wno-class-memaccess -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-result
    }
}

# OpenMP linking (Linux only, macOS handled above)
unix:!macx {
    QMAKE_LFLAGS += -fopenmp
}

# Platform-specific libraries
macx {
    # macOS doesn't have -lrt or -ldl
    LIBS += -lgsl -lgslcblas -lexiv2
    contains(CONFIG, USE_ZMQ) {
        LIBS += -lzmq
        # Add ZeroMQ C++ bindings include path (from cppzmq)
        exists(/opt/homebrew/include) {
            INCLUDEPATH += /opt/homebrew/include
        } else {
            INCLUDEPATH += /usr/local/include
        }
    }
} else {
    # Linux has additional libraries
    LIBS += -lgsl -lgslcblas -lexiv2 -lrt -ldl
    contains(CONFIG, USE_ZMQ) {
        LIBS += -lzmq
    }
}

# Used for build tracking:
DEFINES += HOST=\\\"`hostname`\\\" UNAME=\\\"`whoami`\\\"


# qmake will create this directory automatically:
DESTDIR = ./lv_release
# Copy files into DESTDIR for potential releases:
QMAKE_POST_LINK += cp \"$$PWD/liveview.png\" $$DESTDIR;
QMAKE_POST_LINK += cp \"$$PWD/LiveView.desktop\" $$DESTDIR;

# macOS app bundle configuration: Install launcher and config file
macx {
    # Copy config file to Resources
    config.files = $$PWD/macos_launch_config.txt
    config.path = Contents/Resources
    QMAKE_BUNDLE_DATA += config
    
    # Install the launcher script and rename the binary
    QMAKE_POST_LINK += mv $$DESTDIR/liveview.app/Contents/MacOS/liveview $$DESTDIR/liveview.app/Contents/MacOS/liveview-bin;
    QMAKE_POST_LINK += cp $$PWD/macos_launcher.sh $$DESTDIR/liveview.app/Contents/MacOS/liveview;
    QMAKE_POST_LINK += chmod +x $$DESTDIR/liveview.app/Contents/MacOS/liveview;
    QMAKE_POST_LINK += mv $$DESTDIR/liveview.app/Contents/Resources/macos_launch_config.txt $$DESTDIR/liveview.app/Contents/Resources/launch_config.txt;
}


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

# Link backend library (platform-specific)
macx {
    # macOS: no CUDA libraries, boost_system is header-only on macOS
    LIBS += -L$$PWD/backend/ -l_backend -lboost_thread -lboost_filesystem
    INCLUDEPATH += $$PWD/backend/include
} else:unix:!symbian {
    # Linux: include CUDA libraries and define USE_CUDA
    DEFINES += USE_CUDA
    LIBS += -L$$PWD/backend/ -l_backend -lboost_thread -lboost_filesystem -L/usr/local/cuda/lib64 -lcudart -lgomp -lboost_system -ldl -lrt # -lGL -lQtOpenGL
    INCLUDEPATH += $$PWD/backend/include
    INCLUDEPATH += /usr/local/cuda/include
}

contains(CONFIG, USE_CAMERALINK) {
    INCLUDEPATH += /opt/EDTpdv
    # Add EDT PDV libraries for Linux camera link builds
    unix:!macx {
        LIBS += -L/opt/EDTpdv -lpdv
    }
}

DEPENDPATH += $$PWD/backend

unix: PRE_TARGETDEPS += $$PWD/backend/lib_backend.a

FORMS += \
    flightindicators.ui \
    initialsetup.ui \
    rgbadjustments.ui \
    waterfallviewerwindow.ui
