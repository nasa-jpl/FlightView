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
    controlsbox.cpp

HEADERS  += mainwindow.h \
    frameview_widget.h \
    controlsbox.h

OTHER_FILES += \
    aviris-ng-logo.png \
    aviris-logo-transparent.png \
    icon.png

RESOURCES += \
    images.qrc
