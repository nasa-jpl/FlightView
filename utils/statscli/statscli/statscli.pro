#-------------------------------------------------
#
# Project created by QtCreator 2014-08-28T19:40:36
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = statscli
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    parser.cpp

LIBS += -lgsl -lgslcblas

DEFINES += HOST=\\\"`hostname`\\\" UNAME=\\\"`whoami`\\\"

QMAKE_CXXFLAGS += -fopenmp -march=native -O3 -mtune=native -Wno-unused-but-set-variable
QMAKE_LFLAGS += -fopenmp
QMAKE_CXXFLAGS_RELEASE += -fopenmp -march=native -O3 -mtune=native

HEADERS += \
    main.h \
    parser.h
