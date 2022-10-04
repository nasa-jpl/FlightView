HEADERS += \
    gpsGUI/qfi/qfi_doxygen.h

################################################################################

HEADERS += \
    gpsGUI/qfi/qfi_Colors.h \
    gpsGUI/qfi/qfi_Fonts.h

SOURCES += \
    gpsGUI/qfi/qfi_Colors.cpp \
    gpsGUI/qfi/qfi_Fonts.cpp

################################################################################
# Electronic Flight Instrument System (EFIS)
################################################################################

HEADERS += \
    gpsGUI/qfi/qfi_EADI.h \
    gpsGUI/qfi/qfi_EHSI.h

SOURCES += \
    gpsGUI/qfi/qfi_EADI.cpp \
    gpsGUI/qfi/qfi_EHSI.cpp

################################################################################
# Basic Six
################################################################################

HEADERS += \
    gpsGUI/qfi/qfi_AI.h \
    gpsGUI/qfi/qfi_HI.h \
    gpsGUI/qfi/qfi_VSI.h \
    gpsGUI/qfi/qfi_ASI.h \
    gpsGUI/qfi/qfi_ALT.h \
    gpsGUI/qfi/qfi_TC.h

SOURCES += \
    gpsGUI/qfi/qfi_AI.cpp \
    gpsGUI/qfi/qfi_HI.cpp \
    gpsGUI/qfi/qfi_VSI.cpp \
    gpsGUI/qfi/qfi_ASI.cpp \
    gpsGUI/qfi/qfi_ALT.cpp \
    gpsGUI/qfi/qfi_TC.cpp

################################################################################
# Resources
################################################################################

RESOURCES += \
    gpsGUI/qfi/qfi.qrc
