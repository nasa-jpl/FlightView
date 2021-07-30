#include "flight_widget.h"

flight_widget::flight_widget(frameWorker *fw, QWidget *parent) : QWidget(parent)
{
    this->fw = fw;

    waterfall_widget = new frameview_widget(fw, WATERFALL, this);
    dsf_widget = new frameview_widget(fw, DSF, this);

    gps = new gpsManager();

    // Group Box "Flight Instrument Controls" items:
    flyBtn.setText("FLY!");
    instBtn.setText("Init");
    aircraftLbl.setText("AVIRIS-III");
    gpsLatText.setText("GPS Latitude: ");
    gpsLatData.setText("########");
    gpsLongText.setText("GPS Longitude: ");
    gpsLongData.setText("########");
    gpsLEDLabel.setText("GPS Status: ");
    gpsHeadingText.setText("Heading: ");
    gpsHeadingData.setText("###.###");
    gpsLED.setState(QLedLabel::StateWarning);
    cameraLinkLEDLabel.setText("CameraLink Status: ");
    cameraLinkLED.setState(QLedLabel::StateOk);

    // Format is &item, row, col, rowSpan, colSpan. -1 = to "edge"
    int row=0;

    flightControlLayout.addWidget(&gpsLatLonPlot, row,0,4,3);

    row += 4;

    flightControlLayout.addWidget(&flyBtn, ++row,0,1,1);
    flightControlLayout.addWidget(&instBtn, row,1,1,1);

    flightControlLayout.addWidget(&gpsLEDLabel, ++row,0,1,1);
    flightControlLayout.addWidget(&gpsLED, row,1,1,1);

    flightControlLayout.addWidget(&cameraLinkLEDLabel, ++row,0,1,1);
    flightControlLayout.addWidget(&cameraLinkLED, row,1,1,1);

    flightControlLayout.addWidget(&gpsLatText, ++row,0,1,1);
    flightControlLayout.addWidget(&gpsLatData, row,1,1,1);

    flightControlLayout.addWidget(&gpsLongText, ++row,0,1,1);
    flightControlLayout.addWidget(&gpsLongData, row,1,1,1);

    flightControlLayout.addWidget(&gpsHeadingText, ++row,0,1,1);
    flightControlLayout.addWidget(&gpsHeadingData, row,1,1,1);




    flightControlLayout.setColumnStretch(2,2);
    flightControlLayout.setRowStretch(3,2);



    // Group Box "Flight Instrument Controls"
    flightControls.setTitle("Flight Instrument Controls");
    flightControls.setLayout(&flightControlLayout);

    rhSplitter.setOrientation(Qt::Vertical);
    rhSplitter.addWidget(dsf_widget);
    rhSplitter.addWidget(&flightControls);



    lrSplitter.addWidget(waterfall_widget);
    lrSplitter.addWidget(&rhSplitter);

    lrSplitter.setHandleWidth(5);

    layout.addWidget(&lrSplitter);

    this->setLayout(&layout);

    // Connections to GPS:
    gps->insertLEDs(&gpsLED);
    gps->insertLabels(&gpsLatData, &gpsLongData,
                      &gpsAltitudeData, NULL,
                      NULL, NULL, NULL,
                      &gpsHeadingData, NULL,
                      NULL, NULL, NULL);
    gps->insertPlots(&gpsLatLonPlot);
    gps->prepareElements();
    connect(&flyBtn, SIGNAL(clicked(bool)), gps, SLOT(clearStickyError()));

}


double flight_widget::getCeiling()
{
    return dsf_widget->getCeiling();
}

double flight_widget::getFloor()
{
    return dsf_widget->getFloor();
}

void flight_widget::toggleDisplayCrosshair()
{
    dsf_widget->toggleDisplayCrosshair();
}

void flight_widget::handleNewFrame()
{
    dsf_widget->handleNewFrame();
    waterfall_widget->handleNewFrame();
}

void flight_widget::receiveGPS()
{

}

void flight_widget::handleNewColorScheme(int scheme)
{
    // It should be ok to call these directly:
    waterfall_widget->handleNewColorScheme(scheme);
    dsf_widget->handleNewColorScheme(scheme);
}

void flight_widget::colorMapScrolledX(const QCPRange &newRange)
{

}

void flight_widget::colorMapScrolledY(const QCPRange &newRange)
{

}

void flight_widget::setScrollX(bool Yenabled)
{

}

void flight_widget::setScrollY(bool Xenabled)
{

}

void flight_widget::updateCeiling(int c)
{
    waterfall_widget->updateCeiling(c);
    dsf_widget->updateCeiling(c);
}

void flight_widget::updateFloor(int f)
{
    waterfall_widget->updateFloor(f);
    dsf_widget->updateFloor(f);
}

void flight_widget::rescaleRange()
{
    waterfall_widget->rescaleRange();
    dsf_widget->rescaleRange();
}

void flight_widget::setCrosshairs(QMouseEvent *event)
{
    dsf_widget->setCrosshairs(event);
}

void flight_widget::debugThis()
{
    emit statusMessage("Debug function inside flight widget pressed.");
    gps->initiateGPSConnection("10.0.0.6", 8111);
}
