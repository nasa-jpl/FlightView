#include "waterfall.h"

waterfall::waterfall(frameWorker *fw, int vSize, int hSize, QWidget *parent) : QWidget(parent)
{
    this->fw = fw;
    frHeight = fw->getFrameHeight();
    frWidth = fw->getFrameWidth();
    wfInUse = false;

    //rgbLineStruct blank = allocateLine();
    maxWFlength = 1024;
    wflength = maxWFlength;
    //wf.resize(maxWFlength, blank);
    allocateBlankWF();

    ceiling = 16000;
    floor = 0;

    r_row = 200;
    g_row = 250;
    b_row = 300;

    // Drawing:
    // Pixel format is AA RR GG BB, where AA = 8-bit Alpha value
    // AA = 0x00 = fully transparent
    // AA = 0xff = fully opaque

    vEdge = 0;
    hEdge = 0;
    this->vSize = vSize;
    this->hSize = hSize;
    // Override for now:
    this->vSize = maxWFlength;
    this->hSize = frWidth;
    opacity = 0xff;
    useDSF = false; // default to false since the program can't start up with a DSF mask anyway

    specImage = QImage(this->hSize, this->vSize, QImage::Format_ARGB32);
    statusMessage(QString("Created specImage with height %1 and width %2.").arg(specImage.height()).arg(specImage.width()));

    connect(&rendertimer, SIGNAL(timeout()), this, SLOT(handleNewFrame()));
    rendertimer.setInterval(FRAME_DISPLAY_PERIOD_MSECS);
    rendertimer.start();
    QSizePolicy policy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    this->setSizePolicy(policy);
    statusMessage("Finished waterfall constructor.");
}

void waterfall::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);

    // anti-alias:
    painter.setRenderHint(QPainter::SmoothPixmapTransform);

    painter.setWindow(QRect(0, 0, hSize/4.0, 1024));

    // Target: Where the image ultimately goes
    // Source: A rectangle which defines the portion of the specImage
    //         to copy into the target.

    // The source is stretched out over the target.

    // For length trimming, adjust the wflength parameter on the source
    // for Left-Right trimming, adjust the first two parameters of source.

    QRectF target(0, 0, hSize/4.0, vSize);
    QRectF source(0.0f, 0.0f, hSize, wflength); // use source geometry to "crop" the waterfall image
    painter.drawImage(target, specImage, source);
}

void waterfall::redraw()
{
    QColor c;
    for(int y = 0; y < vSize; y++)
    {
        for(int x = 0; x < hSize; x++)
        {
            c.setAlpha(opacity);
            c.setRed(wf.at(y)->getRed()[x]);
            c.setGreen(wf.at(y)->getGreen()[x]);
            c.setBlue(wf.at(y)->getBlue()[x]);

            specImage.setPixel(x, y, c.rgba());
        }
    }
    this->repaint();
}

void waterfall::allocateBlankWF()
{
    for(int n=0; n < maxWFlength; n++)
    {
        rgbLine* line = new rgbLine(frWidth, false);
        wf.push_back(std::shared_ptr<rgbLine>(line));
    }
}

void waterfall::copyPixToLine(float *image, float *dst, int rowSelection)
{
    for(int p=0; p < frWidth; p++)
    {
        dst[p] = image[ rowSelection + p];
    }
}

void waterfall::copyPixToLine(uint16_t *image, float *dst, int rowSelection)
{
    for(int p=0; p < frWidth; p++)
    {
        dst[p] = (float)image[ rowSelection + p];
    }
}

void waterfall::addNewFrame()
{
    // MUTEX wait-lock
    addingFrame.lock();
    float *local_image_ptr;
    uint16_t* local_image_ptr_uint16;
    if(fw->curFrame == NULL)
    {
        addingFrame.unlock();
        return;
    }

    local_image_ptr = fw->curFrame->dark_subtracted_data;
    local_image_ptr_uint16 = fw->curFrame->image_data_ptr;

    rgbLine *line = new rgbLine(frWidth);

    // Copy portions of the frame into the line

    int r_row_pix = frWidth * r_row; // width times how many rows
    int g_row_pix = frWidth * g_row;
    int b_row_pix = frWidth * b_row;

    if(fw->dsfMaskCollected() && useDSF)
    {
        copyPixToLine(local_image_ptr, line->getr_raw(), r_row_pix);
        copyPixToLine(local_image_ptr, line->getg_raw(), g_row_pix);
        copyPixToLine(local_image_ptr, line->getb_raw(), b_row_pix);
    } else {
        copyPixToLine(local_image_ptr_uint16, line->getr_raw(), r_row_pix);
        copyPixToLine(local_image_ptr_uint16, line->getg_raw(), g_row_pix);
        copyPixToLine(local_image_ptr_uint16, line->getb_raw(), b_row_pix);
    }

    // process initial RGB values:
    processLineToRGB(line);

    // ATOMIC spin-lock on the wf deque
    while(wfInUse)
    {
        usleep(500);
    }
    wfInUse = true;
    // push front
    wf.push_front(std::shared_ptr<rgbLine>(line));

    // pop back to remove oldest line
    //wf.pop_back();
    wf.resize(maxWFlength);
    wfInUse = false;
    addingFrame.unlock();
}

void waterfall::processLineToRGB(rgbLine* line)
{
    // go from float to RGB, with floor and ceiling scaling

    for(int p=0; p < frWidth; p++)
    {
        line->getRed()[p] = redLevel * scaleDataPoint(line->getr_raw()[p]);
        line->getGreen()[p] = greenLevel * scaleDataPoint(line->getg_raw()[p]);
        line->getBlue()[p] = blueLevel * scaleDataPoint(line->getb_raw()[p]);
    }
}

unsigned char waterfall::scaleDataPoint(float dataPt)
{
    // This function must map:
    // FLOOR --  CEILING
    //  \/        \/
    //   0   --  255

    if(ceiling == floor)
        return 127;

    dataPt = dataPt - floor;

    if(dataPt > ceiling-floor)
    {
        dataPt = ceiling-floor;
    }

    // Now that the data are between ceiling and floor:
    float span = ceiling - floor;
    float factor = 255.0f / span;

    if (  (dataPt) * factor >= 0 )
    {
        return (dataPt) * factor;
    } else {
        return 0;
    }
}

void waterfall::handleNewFrame()
{
    // Called externally when a new frame is available.
    // We can add other functions that happen per-frame here.
    // But first, we will copy the frame in:
    addNewFrame();
    // TODO: Update only every 25 ms
    this->redraw();
}

void waterfall::changeRGB(int r, int g, int b)
{
    this->r_row = r;
    this->g_row = g;
    this->b_row = b;
}

void waterfall::setRGBLevels(double r, double g, double b)
{
    this->redLevel = r;
    this->greenLevel = g;
    this->blueLevel = b;
}


void waterfall::setSpecOpacity(unsigned char opacity)
{
    this->opacity = opacity;
}

void waterfall::changeWFLength(int length)
{
    // Note, this actually just changes
    // the length that is displayed.
    if(length <=0)
        return;

    if(length <= maxWFlength)
    {
        wflength = length;
    } else {
        wflength = maxWFlength;
    }
}

void waterfall::updateCeiling(int c)
{
    ceiling = c;
    rescaleWF();
}

void waterfall::updateFloor(int f)
{
    floor = f;
    rescaleWF();
}

void waterfall::setUseDSF(bool useDSF)
{
    this->useDSF = useDSF;
}

void waterfall::rescaleWF()
{
    // mutex lock
    // atomic bool spin-lock on deque data
    scalingValues.lock();
    while(wfInUse)
    {
        usleep(500);
    }
    wfInUse = true;
    for(int wfrow=0; wfrow < maxWFlength; wfrow++)
    {
        processLineToRGB( wf[wfrow].get() );
    }
    wfInUse = false;
    scalingValues.unlock();

}

void waterfall::debugThis()
{
    statusMessage("In debugThis function.");
    this->redraw();
}

void waterfall::statusMessage(QString m)
{
    m.prepend(QString("WATERFALL: "));
    std::cout << m.toLocal8Bit().toStdString() << std::endl;
    emit statusMessageOut(m);
}
