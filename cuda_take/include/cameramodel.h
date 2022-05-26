#ifndef CAMERAMODEL_H
#define CAMERAMODEL_H

#include <stdint.h>
#include <atomic>
#include <string>
#include <iostream>

#ifdef FAIL
#include <QObject>
#endif

#include "image_type.h"
#include "camera_types.h"

class CameraModel
{

public:
    CameraModel() { 
	running.store(false); }

    virtual ~CameraModel() { running.store(false); }


    virtual bool start() { return true; }
    virtual uint16_t *getFrame() = 0;

    virtual void setDir(const char *filename) {
        std::cout << "WARNING: using wrong setDir." << std::endl;
    }

    virtual bool isRunning() { return running.load(); }

    int getFrameWidth() const { return frame_width; }
    int getFrameHeight() const { return frame_height; }
    int getDataHeight() const { return data_height; }
    virtual char* getCameraName() const { return camera_name; }
    camera_t getCameraType() const { return camera_type; }
    source_t getSourceType() const { return source_type; }

protected:
    int frame_width;
    int frame_height;
    int data_height;
    char *camera_name;
    camera_t camera_type;
    source_t source_type;

    std::atomic<bool> running;
};

#endif // CAMERAMODEL_H
