#ifndef RGBLINE_H
#define RGBLINE_H

#include <ostream>
#include <iostream>


// A simple class to hold six arrays of numbers:
class rgbLine {
private:
    float * r_raw;
    float * g_raw;
    float * b_raw;

    unsigned char* red;
    unsigned char* green;
    unsigned char* blue;

    int size;
    void allocate()
    {
        r_raw = (float*)calloc(size, sizeof(float));
        g_raw = (float*)calloc(size, sizeof(float));
        b_raw = (float*)calloc(size, sizeof(float));

        red = (unsigned char*)calloc(size, sizeof(char));
        green = (unsigned char*)calloc(size, sizeof(char));
        blue = (unsigned char*)calloc(size, sizeof(char));
    }
    void quick_allocate()
    {
        r_raw = (float*)malloc(size * sizeof(float));
        g_raw = (float*)malloc(size * sizeof(float));
        b_raw = (float*)malloc(size * sizeof(float));

        red = (unsigned char*)malloc(size * sizeof(char));
        green = (unsigned char*)malloc(size * sizeof(char));
        blue = (unsigned char*)malloc(size * sizeof(char));
    }

public:
    rgbLine(int size, bool quick)
    {
        this->size = size;
        if(quick)
            quick_allocate();
        else
            allocate();
    }

    // Default to quick mode
    rgbLine(int size)
    {
        this->size = size;
        quick_allocate();
    }

    rgbLine()
    {
    }

    ~rgbLine()
    {
        //std::cout << "destructor running" << std::endl;
        free(r_raw);
        free(g_raw);
        free(b_raw);
        free(red);
        free(blue);
        free(green);
        //std::cout << "destructor finished" << std::endl;
    }

    void setSize(int size)
    {
        this->size = size;
        this->allocate();
    }

    float* getr_raw()
    {
        return r_raw;
    }

    float* getg_raw()
    {
        return g_raw;
    }

    float* getb_raw()
    {
        return b_raw;
    }

    unsigned char* getRed()
    {
        return red;
    }

    unsigned char* getGreen()
    {
        return green;
    }

    unsigned char* getBlue()
    {
        return blue;
    }
};


#endif // RGBLINE_H
