/*
 * frame.h
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#ifndef FRAME_H_
#define FRAME_H_

#ifndef BYTES_PER_PIXEL
#define BYTES_PER_PIXEL 2
#endif //Bytes per pixel

#include "edtinc.h"
struct frame
{	//All these default to public since declaration was as struct
	unsigned int height; //The height and width of the image not including the first header row
	unsigned int width;
	u_char * raw_data;
	u_char * image_data_ptr; //= raw_data + width*BYTES_PER_PIXEL; //To get where the data actually begins


	frame(u_char * data_in, int size, int height, int width);
	virtual ~frame();
	//To get this as a 2d array, use a reinterpret cast, not going use a union here.

};

#endif /* FRAME_H_ */
