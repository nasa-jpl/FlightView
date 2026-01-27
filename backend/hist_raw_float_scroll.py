#! /usr/bin/env python
import sys,struct
import numpy
import matplotlib.pyplot as plt
import os
raw_file_name = sys.argv[1]

width = 640
height = 480
frame_offset = width*height*4
short_num = width*(height)
f = open(raw_file_name, 'rb')


def readFrame(count):
    ba = f.read(frame_offset)
    fmtstring = str(width*height) + 'f'
    raw_nums = struct.unpack(fmtstring, ba)
    raw_1d = numpy.asarray(raw_nums)
    raw_2d = raw_1d.reshape((height,width));
    print raw_2d
    print "mean", numpy.mean(raw_1d)
    print "median", numpy.median(raw_1d)
    plt.imshow(raw_2d,cmap=plt.cm.gray)
    label = str("tell: %i frame_num: %i" % (f.tell(), change_frame.counter))
    plt.title(label)
    plt.draw()
    return raw_nums[160]

def change_frame(event):
    'toggle the visible state of the two images'
    event_char = event.key
    if event_char is 'e': 
	f.close()
	return
    if event_char is 'd' or (event_char is 'a' and change_frame.counter > 2):
    	if event_char is 'a' and change_frame.counter > 2:
		change_frame.counter = change_frame.counter - 1
		f.seek(-2*offset, os.SEEK_CUR)
    	if event_char is 'd':
		change_frame.counter = change_frame.counter + 1
	fc = readFrame(change_frame.counter)
plt.connect("key_press_event",change_frame)
#fig.canvas.mpl_connect("key_press_event",change_frame)
change_frame.counter = 0# initial_fnum
plt.show()
#plt.show(block=False)
#while True:
#    fig.canvas.get_tk_widget().update() # process events

