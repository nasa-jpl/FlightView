filename_sd="/tmp/out_sd.raw"
filename_mean="/tmp/out_mean.raw"

print "Plotting using mean values from file: ", filename_mean;
    print "Plotting using stdv values from file: ", filename_sd;

### Standard Deviation plot:
titletext="Standard Deviation Plot"
filename=filename_sd
bot_val = 0;
top_val = 3;
fig=0;

set term wxt fig; fig = fig + 1;
call "image_plot.gnu" 

### SD. Histogram:

set term wxt fig; fig = fig + 1;
titletext="Histogram of sigma"
min_bin=0;
max_bin=4;
call "hist_plot.gnu"


### Mean image:

titletext="Mean pixel value Plot"
filename=filename_mean
bot_val = 0;
top_val = 2500;

set term wxt fig; fig = fig + 1;
call "image_plot.gnu" 

### Mean histogram:

set term wxt fig; fig = fig + 1;
titletext="Histogram of pixel values"
min_bin=0;
max_bin=16535;
min_x = 0;
max_x = 2500;
call "hist_plot.gnu"


