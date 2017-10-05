#filename_sd="/tmp/out_sd.raw"
#filename_mean="/tmp/out_mean.raw"
#filename_sd=system("echo $output_stdev")
#filename_mean=system("echo $output_mean")

print "Plotting using mean values from file: ", filename_mean;
    print "Plotting using stdv values from file: ", filename_sd;

### Standard Deviation plot:
titletext="Standard Deviation Plot"
filename=filename_sd
bot_val = 0;
top_val = 10;
fig=0;

set term wxt fig; fig = fig + 1;
call "image_plot.gnu" 

### SD. Histogram:

set term wxt fig; fig = fig + 1;
titletext="Histogram of sigma"
min_bin=0;
max_bin=10;
min_x = 0;
max_x = 10
call "hist_plot.gnu"


### Mean image:

titletext="Mean pixel value Plot"
filename=filename_mean
bot_val = 0;
top_val = 61000;

set term wxt fig; fig = fig + 1;
call "image_plot.gnu" 

### Mean histogram:

set term wxt fig; fig = fig + 1;
titletext="Histogram of pixel values"
min_bin=0;
max_bin=61000;
min_x = 0;
max_x = 61000;
call "hist_plot.gnu"


