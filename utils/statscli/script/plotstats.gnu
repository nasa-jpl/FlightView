#filename_sd="/tmp/out_sd.raw"
#filename_mean="/tmp/out_mean.raw"
#filename_sd=system("echo $output_stdev")
#filename_mean=system("echo $output_mean")

print "Plotting using mean values from file: ", filename_mean;
    print "Plotting using stdv values from file: ", filename_sd;

set loadpath "/home/eliggett/NGIS_DATA/eliggett/gnuplot"

### Standard Deviation plot:
titletext="Standard Deviation Plot for filename ".filename_sd
filename=filename_sd
# 0 to 30 for image, 0 to 10 for LD
bot_val = 0;
top_val = 20;
fig=0;

#set term wxt fig; fig = fig + 1;
set terminal postscript eps noenhanced color font 'Helvetica,10'
set output plot_stdev
set xlabel "Horizontal"
set ylabel "Vertical"
call "image_plot.gnu" 

### SD. Histogram:
#set term wxt fig; fig = fig + 1;
set terminal postscript eps noenhanced color font 'Helvetica,10'
set output plot_stdev_hist
titletext="Histogram of sigma for file ".filename_sd
# 0 to 12 for LD mode, 0 to 32 for image
min_bin=0;
max_bin=25;
# 0 to 10 for LD mode, 0 to 30 for image
min_x = 0;
max_x = 20;
set xlabel "Standard Deviation"
set ylabel "Occurrences"
call "hist_plot.gnu"


### Mean image:

titletext="Mean pixel value Plot for file: ".filename_mean
filename=filename_mean
# 0 to 61000 for LD mode, 0 to 20000 for imaging warm
bot_val = 0;
top_val = 61000;

#set term wxt fig; fig = fig + 1;
set terminal postscript eps noenhanced color font 'Helvetica,10'
set output plot_mean
set xlabel "Horizontal"
set ylabel "Vertical"
call "image_plot.gnu" 

### Mean histogram:

#set term wxt fig; fig = fig + 1;
set terminal postscript eps noenhanced color font 'Helvetica,10'
set output plot_mean_hist
titletext="Histogram of pixel values for filename ".filename_mean
# both 0 to 61000 for LD mode, 0 to 20k for imaging warm
min_bin=0;
max_bin=61000;
min_x = 0;
max_x = 61000;
set ylabel "Occurrences"
set xlabel "Pixel Intensity (DN)"
call "hist_plot.gnu"


