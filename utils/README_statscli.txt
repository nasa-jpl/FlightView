# stats
FPA Statistic Processing Files

TOC:
Part 1: Build and install
Part 2: How to run it
Part 3: How it works

--- --- --- --- --- --- --- --- ---

---------------------- Part 1: Build and install files: ----------------------

1.1: build and install statscli for all users:
$cd statscli
$qmake
$make
$sudo cp statscli /opt/EDTpdv/
Verify /opt/EDTpdv is in path:
$echo $PATH
if not, add to ~/.bashrc:
PATH=$PATH:/opt/EDTpdv

1.2: Copy shell script foreachmatch.sh into path:
$cp foreachmatch.sh ~/bin
If necessary, add ~/bin to your path by editing ~/.bashrc:
PATH=~/bin:$PATH

Note: Issue the following command for bash to re-process your .bashrc file if you modified it:
$. ~/.bashrc
(dot space ~/.bashrc)

1.3: Copy gnuplot files into a convenient location:
$cd script
$mkdir ~/Documents/gnuplot
$cp -r \*.gnu ~/Documents/gnuplot

1.4: Modify files to indicate proper paths:
a: Edit ~/bin/foreachmatch.sh
Change line 10 to point to gnuplot plotstats.gnu file:
script=~/Documents/gnuplot/plotstats.gnu

b: Edit plotstats.gnu so that gnuplot can find other gnuplot scripts:
Change line 9:
set loadpath "~/Documents/gnuplot"

---------------------- Part 2: How to run it ----------------------

First enter the directory where the raw files are that you wish to process
$cd ~/Desktop/data
Now determine the proper wildcard to match to those files, for example:
$ls -l *dark.raw
$ls -l 2017*/*collection.raw
Note, it is not a good idea to include *.raw alone because this will cause the processed statistical files to potentially be re-processed infinitely. (Perhaps a newer version of this code would name the outputs something entirely differnet). Therefore, always include some text before '.raw' that is not a star. 
Bad: Collection*.raw
Good: *Collection.raw
Yes there is room for improvement on this interface!

Edit the foreachmatch.sh script so that it will find the files you are interested in:
$vim ~/bin/foreachmatch.sh
Line 12 contains the wildcard matching text, which uses a bash shell interator inside a for/do loop.
Example:
for i in $DIRECTORY/*dark.raw; do
Also, if the data come from an LVDS collection (2s compliment 16-bit), set line 7, 'dtype' to 'lvds'. If the data are in uint16, set the value to 'uint16'. 

Once this edit is complete, simply run the script:
$foreachmatch.sh

The script creates the following files, where 'x' is the "basename" of the input file:
x-mean.raw
x-stdev.raw
x-stats.txt
x-mean-img.eps
x-stdev-img-eps
x-mean-hist.eps
x-stdev-hist.eps
x-plots.pdf

The plots.pdf file is the ultimate file that contains all the EPS images.

---------------------- Part 3: How it works: ----------------------

The foreachmatch.sh script runs statscli on each matching file. This script then calls a master gnuplot script, plotstats.gnu, which then calls other gnuplot script to plot images and histograms. Much editing can be done in plotstats.gnu to, for example, change axis ranges or labels, titles, etc. Note: always extend the histogram range beyond the histogram's X axis by at least one bin, otherwise there will be a bin shown for "everything else" which can be hard to understand. Sometimes this is not so obvious. 

statscli is a program that uses libgsl (GNU Scientific Library, heavily based on FORTRAN recipies) and Intel's OpenMP (for multiprocessor support) to read raw files and write out raw files which contain the pixel mean value and pixel standard deviation value. The output format for the raw files is double-precision floating point. 

statscli has a number of useful command-line arguments. Execute statscli without any arguments to see them:
eliggett@AVIRIS-CAL:bin$ statscli
Arguments recognized: (may be used in any order)
	short:	long:     explanation:
	-i	--in      input data filename
	-m	--mean    output for mean file
	-s	--std     output for standard deviation file
	-x	--txt     output for quick stats text file
	-w	--width   image width
	-h	--height  image height
	-t	--type    data type (lvds or uint16, default)
	-z	--zap     set first row of each frame to zeros (optional)
	-o	--offset  set an initial offset to the data in bytes (optional)
	-n	--nframes set the number of frames to read (defaults to entire file minus any offset)
	-v	--verbose enable verbose mode
	-d	--debug   enable debug mode

Example:
statscli -i input.raw -m mean.raw-s std_dev.raw -x stats.txt -w 640 -h 480 -t uint16 --zap

Many of the arguments are optional. Statscli will assume geometry of 640x480, and also output to /tmp if an output filename is not given. Use the -v flag to see which (if any) default arguments are being assumed. 

The debug flag (-d) enables additional output, such as line-by-line command arguments and output file sizes. 

The offset flag causes the first frame to be read after a specified number of bytes. 

The user can also specify the number of frames to read. Combined with the offset flag, frames can be read in the middle of a collection, for example. 

The --zap option causes row zero (first row) to be all zeros, and also excludes the zero row from the overall FPA calculations shown in the text output and in the txt file. This may be useful when judging electrical noise as the metadata in the first row will appear as noise. 

The data type is assumed to be unsigned 16-bit integers. If the data were collected via LVDS, and are thus 16-bit 2s compliment, set the data type to 'lvds'. For unsigned 16-bit operation, omit the -t flag or specify 'uint16'.
