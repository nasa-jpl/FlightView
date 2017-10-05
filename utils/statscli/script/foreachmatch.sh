#!/bin/bash

DIRECTORY=.

height=640
width=480
dtype='lvds'


script=/home/eliggett/NGIS_DATA/eliggett/gnuplot/plotstats.gnu

for i in $DIRECTORY/*.decomp_no-header; do
    echo "Will now process $i."
    inputfile=$i
    shortfile="${inputfile%.*}"
    export output_mean="${inputfile%.*}"-mean.raw
    export output_stdev="${inputfile%.*}"-stdev.raw
    export output_stats="${inputfile%.*}"-stats.txt

    export plot_mean=$shortfile-mean-img.eps
    export plot_stdev=$shortfile-stdev-img.eps
    export plot_mean_hist=$shortfile-mean-hist.eps
    export plot_stdev_hist=$shortfile-stdev-hist.eps
    export pdfout=$shortfile-plots.pdf

    newstats -i $inputfile -m $output_mean -s $output_stdev -x $output_stats -h $height -w $width -t $dtype
    #lvdsstatscli $inputfile $output_mean $output_stdev $output_stats
    #lvdsstatscli $inputfile $output_mean $output_stdev $output_stats

    gnuplot -e "filename_sd=\"$output_stdev\"; filename_mean=\"$output_mean\"; \
       plot_mean=\"$plot_mean\"; plot_stdev=\"$plot_stdev\"; \
       plot_mean_hist=\"$plot_mean_hist\"; plot_stdev_hist=\"$plot_stdev_hist\"; call \"$script\""

    gs -q -dSAFER -dNOPAUSE -dBATCH -dEPSCrop -sOutputFile=$pdfout -sDEVICE=pdfwrite -c .setpdfwrite \
        -f $plot_mean $plot_mean_hist $plot_stdev $plot_stdev_hist

    #echo "mean: $output_mean"
    #echo "stdev: $output_stdev"
    #echo "stats: $output_stats"
done
