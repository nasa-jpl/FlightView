
if( !exists("filename")) filename=$0
if( !exists("titletext")) titletext=$1
if( !exists("bot_val")) bot_val=$2
if( !exists("top_val")) top_val=$3

# get histogram bins:
call "hist_bins.gnu"

set yrange [] noreverse;
set autoscale y
if( exists("min_x")) {
    set xrange [min_x:max_x];
} else {
    set xrange [min_bin:max_bin];
}
set title titletext
set nokey


plot filename binary  format="%double" using (bin(column(1))):(1.0) smooth freq with boxes lt rgb "blue"
