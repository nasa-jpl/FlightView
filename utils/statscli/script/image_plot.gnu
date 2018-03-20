if( !exists("filename")) filename=$0
if( !exists("titletext")) titletext=$1
if( !exists("bot_val")) bot_val=$2
if( !exists("top_val")) top_val=$3

set title titletext

call "jet_palette.gnu"
#set palette model HSV
#set palette rgb 3,2,2

if( exists("chroma")) {
    height = 480;
    width = 1280;
} else {
#    height = 481;
# for liveviewdcse.jpl.nasa.gov
    height = 480;
    width = 640;
}
    set yrange [height-1:0]
    set xrange [0:width-1]

set cbrange [bot_val:top_val]

plot filename binary array=(width,height) format="%double" with image
