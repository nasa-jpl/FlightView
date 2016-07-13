

#min_bin = 0;
#max_bin = 4;

n_bins = 16384.0;
width = (max_bin-min_bin)/n_bins;
bin(x) = width*(floor((x-min_bin)/width) + 0.5) + min_bin
set boxwidth width
