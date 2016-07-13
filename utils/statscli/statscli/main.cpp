#include <QCoreApplication>
#include <stdio.h>
#include <iostream>
#include <gsl/gsl_statistics_uint.h>
#include <gsl/gsl_statistics.h>
#include <stdint-gcc.h>

/* If the macros to define the development environment are not defined at compile time, use defaults */
#ifndef HOST
#define HOST "unknown location"
#endif
/* If the macros to define the version author are not defined at compile time, use defaults */
#ifndef UNAME
#define UNAME "unknown person"
#endif


int main(int argc, char *argv[])
{
    unsigned int pixel_size = sizeof(uint16_t);
    // For AVIRIS-NG and other NGIS instruments:
    // height = 481
    // width = 640
    // For chroma 1280:
    // height = 480
    // width = 1280
    const unsigned int height = 481;
    const unsigned int width = 640;
    // unsigned int frame_size_bytes = height*width*pixel_size;
    unsigned int frame_size_numel = height*width;
    unsigned int nframes = 0;
    uint16_t * frames;
    double * sd_frame;
    double * mean_frame;
    double mean_frame_value = 0;
    double mean_sd_value = 0;
    unsigned int * input_array;

    size_t bytes_read = 0;
    size_t bytes_written_mean = 0;
    size_t bytes_written_sd = 0;

    if(argc == 1)
    {
        // arg[0] is program name, so this is the case where no arguments are supplied
        std::cout << "statscli compiled at " << __DATE__ << ", " << __TIME__ << " PDT\n";
        std::cout << "                  by " << UNAME << "@" << HOST << "\n";
        std::cout << "Usage: \n";
        std::cout << "./statscli input.raw out_mean.raw out_sd.raw stats.txt" << std::endl;
        return 0;
    }

    // Load the file:


    // argument parsing:
    if(argc != 5)
    {
        // wrong number of arguments
        std::cout << "ERROR: Must supply all arguments:" <<std::endl;
        std::cout << "./statscli input.raw out_mean.raw out_sd.raw stats.txt" << std::endl;
        return -1;
    }
    /*
    std::cout << "input arguments: " << argc << std::endl;
    std::cout << "input file: " << argv[1] << std::endl;
    std::cout << "mean_out file: " << argv[2] << std::endl;
    std::cout << "sd_out file: " << argv[3] << std::endl;
*/

    FILE * file = fopen(argv[1], "r");
    if (file == NULL)
    {
        std::cout << "ERROR! NULL pointer for file\n";
        std::cout << "Go to jail, do not collect $200.\n";
        std::cout << "--- Cannot open file, please check the file path for [" << argv[1] << "]\n";
        return -1;
    }

    // Pt 1: how much data is there?
    fseek(file, 0, SEEK_END);
    long int filesize = ftell(file);
    // std::cout << "File has " << filesize << " bytes\n";
    // std::cout << "Which is " << filesize/pixel_size << " pixels.\n";
    nframes = filesize / pixel_size / (height * width);
    // std::cout << "nframes: " << nframes << "\n";
    fseek(file, 0, SEEK_SET);
    // If you want to load from an offset, computethe offset and put it here:
    // fseek(file, frame_offset, SEEK_SET);
    // You will also need to change the 'nframes' variable to reflect loading fewer frames.

    // Pt 2: malloc space to load file into:
    frames = (uint16_t *) malloc(filesize);
    input_array = (unsigned int *) malloc(sizeof(double) * nframes * frame_size_numel); // native size

    // Pt 3: Load the data form the file
    std::cout << "Loading...";
    bytes_read = fread(frames, sizeof(uint16_t), filesize/pixel_size, file);
    std::cout << "Done!\nRead in " << nframes  << " frames at (HxW): " << height << "x" << width <<"\n";
    fclose(file); // all done with input file

    // Pt 4: convert to standard unsigned int to match GSL:
    // GSL is written for native unsigned int rather than 16 bit.
    std::cout << "Converting data...";
    for(unsigned int nth_element = 0; nth_element < frame_size_numel * nframes; nth_element++)
    {
        input_array[nth_element] = (unsigned int)frames[nth_element];
    }
    free(frames); // careful, but if we don't need this anymore we can free it.
    std::cout << "Done.\n";


    // do stats:
    // Stats pt 1: Allocate data for the stats to go:
    sd_frame = (double *) malloc(sizeof(double) * frame_size_numel);
    mean_frame = (double *) malloc(sizeof(double) * frame_size_numel);

    // Stats pt 2: loop around the file:
    std::cout << "Computing statistics...";

    #pragma omp parallel for
    for(unsigned int nth_frame_el = 0; nth_frame_el < frame_size_numel; nth_frame_el++)
    {
        // iterate over each pixel in a frame
        mean_frame[nth_frame_el] = gsl_stats_uint_mean(input_array+nth_frame_el, frame_size_numel, nframes);
        sd_frame[nth_frame_el] = gsl_stats_uint_sd(input_array+nth_frame_el, frame_size_numel, nframes);
    }

    if (height==481)
    {
        // Skip first row of metadata.
        mean_frame_value = gsl_stats_mean(mean_frame+width, 1, frame_size_numel-width);
        mean_sd_value = gsl_stats_mean(sd_frame+width, 1, frame_size_numel-width);
        std::cout << "Done! (skipped first metadata row)\n";
    } else {
        // examine entire frame
        mean_frame_value = gsl_stats_mean(mean_frame, 1, frame_size_numel);
        mean_sd_value = gsl_stats_mean(sd_frame, 1, frame_size_numel);
        std::cout << "Done!\n";
    }

    std::cout << "Mean value: " << std::fixed << mean_frame_value << std::endl;
    std::cout << "Mean dev  : " << std::fixed << mean_sd_value << std::endl;

    // Save the stats out to a file:
    FILE * outfile_values = fopen(argv[4], "w");
    fprintf(outfile_values, "%lf,%lf\n", mean_frame_value, mean_sd_value);
    fclose(outfile_values);

    // save file

    FILE * outfile_mean = fopen(argv[2], "w");
    FILE * outfile_sd = fopen(argv[3], "w");

    bytes_written_mean = fwrite(mean_frame, sizeof(double), frame_size_numel, outfile_mean);
    bytes_written_sd = fwrite(sd_frame, sizeof(double), frame_size_numel, outfile_sd);

    // std::cout << "Wrote " << bytes_written_mean << " elements to mean file\n";
    // std::cout << "Wrote " << bytes_written_sd << " elements to sd file\n";


    fclose(outfile_mean);
    fclose(outfile_sd);

    // clean up
    free(mean_frame);
    free(sd_frame);
    free(input_array);

    return 0;
}
