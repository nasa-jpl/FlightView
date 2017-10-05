#include "parser.h"
#include <iostream>

parser::parser(int argc, char *argv[])
{

    height=0;
    width=0;
    from_lvds=false;
    zap_firstrow=false;
    ok=false;
    success=false;

    bool show_help=false;

    nargs=argc-1;

    // Copy args into a list

    for(int n=1; n<argc; n++)
    {
        args.insert(n-1, QString::fromAscii(argv[n]));
    }

    program_name = QString::fromAscii(argv[0]);

    // Parse!
    for(int n=0; n < args.length(); n++)
    {
        if((args.at(n)=="-i") || (args.at(n)=="--in"))
        {
            input = args.at(next(n));
            n++;
        } else if ((args.at(n) == "--std") || (args.at(n) == "-s"))
        {
            output_std = args.at(next(n));
            n++;
        } else if ((args.at(n) == "--mean") || (args.at(n) == "-m"))
        {
            output_mean = args.at(next(n));
            n++;
        } else if ((args.at(n) == "-h") || (args.at(n) == "--height"))
        {
            height = args.at(next(n)).toInt(&ok);
            n++;
        } else if ((args.at(n) == "-w") || (args.at(n) == "--width"))
        {
            width = args.at(next(n)).toInt(&ok);
            n++;
        } else if ((args.at(n) == "-x") || (args.at(n) == "--txt"))
        {
            output_txt = args.at(next(n));
            n++;
        } else if ((args.at(n) == "-z") || (args.at(n) == "--zap"))
        {
            zap_firstrow = true;
        }
        else if ((args.at(n) == "-t") || (args.at(n) == "--type"))
        {
            // uint16
            //
            // lvds
            if( (args.at(next(n)).toLower() == "lvds") )
            {
                from_lvds = true;
            } else {
                from_lvds = false;
            }
            n++;

        } else {
            std::cout << "Unrecognized option: args.at(" << n << "): ";
            std::cout << args.at(n).toStdString() << std::endl;
            show_help=true;
        }
    }

    if(show_help || nargs == 0)
    {
        help();
    }

    judge_success(); // set success to true if parameters are good.


}

int parser::next(int current_index)
{
    if(current_index+1 > nargs)
    {
        return 0;
    } else {
        return current_index + 1;
    }
}

void parser::help()
{
    std::cout << "Arguments recognized: (may be used in any order)\n";
    std::cout << "\tshort:\tlong:    explanation:\n";
    std::cout << "\t-i\t--in     input data filename\n";
    std::cout << "\t-m\t--mean   output for mean file\n";
    std::cout << "\t-s\t--std    output for standard deviation file\n";
    std::cout << "\t-x\t--txt    output for quick stats text file\n";
    std::cout << "\t-w\t--width  image width\n";
    std::cout << "\t-h\t--height image height\n";
    std::cout << "\t-t\t--type   data type (lvds or uint16, default)\n";
    std::cout << "\t-z\t--zap    set first row of each frame to zeros (optional)\n";
    std::cout << "\nExample:\n";
    std::cout << program_name.toStdString() << " -i input.raw -m mean.raw";
    std::cout << "-s std_dev.raw -x stats.txt -w 640 -h 480 -t uint16 --zap\n";
}

void parser::set_defaults()
{
    if(!height)
        height=480;
    if(!width)
        width=640;
    if(output_mean.isEmpty())
        output_mean = "/tmp/output_mean.raw";
    if(output_std.isEmpty())
        output_std = "/tmp/output_stdev.raw";
    if(output_txt.isEmpty())
        output_txt = "/tmp/output_stats.txt";

}

void parser::judge_success()
{
    // set success parameter based on required inputs.
    set_defaults();
    if(input.isEmpty() || (!ok) )
    {
        success=false;
    } else {
        success=true;
    }
}

void parser::debug()
{
    std::cout << "Number of options: " << nargs << std::endl;
    std::cout << "Length of QStringList: " << args.length() << std::endl;
    std::cout << "Input: " << input.toStdString() << std::endl;
    std::cout << "Output Mean: " << output_mean.toStdString() << std::endl;
    std::cout << "Output Std Dev: " << output_std.toStdString() << std::endl;
    std::cout << "Output Text: " << output_txt.toStdString() << std::endl;
    std::cout << "Height: " << height << " Width: " << width << std::endl;
    std::cout << "From LVDS-capture: " << from_lvds << std::endl;
    std::cout << "Zap first row: " << zap_firstrow << std::endl;
    std::cout << "Successufl integer conversions: " << ok << std::endl;

    std::cout << "Complete argument list from QStringList: " << std::endl;

    for(int n=0; n < args.length(); n++)
    {
        std::cout << "args.at(" << n << ")\t";
        std::cout << args.at(n).toStdString() << std::endl;
    }

//    std::cout << "Complete argc/argv direct output: " << std::endl;

//    for(int n=0; n < argc; n++)
//    {
//        std::cout << "argv[" << n << "]\t\t";
//        std::cout << args.at(n).toStdString() << std::endl;
//    }

}

