#ifndef PARSER_H
#define PARSER_H

#include <QString>
#include <QStringList>

class parser
{
public:
    parser(int argc, char * argv[]);
    QStringList args;
    QString input;
    QString output_mean;
    QString output_std;
    QString output_txt;
    QString program_name;

    unsigned int height;
    unsigned int width;
    unsigned int byte_offset;
    unsigned int nframes;
    bool nframes_supplied;
    bool from_lvds;
    bool zap_firstrow;
    bool ok;

    bool success;
    bool debug_mode;
    bool verbose_mode;

    int nargs;
    void debug();
    void help();

private:
    int next(int current_index);
    void set_defaults();
    void judge_success();

};

#endif // PARSER_H
