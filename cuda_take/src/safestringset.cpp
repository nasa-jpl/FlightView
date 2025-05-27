#include "safestringset.h"

void safeStringSet(std::string dst, std::string source)
{
//    if(&dst == NULL)
//    {
//        abort();
//    }

    dst.assign(source);
}

void safeStringSetC(char **dst, std::string source)
{
    if(*dst == NULL)
    {
        *dst = (char*)calloc(1024, 1);
    }
    strncpy(*dst, source.c_str(), 1023);
    (*dst)[1023] = '\0'; // Ensure null-termination
}

void safeStringSet(std::string *dst, std::string source)
{
    if(dst == NULL)
    {
        dst = new std::string;
    }

    dst->assign(source);
}

void safeStringSet(std::string *dst, std::string *source)
{
    if(dst == NULL)
    {
        dst = new std::string;
    }

    dst->assign(*source);
}

void safeStringSet(std::string *dst, const char* source)
{
    if(dst == NULL)
    {
        dst = new std::string;
    }

    dst->assign(source);
}

void safeStringDelete(std::string *p)
{
    if(p == NULL)
        return;
    delete p;
    return;
}
