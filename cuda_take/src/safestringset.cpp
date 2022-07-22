#include "safestringset.h"

void safeStringSet(std::string dst, std::string source)
{
//    if(&dst == NULL)
//    {
//        abort();
//    }

    dst.assign(source);
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
