#ifndef SAFESTRINGSET_H
#define SAFESTRINGSET_H

#include <string>
#include <string.h>
#include <iostream>

void safeStringSet(std::string dst, std::string source);
void safeStringSet(std::string *dst, std::string source);
void safeStringSet(std::string *dst, std::string *source);
void safeStringSet(std::string *dst, const char* source);
void safeStringSetC(char **dst, std::string source);
void safeStringDelete(std::string *p);

#endif // SAFESTRINGSET_H
