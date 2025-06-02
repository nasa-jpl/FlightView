#ifndef UDPBINARYLOGGER_H
#define UDPBINARYLOGGER_H

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "linebuffer.h"

// Characters allowed in IP address:
#define MAXIP (20)
// Characters per line of text:
// Make sure to also change this number in
// linebuffer.h around line 40.
#define MAXLINE (200)
// Bytes per packet:
#define PACKETSIZE (512)

class udpbinarylogger
{
public:
    udpbinarylogger(lineBuffer *buf, const char *ipaddress, unsigned short port, bool useHTML);
    void startNow(); // thread entry point;
    void setProcess(bool processNow);
    void closeDown();

private:
    lineBuffer* buffer = NULL;
    int ip_aton(const char *cp, struct in_addr *addr);
    void process_buffer(const char *ip, unsigned short port);
    void logError(const char *message);
    int verbose = 0;
    int starteof = 0;
    int binaryfile = 0;
    int html = 0;

    unsigned short port = 0;
    char ipaddress[24] = {'\0'};

    bool closing = false;
    bool processNow = true;
    char errorMessage[256] = {'\0'};
    bool usedErrorMessage = false;
    uint64_t count = 0;
};

#endif // UDPBINARYLOGGER_H
