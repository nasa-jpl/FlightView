#include "udpbinarylogger.h"

// UDP Binary Logger

// Based off code from:
// Brian Rheingans 2/2013 AVIRIS Classic UDP Binary Logger code

// The ipaddress variable must be an IPV4 dotted-decimal ip address.

udpbinarylogger::udpbinarylogger(lineBuffer* buffer, const char *ipaddressIn,
                                 unsigned short port, bool useHTML)
{

    this->port = port;
    this->buffer = buffer;
    strncpy(this->ipaddress, ipaddressIn, 16); // XXX.XXX.XXX.XXX

    fprintf(stdout, "UDP Logger IP address: %s\n", ipaddress);
    fprintf(stdout, "UDP Logger       Port: %hu\n", port);

    html = useHTML; // add "pre" tags for unformatted text display
}

void udpbinarylogger::startNow() {
    pthread_setname_np(pthread_self(), "UDPLogger");
    process_buffer(ipaddress, port);
}

void udpbinarylogger::closeDown() {
    this->closing = true;
}

void udpbinarylogger::setProcess(bool processNow) {
    this->processNow = processNow;
}

void udpbinarylogger::logError(const char *message) {
    // For now, this is simply a debugging mechanism.
    strncpy(errorMessage, message, 256-1);
    usedErrorMessage = true;
    std::cerr << message << std::endl;
}

int udpbinarylogger::ip_aton(const char *cp, struct in_addr *addr) {

    unsigned int a, b, c, d;
    char *p;
    u_int32_t ip;
    char ipstr[MAXIP+1];

    strncpy(ipstr, cp, MAXIP);

    if ((p = strtok(ipstr, ".")) == 0) {
        logError("Bad 'a' part of ip address");
        return(0);
    }
    a = atoi(p);
    if ((p = strtok(NULL, ".")) == 0) {
        logError("Bad 'b' part of IP address");
        return(0);
    }
    b = atoi(p);
    if ((p = strtok(NULL, ".")) == 0) {
        logError("Bad 'c' part of IP address\n");
        return(0);
    }
    c = atoi(p);
    if ((p = strtok(NULL, "")) == 0) {
        logError("Bad 'd' part of IP address\n");
        return(0);
    }
    d = atoi(p);

    ip = (a << 24) | (b << 16) | (c << 8) | d;
    addr->s_addr = htonl(ip);

    return(1);
}


void udpbinarylogger::process_buffer(const char *ip, unsigned short port) {

    int handle;
    struct sockaddr_in src_address;
    struct sockaddr_in dest_address;
    FILE *fp;
    char line[MAXLINE];
    char buf[PACKETSIZE];
    char *packet_bufptr = buf;
    unsigned int bufbytes;
    unsigned int segnum = 0;
    int sent_bytes;
    int reserved_bytes;
    int i;

    /* Setup UDP socket interface */

    handle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (handle < 0) {
        logError("Failed to create socket");
        return;
    }

    /* Bind source address to this IP addres, any port */

    memset(&src_address, 0, sizeof(src_address));
    src_address.sin_family = AF_INET;
    src_address.sin_addr.s_addr = htonl(INADDR_ANY);	/* Any address */
    src_address.sin_port = htons(0);			/* Random port */
    if (bind(handle, (const struct sockaddr*)&src_address, sizeof(src_address)) < 0) {
        logError("Failed to bind source socket");
        return;
    }

    /* Set destination IP address and port from arguments */

    memset(&dest_address, 0, sizeof(dest_address));
    dest_address.sin_family = AF_INET;
    dest_address.sin_port = htons(port);
    if (ip_aton(ip, &dest_address.sin_addr) == 0) {
        logError("Bad IP address");
        return;
    }

    if (!buffer) {
        logError("Buffer is NULL");
        return;
    }


    /* Read log lines (like tail -f), packetize and send UDP packets */

    if (html) {						/* optionally, output html tags */
        sprintf(packet_bufptr, "<pre>");
        packet_bufptr += strlen("<pre>");
        reserved_bytes = strlen("</pre>")+1;
    } else {
        reserved_bytes = 1;
    }
    // bool bufferSuccess = false;
    size_t lineLength = 0;
    sprintf(packet_bufptr, "%5d\n", segnum++);			/* put segment number in packet */
    packet_bufptr += 6;						/* adjust packet pointer */
    while (!closing) {						/* loop forever */

        while ( buffer->readLine(line, lineLength) && (!closing) ) {
            bufbytes = PACKETSIZE-reserved_bytes-(int)(packet_bufptr-buf); /* adjust packet size */
            if (verbose) {
                printf("Bufbytes: %3d - %3d - %2ld  Line: %s",
                       bufbytes, (int)(packet_bufptr-buf), lineLength, line);
            }
            if ((int)(bufbytes-lineLength) > 0) {		/* will line fit in packet? */
                strncpy(packet_bufptr, line, bufbytes);		/* put line in packet */
                packet_bufptr += lineLength;				/* adjust packet pointer */
            } else {
                if (html) {				        /* optionally, output html tags */
                    sprintf(packet_bufptr, "</pre>");
                    packet_bufptr += strlen("</pre>");
                }
                while (bufbytes > 0) {				/* pad end of packet */
                    *packet_bufptr = ' ';				/* with spaces */
                    packet_bufptr++;					/* adjust packet pointer */
                    bufbytes--;					/* adjust packet size */
                    if (verbose) {
                        printf("%d ",bufbytes);
                    }
                }
                *packet_bufptr = '\n';					/* append newline to end of packet  */
                if (verbose) {
                    printf("\nPacket:\n|");
                    for (i = 0; i < PACKETSIZE; i++) {
                        printf("%c", buf[i]);
                    }
                    printf("|\n");
                }
                /* send packet to destination */
                sent_bytes = sendto(handle, (const void*)buf, sizeof(buf), 0,
                                    (struct sockaddr*)&dest_address, sizeof(dest_address));
                if (sent_bytes != sizeof(buf)) {
                    logError("Failed to send packet");
                }
                usleep(10000);				/* sleep 10 ms after sending packet */
                packet_bufptr = buf;					/* reset packet pointer */
                if (html) {					/* optionally, output html tags */
                    sprintf(packet_bufptr, "<pre>");
                    packet_bufptr += strlen("<pre>");
                }
                sprintf(packet_bufptr, "%5d\n", segnum++);		/* put segment number in packet */
                packet_bufptr += 6;					/* adjust packet pointer */
                strncpy(packet_bufptr, line, PACKETSIZE-(int)(packet_bufptr-buf)); /* put outstanding line in packet */
                packet_bufptr += lineLength;				/* adjust packet pointer */
            }
            count++;
        }
        usleep(10000);				/* sleep 10 ms waiting for buffer to grow */
    }
}


