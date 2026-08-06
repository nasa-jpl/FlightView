// High-precision RTP frame replay server.
// Compile: g++ -O3 -march=native -o server-fast server-fast.cpp
//
// Differences from server.cpp, and why:
//   1. sendmmsg()      - one syscall per frame instead of one per packet.
//   2. scatter-gather  - iovec points straight at the mapped file; the 820 KB
//                        payload memcpy per frame is gone entirely.
//   3. absolute pacing - each frame targets startTime + n*period, so the timer
//                        overshoot that cost ~73 us/frame no longer accumulates.
//   4. hybrid sleep    - clock_nanosleep to just short of the deadline, then a
//                        short spin. Removes nanosleep granularity from jitter.
//   5. timer slack     - default 50 us slack is what made sleep_for(300ns) cost
//                        57 us. Set to 1 us.
//   6. mmap            - no 12 GB calloc-and-zero plus 12 GB read at startup.
//   7. no packet-pacing sleeps - the every-32-packets sleep in server.cpp cost
//                        7 x 57 us = ~400 us per frame of the 4444 us budget.
//
// Linux only (sendmmsg, clock_nanosleep, PR_SET_TIMERSLACK).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <algorithm>
#include <cmath>
#include <vector>

#define PORT 5004

#define DEFAULT_WIDTH   1280
#define DEFAULT_HEIGHT  328
#define DEFAULT_FPS     225.0
#define DEFAULT_PACKETS 128     // 6560 B payload; fits the 9710 B loopback MTU
#define DEFAULT_DEST    "127.0.0.1"

// How long before the deadline to stop sleeping and start spinning.
#define SPIN_MARGIN_NS  (120000)   // 120 us

static inline uint64_t nowNs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Sleep until an absolute CLOCK_MONOTONIC deadline, then spin the last stretch.
static inline void waitUntil(uint64_t deadlineNs) {
    uint64_t now = nowNs();
    if (deadlineNs <= now) return;

    if (deadlineNs - now > SPIN_MARGIN_NS) {
        uint64_t wake = deadlineNs - SPIN_MARGIN_NS;
        struct timespec ts;
        ts.tv_sec  = wake / 1000000000ULL;
        ts.tv_nsec = wake % 1000000000ULL;
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
    }
    while (nowNs() < deadlineNs) {
        __builtin_ia32_pause();
    }
}

static void printUsage(const char* p) {
    printf("\n=== RTP Replay Server (high-precision) ===\n\n");
    printf("Usage: %s [options] <filename>\n\n", p);
    printf("  -w, --width <px>      Frame width (default %d)\n", DEFAULT_WIDTH);
    printf("  -h, --height <px>     Frame height (default %d)\n", DEFAULT_HEIGHT);
    printf("  -f, --fps <rate>      Target frame rate (default %.1f)\n", DEFAULT_FPS);
    printf("  -p, --packets <n>     Packets per frame (default %d)\n", DEFAULT_PACKETS);
    printf("  -d, --dest <ip>       Destination address (default %s)\n", DEFAULT_DEST);
    printf("      --port <n>        Destination port (default %d)\n", PORT);
    printf("  -n, --frames <n>      Stop after n frames (default: run forever)\n");
    printf("      --rt              Request SCHED_FIFO priority (needs privilege)\n");
    printf("      --cpu <n>         Pin to CPU core n\n");
    printf("      --no-populate     Skip prefaulting the file (faster start, more jitter)\n");
    printf("      --help\n\n");
    printf("Example:\n");
    printf("  %s -w 1280 -h 328 -f 225 -p 128 --cpu 4 --rt scene.raw\n\n", p);
}

int main(int argc, char* argv[]) {
    uint16_t width = DEFAULT_WIDTH, height = DEFAULT_HEIGHT;
    double targetFPS = DEFAULT_FPS;
    int desiredPackets = DEFAULT_PACKETS;
    const char* filename = nullptr;
    const char* destIP = DEFAULT_DEST;
    int destPort = PORT;
    long maxFrames = -1;
    bool wantRT = false, populate = true;
    int pinCpu = -1;

    for (int i = 1; i < argc; ) {
        const char* a = argv[i];
        auto need = [&](void) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "Error: %s requires an argument\n", a); exit(1); }
            return argv[i + 1];
        };
        if (!strcmp(a, "-w") || !strcmp(a, "--width"))        { width = atoi(need()); i += 2; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--height"))  { height = atoi(need()); i += 2; }
        else if (!strcmp(a, "-f") || !strcmp(a, "--fps"))     { targetFPS = atof(need()); i += 2; }
        else if (!strcmp(a, "-p") || !strcmp(a, "--packets")) { desiredPackets = atoi(need()); i += 2; }
        else if (!strcmp(a, "-d") || !strcmp(a, "--dest"))    { destIP = need(); i += 2; }
        else if (!strcmp(a, "--port"))                        { destPort = atoi(need()); i += 2; }
        else if (!strcmp(a, "-n") || !strcmp(a, "--frames"))  { maxFrames = atol(need()); i += 2; }
        else if (!strcmp(a, "--rt"))                          { wantRT = true; i += 1; }
        else if (!strcmp(a, "--cpu"))                         { pinCpu = atoi(need()); i += 2; }
        else if (!strcmp(a, "--no-populate"))                 { populate = false; i += 1; }
        else if (!strcmp(a, "--help"))                        { printUsage(argv[0]); return 0; }
        else if (a[0] == '-') { fprintf(stderr, "Error: unknown option %s\n", a); printUsage(argv[0]); return 1; }
        else { filename = a; i += 1; }
    }

    if (!filename) { fprintf(stderr, "Error: no filename specified\n"); printUsage(argv[0]); return 1; }
    if (width <= 0 || height <= 0 || targetFPS <= 0 || desiredPackets <= 0) {
        fprintf(stderr, "Error: invalid geometry/rate arguments\n"); return 1;
    }

    const size_t frameSize = (size_t)width * height * 2;

    // Packets must divide the frame evenly, same convention as server.cpp.
    int chunksPerFrame = desiredPackets;
    while (chunksPerFrame > 1 && (frameSize % chunksPerFrame) != 0) chunksPerFrame--;
    const size_t bytesPerPacket = frameSize / chunksPerFrame;

    const uint64_t periodNs = (uint64_t)llround(1e9 / targetFPS);

    printf("\n=== RTP Replay Server Configuration ===\n");
    printf("Frame geometry:    %u x %u\n", width, height);
    printf("Frame size:        %zu bytes (%.1f KiB)\n", frameSize, frameSize / 1024.0);
    printf("Target frame rate: %.2f FPS\n", targetFPS);
    printf("Frame period:      %.2f us\n", periodNs / 1000.0);
    printf("Packets per frame: %d", chunksPerFrame);
    if (chunksPerFrame != desiredPackets) printf(" (requested %d, adjusted for even division)", desiredPackets);
    printf("\n");
    printf("Payload per packet:%zu bytes (UDP datagram %zu bytes)\n", bytesPerPacket, bytesPerPacket + 12);
    printf("Target data rate:  %.2f Mbps\n", (frameSize * 8 * targetFPS) / 1e6);
    printf("Destination:       %s:%d\n", destIP, destPort);
    printf("=======================================\n\n");

    if (bytesPerPacket + 12 + 28 > 9710) {
        printf("NOTE: datagram exceeds the 9710 B loopback MTU and will be IP-fragmented.\n"
               "      Increase --packets to avoid fragmentation.\n\n");
    }

    // Lock our own code/stack/heap resident before the file is mapped, so this
    // does not try to pin the whole (potentially multi-GB) frame file.
    // Prefaulting the file itself is MAP_POPULATE's job below.
    mlockall(MCL_CURRENT);

    // --- map the file -------------------------------------------------------
    int fd = open(filename, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); return 1; }
    const size_t fileLen = st.st_size;
    const size_t framesInFile = fileLen / frameSize;
    if (framesInFile == 0) {
        fprintf(stderr, "Error: file is smaller than one frame (%zu < %zu)\n", fileLen, frameSize);
        return 1;
    }

    printf("Mapping %s (%.2f GiB, %zu whole frames)...\n",
           filename, fileLen / 1073741824.0, framesInFile);
    int mapFlags = MAP_PRIVATE | (populate ? MAP_POPULATE : 0);
    uint8_t* imageData = (uint8_t*)mmap(NULL, fileLen, PROT_READ, mapFlags, fd, 0);
    if (imageData == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(imageData, fileLen, MADV_SEQUENTIAL);
    printf("\tDone.%s\n", populate ? " (prefaulted)" : "");

    // --- socket -------------------------------------------------------------
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) { perror("socket"); return 1; }

    int sndbuf = 64 * 1024 * 1024;
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    int actualSnd = 0; socklen_t ol = sizeof(actualSnd);
    getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &actualSnd, &ol);
    printf("Socket send buffer: %d bytes (requested %d)\n", actualSnd, sndbuf);
    if (actualSnd < sndbuf) {
        printf("  -> capped by net.core.wmem_max. Raise it:\n"
               "     sudo sysctl -w net.core.wmem_max=134217728\n");
    }

    sockaddr_in servaddr; memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(destPort);
    if (inet_pton(AF_INET, destIP, &servaddr.sin_addr) != 1) {
        fprintf(stderr, "Error: bad destination address %s\n", destIP); return 1;
    }
    // connect() lets the kernel skip route lookup on every send.
    if (connect(sockfd, (sockaddr*)&servaddr, sizeof(servaddr)) < 0) { perror("connect"); return 1; }

    // --- scheduling ---------------------------------------------------------
    prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);   // 1 us instead of the 50 us default

    if (pinCpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(pinCpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) == 0) printf("Pinned to CPU %d\n", pinCpu);
        else perror("sched_setaffinity");
    }
    if (wantRT) {
        struct sched_param sp; sp.sched_priority = 80;
        if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) printf("SCHED_FIFO priority 80 acquired\n");
        else printf("WARNING: SCHED_FIFO not granted (%s). Run with sudo or grant CAP_SYS_NICE.\n",
                    strerror(errno));
    }

    // --- preallocate the message batch --------------------------------------
    // One mmsghdr per packet, each with two iovecs: our 12-byte header and a
    // pointer directly into the mapped file. Nothing is copied per frame.
    std::vector<mmsghdr> msgs(chunksPerFrame);
    std::vector<iovec>   iovs(chunksPerFrame * 2);
    std::vector<uint8_t> headers(chunksPerFrame * 12, 0);

    const uint8_t ver = 2, payloadType = 1, csrcCount = 0;
    const uint32_t ssrc = 0xdeadbeef;

    for (int c = 0; c < chunksPerFrame; c++) {
        uint8_t* h = &headers[c * 12];
        h[0] = (ver << 6) | (csrcCount & 0x0F);       // no padding, no extension
        h[1] = payloadType & 0x7f;                     // marker set per frame below
        h[8]  = (ssrc >> 24) & 0xff;
        h[9]  = (ssrc >> 16) & 0xff;
        h[10] = (ssrc >> 8)  & 0xff;
        h[11] =  ssrc        & 0xff;

        iovs[c * 2 + 0].iov_base = h;
        iovs[c * 2 + 0].iov_len  = 12;
        iovs[c * 2 + 1].iov_len  = bytesPerPacket;     // iov_base set per frame

        memset(&msgs[c], 0, sizeof(mmsghdr));
        msgs[c].msg_hdr.msg_iov    = &iovs[c * 2];
        msgs[c].msg_hdr.msg_iovlen = 2;
    }
    // Marker bit marks the last packet of each frame; it never moves.
    headers[(chunksPerFrame - 1) * 12 + 1] = 0x80 | (payloadType & 0x7f);

    // --- main loop ----------------------------------------------------------
    printf("Sending frames...\n\n");

    uint16_t sequenceNumber = 0;
    uint32_t timestamp = 0;
    uint64_t framesSent = 0, packetsSent = 0, bytesSentTotal = 0;
    uint64_t sendErrors = 0, shortBatches = 0, lateFrames = 0;
    size_t frameIndex = 0;

    std::vector<double> lateness;      // how far past its deadline each frame went out
    lateness.reserve(1 << 20);

    const uint64_t t0 = nowNs();
    uint64_t deadline = t0;

    while (maxFrames < 0 || (long)framesSent < maxFrames) {
        waitUntil(deadline);

        const uint8_t* frameImage = imageData + frameIndex * frameSize;

        // Per-packet header fields that change, plus the payload pointer.
        for (int c = 0; c < chunksPerFrame; c++) {
            uint8_t* h = &headers[c * 12];
            h[2] = (sequenceNumber >> 8) & 0xff;
            h[3] =  sequenceNumber       & 0xff;
            h[4] = (timestamp >> 24) & 0xff;
            h[5] = (timestamp >> 16) & 0xff;
            h[6] = (timestamp >> 8)  & 0xff;
            h[7] =  timestamp        & 0xff;
            sequenceNumber++;
            iovs[c * 2 + 1].iov_base = (void*)(frameImage + (size_t)c * bytesPerPacket);
        }

        // One syscall for the whole frame.
        int sent = sendmmsg(sockfd, msgs.data(), chunksPerFrame, 0);
        if (sent < 0) {
            sendErrors++;
            if (sendErrors < 10) fprintf(stderr, "sendmmsg failed: %s\n", strerror(errno));
        } else {
            if (sent < chunksPerFrame) shortBatches++;
            packetsSent += sent;
            for (int c = 0; c < sent; c++) bytesSentTotal += msgs[c].msg_len;
        }

        const uint64_t after = nowNs();
        double lateUs = (double)((int64_t)after - (int64_t)deadline) / 1000.0;
        lateness.push_back(lateUs);
        if (lateUs > periodNs / 1000.0) lateFrames++;

        timestamp++;
        framesSent++;
        frameIndex++;
        if (frameIndex >= framesInFile) frameIndex = 0;   // loop the file

        deadline += periodNs;
        // If we have fallen more than a frame behind, resync rather than
        // sprinting to catch up (which would burst the receiver).
        if ((int64_t)(after - deadline) > (int64_t)periodNs) deadline = after + periodNs;

        if ((framesSent % 2000) == 0) {
            double el = (nowNs() - t0) / 1e9;
            printf("frames=%lu  avg=%.2f FPS  late frames=%lu  send errors=%lu\n",
                   framesSent, framesSent / el, lateFrames, sendErrors);
            fflush(stdout);
        }
    }

    const double elapsed = (nowNs() - t0) / 1e9;
    printf("\n=== TX SUMMARY ===\n");
    printf("elapsed:        %.3f s\n", elapsed);
    printf("frames sent:    %lu  (%.3f FPS average)\n", framesSent, framesSent / elapsed);
    printf("packets sent:   %lu\n", packetsSent);
    printf("data rate:      %.3f Gbit/s\n", bytesSentTotal * 8.0 / elapsed / 1e9);
    printf("late frames:    %lu (%.3f%%)\n", lateFrames, 100.0 * lateFrames / framesSent);
    printf("send errors:    %lu, short batches: %lu\n", sendErrors, shortBatches);
    if (lateness.size() > 10) {
        std::sort(lateness.begin(), lateness.end());
        printf("send lateness vs deadline (us): p50=%.1f p99=%.1f p99.9=%.1f max=%.1f\n",
               lateness[lateness.size() / 2],
               lateness[(size_t)(lateness.size() * 0.99)],
               lateness[(size_t)(lateness.size() * 0.999)],
               lateness.back());
    }

    munmap(imageData, fileLen);
    close(fd);
    close(sockfd);
    return 0;
}
