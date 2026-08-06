all: server server-fast pcap-rtp-sender

server: server.cpp
	g++ -o server -march=native -O3 server.cpp

# Linux-only: uses sendmmsg, clock_nanosleep, PR_SET_TIMERSLACK.
server-fast: server-fast.cpp
	g++ -o server-fast -march=native -O3 server-fast.cpp

pcap-rtp-sender: pcap-rtp-sender.cpp
	g++ -o pcap-rtp-sender -march=native -O3 pcap-rtp-sender.cpp

clean:
	rm -f server server-fast pcap-rtp-sender
