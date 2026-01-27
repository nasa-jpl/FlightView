all: server pcap-rtp-sender

server: server.cpp
	g++ -o server -march=native -O3 server.cpp

pcap-rtp-sender: pcap-rtp-sender.cpp
	g++ -o pcap-rtp-sender -march=native -O3 pcap-rtp-sender.cpp

clean:
	rm -f server pcap-rtp-sender

