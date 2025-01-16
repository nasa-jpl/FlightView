#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h> // For inet_ntoa

constexpr int MAX_BUFFER_SIZE = 1024;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <port>\n";
        return 1;
    }

    int port = std::stoi(argv[1]);

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error creating socket\n";
        return 1;
    }

    sockaddr_in serverAddress{};
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = INADDR_ANY;
    serverAddress.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) < 0) {
        std::cerr << "Error binding socket\n";
        close(sockfd);
        return 1;
    }

    std::cout << "Listening on UDP port " << port << "...\n";

    char buffer[MAX_BUFFER_SIZE];
    sockaddr_in clientAddress{};
    socklen_t clientAddressLength = sizeof(clientAddress);

    while (true) {
        ssize_t bytesReceived = recvfrom(sockfd, buffer, MAX_BUFFER_SIZE - 1, 0,
                                         (struct sockaddr*)&clientAddress, &clientAddressLength);

        if (bytesReceived < 0) {
            std::cerr << "Error receiving data\n";
            close(sockfd);
            return 1;
        }

        buffer[bytesReceived] = '\0'; // Null-terminate the received data
        
        // Convert client IP address to string
        char clientIp[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(clientAddress.sin_addr), clientIp, INET_ADDRSTRLEN);

        std::cout << "Received " << bytesReceived << " bytes from " << clientIp << ":" << ntohs(clientAddress.sin_port) << ": ";
        std::cout << std::endl;
        // Print only printable ASCII characters
        for (ssize_t i = 0; i < bytesReceived; ++i) {
            if (buffer[i] >= 32 && buffer[i] <= 126) { // Printable ASCII range
                std::cout << buffer[i];
            } else if (buffer[i] == 0x0a) {
                std::cout << std::endl;
            } else {
                std::cout << "."; // Replace non-printable characters with dots
            }
        }
        std::cout << "\n";
    }

    close(sockfd); // This won't be reached in the infinite loop, but it's good practice
    return 0;
}
