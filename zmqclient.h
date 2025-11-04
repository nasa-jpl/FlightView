#pragma once

#include <QObject>
#include <QString>
#include <QThread>
#include <memory> // For std::unique_ptr

// Forward declarations to avoid heavy includes in the header
namespace zmq {
    class context_t;
    class socket_t;
}

/**
 * @brief ZmqClient is a worker object for ZeroMQ communications.
 * * This object is designed to be moved to its own QThread because ZMQ operations
 * (like connect and send) must run on a single, dedicated thread.
 * * It uses the PUB socket type for fire-and-forget broadcasting.
 */
class ZmqClient : public QObject
{
    Q_OBJECT

public:
    explicit ZmqClient(QObject *parent = nullptr);
    ~ZmqClient();

private:
    // ZMQ members must be managed carefully. Using unique_ptr for cleanup.
    std::unique_ptr<zmq::context_t> m_context;
    std::unique_ptr<zmq::socket_t> m_socket;

    QString m_host;
    int m_port = 0;

    bool m_isConnected = false;

    // Helper method to send the two-part message over ZMQ
    bool sendMultipart(const QString& type, const QByteArray& payload);

    // Internal method to safely clean up ZMQ resources
    void cleanupZmq();

signals:
    /**
     * @brief Emitted when the connection status changes (must be connected
     * before sending data).
     * @param isConnected True if the socket is successfully connected.
     * @param message Status or error message.
     */
    void connectionStatus(bool isConnected, const QString& message);

    /**
     * @brief Emitted when a critical error occurs (e.g., ZMQ exception).
     * @param errorMessage Description of the error.
     */
    void errorOccurred(const QString& errorMessage);

public slots:
    /**
     * @brief Sets the connection parameters and attempts to connect the socket.
     * * This slot should be connected to the QThread::started() signal
     * or called once the object is in its dedicated thread.
     * @param host IP address or hostname.
     * @param port Port number.
     */
    void setConnection(const QString& host, int port);

    /**
     * @brief Sends a log or text message payload as a multi-part ZMQ message.
     * @param type The message type identifier (e.g., "LOG", "STATUS").
     * @param payload The text payload.
     */
    void sendText(const QString& type, const QString& payload);

    /**
     * @brief Sends raw binary data as a multi-part ZMQ message.
     * @param type The message type identifier (e.g., "TELEM", "PIXEL").
     * @param data Pointer to the binary data buffer.
     * @param size Size of the data buffer in bytes.
     */
    void sendBinary(const QString& type, const char* data, int size);

    /**
     * @brief Slot to explicitly close the ZMQ socket and context.
     * Should be called when the application shuts down.
     */
    void closeConnection();
};
