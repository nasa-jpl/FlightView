#include "zmqclient.h"
#include <QDebug>
#include <QByteArray>
#include <QThread>

// ZMQ requires these headers for implementation details
#include <zmq.hpp>

// --- Helper Functions ---

/**
 * @brief Utility function to safely clean up existing ZMQ resources.
 */
void ZmqClient::cleanupZmq()
{
    if (m_socket) {
        // Set linger to 0 to prevent blocking during close()
        int linger = 0;
        m_socket->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
        m_socket.reset();
    }
    m_context.reset();
    m_isConnected = false;
}

// --- Class Implementation ---

ZmqClient::ZmqClient(QObject *parent)
    : QObject(parent)
{
    // ZMQ context and socket are intentionally not initialized here.
    // They must be initialized in the target thread (in setConnection).
    qRegisterMetaType<QString>("QString");
}

ZmqClient::~ZmqClient()
{
    // Ensure all resources are cleaned up safely
    cleanupZmq();
}

/**
 * @brief Helper to send the atomic two-part message: [Type] + [Payload]
 */
bool ZmqClient::sendMultipart(const QString& type, const QByteArray& payload)
{
    if (!m_isConnected || !m_socket) {
        // Drop the message silently if not connected, which is the ZMQ PUB philosophy.
        return false;
    }

    try {
        // --- Frame 1: Type Identifier (Text) ---
        // Convert QString to UTF-8 QByteArray for ZMQ sending
        QByteArray typeData = type.toUtf8();
        zmq::message_t type_frame(typeData.constData(), typeData.size());

        // Use SNDMORE flag to indicate another frame is coming
        m_socket->send(type_frame, zmq::send_flags::sndmore);

        // --- Frame 2: Payload (Binary or Text) ---
        zmq::message_t payload_frame(payload.constData(), payload.size());

        // No SNDMORE flag, this is the last frame
        m_socket->send(payload_frame, zmq::send_flags::none);

        return true;
    } catch (const zmq::error_t& e) {
        // If a ZMQ error occurs, emit a signal back to the main thread
        emit errorOccurred(QString("ZMQ send failed: %1").arg(e.what()));
        return false;
    }
}


// --- Public Slots (Called from GUI/Other Threads via signal/slot) ---

void ZmqClient::setConnection(const QString& host, int port)
{
    // CRITICAL: This method MUST run on the thread the ZmqClient object is in!
    if (QThread::currentThread() != this->thread()) {
        qCritical() << "ZmqClient::setConnection called from the wrong thread! Use QMetaObject::invokeMethod.";
        return;
    }

    if (m_isConnected) {
        cleanupZmq();
    }

    m_host = host;
    m_port = port;
    QString endpoint = QString("tcp://%1:%2").arg(m_host).arg(m_port);

    try {
        // 1. Initialize Context and PUB Socket
        m_context = std::unique_ptr<zmq::context_t>(new zmq::context_t(1));
        m_socket = std::unique_ptr<zmq::socket_t>(new zmq::socket_t(*m_context, zmq::socket_type::pub));
        //m_context = std::make_unique<zmq::context_t>(1);
        //m_socket = std::make_unique<zmq::socket_t>(*m_context, zmq::socket_type::pub);

        // 2. Set Linger to 0: Ensures closeConnection() won't block
        int linger = 0;
        m_socket->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));

        // 3. Connect to the server
        m_socket->connect(endpoint.toStdString());

        m_isConnected = true;
        emit connectionStatus(true, QString("Successfully connected ZMQ PUB to %1").arg(endpoint));

    } catch (const zmq::error_t& e) {
        cleanupZmq();
        emit connectionStatus(false, QString("Failed to connect ZMQ PUB to %1: %2").arg(endpoint).arg(e.what()));
        emit errorOccurred(QString("ZMQ connection error: %1").arg(e.what()));
    }
}

void ZmqClient::sendText(const QString& type, const QString& payload)
{
    // Convert QString to QByteArray (UTF-8)
    QByteArray payloadData = payload.toUtf8();

    if (sendMultipart(type, payloadData)) {
        // Optional: Debugging output within the worker thread
        // qDebug() << QThread::currentThreadId() << "Sent TEXT:" << type;
    }
}

void ZmqClient::sendBinary(const QString& type, const char* data, int size)
{
    // Create QByteArray from the raw pointer and size
    QByteArray payloadData(data, size);

    if (sendMultipart(type, payloadData)) {
        // Optional: Debugging output within the worker thread
        // qDebug() << QThread::currentThreadId() << "Sent BINARY:" << type << "Size:" << size;
    }
}

void ZmqClient::closeConnection()
{
    // CRITICAL: This method MUST run on the thread the ZmqClient object is in!
    if (QThread::currentThread() != this->thread()) {
        qCritical() << "ZmqClient::closeConnection called from the wrong thread! Use QMetaObject::invokeMethod.";
        return;
    }

    if (m_isConnected) {
        cleanupZmq();
        emit connectionStatus(false, "Connection explicitly closed.");
    }
}
