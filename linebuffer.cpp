#include "linebuffer.h"

lineBuffer::lineBuffer(size_t numLines) :
    m_numLines(numLines),
    m_head(0),
    m_tail(0),
    m_available(0) {
    if (numLines == 0) {
        throw std::invalid_argument("Number of lines must be greater than 0");
    }
    m_buffer = new char[numLines * MAX_LINE_SIZE];
    m_lineSize = new size_t[numLines];
    m_lineState = new std::atomic<LineState>[numLines];

    // Initialize all lines as free
    for (size_t i = 0; i < numLines; i++) {
        m_lineState[i].store(LineState::Free, std::memory_order_release);
    }
}

lineBuffer::~lineBuffer() {
    delete[] m_buffer;
    delete[] m_lineSize;
    delete[] m_lineState;
}

bool lineBuffer::writeLine(const char* data, size_t length) {
    if (length >= MAX_LINE_SIZE) {
        length = MAX_LINE_SIZE-1; // write the first length-1 bytes anyway.
        //return false;
    }

    // Get next write position
    size_t currentHead;
    {
        std::lock_guard<std::mutex> lock(m_pointerMutex);
        if (m_available.load(std::memory_order_acquire) >= m_numLines) {
            return false;  // Buffer is full
        }
        currentHead = m_head;
        m_head = (m_head + 1) % m_numLines;
    }

    // Mark line as being written
    LineState expectedState = LineState::Free;
    if (!m_lineState[currentHead].compare_exchange_strong(expectedState,
                                                          LineState::Writing,
                                                          std::memory_order_acquire,
                                                          std::memory_order_relaxed)) {
        return false;  // Line is not free
    }

    // Copy data
    char* writePos = m_buffer + (currentHead * MAX_LINE_SIZE);
    std::memcpy(writePos, data, length);
    m_lineSize[currentHead] = length;

    // Mark line as ready to read and increment available count
    m_lineState[currentHead].store(LineState::Ready, std::memory_order_release);
    m_available.fetch_add(1, std::memory_order_release);

    return true;
}

bool lineBuffer::readLine(char* dest, size_t& length) {
    // The calling function must allocate the dest pointer,
    // and monitor the return value bool for success.
    // The length of the data written may be found in the length parameter's
    // modified value upon return.

    //std::cout << "start readLine. Current size used is: " << m_available << std::endl;

    if(!dest)
        return false;

    // Get current read position
    size_t currentTail;
    {
        std::lock_guard<std::mutex> lock(m_pointerMutex);
        if (m_available.load(std::memory_order_acquire) == 0) {
            return false;  // Buffer is empty
        }
        currentTail = m_tail;
        m_tail = (m_tail + 1) % m_numLines;
    }

    // Try to mark line as being read
    LineState expectedState = LineState::Ready;
    if (!m_lineState[currentTail].compare_exchange_strong(expectedState,
                                                          LineState::Reading,
                                                          std::memory_order_acquire,
                                                          std::memory_order_relaxed)) {
        return false;  // Line is not ready
    }

    // Copy data
    char* readPos = m_buffer + (currentTail * MAX_LINE_SIZE);
    length = m_lineSize[currentTail];
    std::memcpy(dest, readPos, length);

    // Mark line as free and decrement available count
    m_lineState[currentTail].store(LineState::Free, std::memory_order_release);
    m_available.fetch_sub(1, std::memory_order_release);
    //std::cout << "ending readLine. Current size used is: " << m_available << std::endl;

    return true;
}




