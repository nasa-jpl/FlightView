#ifndef LINEBUFFER_H
#define LINEBUFFER_H

#include <cstring>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <array>

// For debug builds:
#include <iostream>

class lineBuffer
{
public:
    lineBuffer(size_t numLines);
    ~lineBuffer();

    lineBuffer(const lineBuffer&) = delete;
    lineBuffer& operator=(const lineBuffer&) = delete;

    bool writeLine(const char* data, size_t length);
    bool readLine(char* dest, size_t& length);
    //size_t availableLines();
    size_t availableLinesToRead() const {
        return m_available.load(std::memory_order_acquire);
    }
    size_t unusedBufferLines() const {
        return m_numLines-m_available.load(std::memory_order_acquire);
    }
    size_t getBufferNumLines() {
        return m_numLines;
    }
    //static constexpr size_t maxLineSize();
    static constexpr size_t maxLineSize() {
        return MAX_LINE_SIZE;
    }

private:
    static const size_t MAX_LINE_SIZE = 200;
    enum class LineState {
        Free,       // Line can be written to
        Writing,    // Line is being written
        Ready,      // Line is ready to be read
        Reading     // Line is being read
    };
    char* m_buffer;
    size_t* m_lineSize;
    std::atomic<LineState>* m_lineState;  // State of each line
    size_t m_numLines;
    std::atomic<size_t> m_head;
    std::atomic<size_t> m_tail;
    std::atomic<size_t> m_available;
    std::mutex m_pointerMutex;  // Only used for head/tail updates
};

#endif // LINEBUFFER_H
