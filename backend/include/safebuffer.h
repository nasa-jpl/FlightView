#include <atomic>
#include <memory>
// include <thread>
#include <iostream>
// include <chrono>
// include <vector>
// include <queue>

template<typename T, size_t Size>
class LockFreeRingBuffer {
private:
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
    struct alignas(64) memSlot_t {  // Cache line alignment
        std::atomic<T*> data{nullptr};
        std::atomic<bool> ready{false};
    };
    
    memSlot_t memSlots[Size];
    alignas(64) std::atomic<size_t> write_pos{0};  // Producer position
    alignas(64) std::atomic<size_t> read_pos{0};   // Consumer position
    int overrides=0;
    int emptyRequests = 0;
    
public:
    // Producer: Try to add, returns false if buffer full

    bool try_enqueue(T* item) {
        const size_t current_write = write_pos.load(std::memory_order_relaxed);
        const size_t current_read = read_pos.load(std::memory_order_acquire);

        // Check if buffer is full using proper wraparound arithmetic
        if (current_write - current_read >= Size) {
            return false;
        }

        memSlot_t& memSlot = memSlots[current_write & (Size - 1)];

        // Double-check slot availability
        if (memSlot.ready.load(std::memory_order_acquire)) {
            return false;  // Buffer full
        }

        memSlot.data.store(item, std::memory_order_relaxed);
        memSlot.ready.store(true, std::memory_order_release);
        write_pos.store(current_write + 1, std::memory_order_relaxed);
        return true;
    }

    bool try_enqueue_unsafe(T* item) {
        const size_t pos = write_pos.load(std::memory_order_relaxed);
        memSlot_t& memSlot = memSlots[pos & (Size - 1)];
        
        // Check if slot is available
        if (memSlot.ready.load(std::memory_order_acquire)) {
            return false;  // Buffer full
        }
        
        memSlot.data.store(item, std::memory_order_relaxed);
        memSlot.ready.store(true, std::memory_order_release);
        write_pos.store(pos + 1, std::memory_order_relaxed);
        return true;
    }
    
    // Producer: Add item, overwrite oldest if buffer full (for real-time systems)
    void enqueue_overwrite(T* item) {
        const size_t current_write = write_pos.load(std::memory_order_relaxed);
        const size_t current_read = read_pos.load(std::memory_order_acquire);

        memSlot_t& memSlot = memSlots[current_write & (Size - 1)];

        // If we're about to overwrite and buffer is full, advance read position
        if (current_write - current_read >= Size) {
            // We're overwriting the oldest unread item
            const size_t old_read = current_read;
            const size_t new_read = current_write - Size + 1;

            // Try to advance read position atomically
            if (read_pos.compare_exchange_weak(const_cast<size_t&>(old_read), new_read,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
                overrides++;
            }
        }

        // Handle old data if present
        if (memSlot.ready.load(std::memory_order_acquire)) {
            T* old_data = memSlot.data.load(std::memory_order_relaxed);
            if (old_data) {
                delete old_data;
            }
        }

        memSlot.data.store(item, std::memory_order_relaxed);
        memSlot.ready.store(true, std::memory_order_release);
        write_pos.store(current_write + 1, std::memory_order_relaxed);
    }

    void enqueue_overwrite_unsafe(T* item) {
        const size_t pos = write_pos.load(std::memory_order_relaxed);
        memSlot_t& memSlot = memSlots[pos & (Size - 1)];
        
        // If slot occupied, we're overwriting (frame drop)
        if (memSlot.ready.load(std::memory_order_acquire)) {
            // Optionally handle overwrite (e.g., delete old data, count drops)
            T* old_data = memSlot.data.load(std::memory_order_relaxed);
            if (old_data) {
                delete old_data;  // Free overwritten data
                overrides++;
                // std::cerr << "[WARNING] Buffer full, dropping frame\n";
            }
        }
        
        memSlot.data.store(item, std::memory_order_relaxed);
        memSlot.ready.store(true, std::memory_order_release);
        write_pos.store(pos + 1, std::memory_order_relaxed);
    }
    
    // Consumer: Try to get item, returns nullptr if empty
    T* try_dequeue() {
        const size_t pos = read_pos.load(std::memory_order_relaxed);
        memSlot_t& memSlot = memSlots[pos & (Size - 1)];
        
        if (!memSlot.ready.load(std::memory_order_acquire)) {
            emptyRequests++;
            return nullptr;  // Buffer empty
        }
        
        T* data = memSlot.data.load(std::memory_order_relaxed);
        memSlot.ready.store(false, std::memory_order_release);
        read_pos.store(pos + 1, std::memory_order_relaxed);
        return data;
    }
    
    size_t size() const {
        return write_pos.load(std::memory_order_relaxed) - 
               read_pos.load(std::memory_order_relaxed);
    }

    size_t getWritePos() const {
        return write_pos.load(std::memory_order_relaxed);
    }

    size_t getReadPos() const {
        return read_pos.load(std::memory_order_relaxed);
    }
    
    bool empty() const {
        return size() == 0;
    }
    
    bool full() const {
        return size() >= Size;
    }

    int getOverrideCount() {
        return overrides;
    }

    int getEmptyRequestCount() {
        return emptyRequests;
    }

    void clearStats() {
        emptyRequests = 0;
        overrides = 0;
    }
};
