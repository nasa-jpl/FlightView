#include <list>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <iostream>
#include <chrono>
#include <atomic>

class safelist {
private:
    std::list<uint16_t*> saving_list;
    mutable std::mutex list_mutex;
    std::condition_variable data_available;
    std::atomic<bool> shutdown{false};

public:
    void addData(uint16_t* data);
    uint16_t* extractData();
    uint16_t* extractDataWithTimeout(std::chrono::milliseconds timeout);
    void requestShutdown() {
        shutdown.store(true);
        data_available.notify_all();
    }
    bool empty() const {
        // aka "isEmpty()"
        std::lock_guard<std::mutex> lock(list_mutex);
        return saving_list.empty();
    }

};
