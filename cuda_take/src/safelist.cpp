#include "safelist.h"

safelist::addData(uint16_t *data) {
    {
        std::lock_guard<std::mutex> lock(list_mutex);
        saving_list.push_back(data);
        std::cout << "Added data: " << data << std::endl;
    }
    data_available.notify_one(); // Wake up waiting consumer
}

uint16_t* safelist::extractData() {
    std::unique_lock<std::mutex> lock(list_mutex);

    // Wait until data is available or shutdown is requested
    data_available.wait(lock, [this] {
        return !saving_list.empty() || shutdown.load();
    });

    if (saving_list.empty()) {
        return nullptr; // Shutdown was requested
    }

    uint16_t* data = saving_list.front();
    saving_list.pop_front();
    std::cout << "Extracted data: " << data << std::endl;
    return data;
}

uint16_t* safelist::extractDataWithTimeout(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(list_mutex);

    if (data_available.wait_for(lock, timeout, [this] {
        return !saving_list.empty() || shutdown.load();
    })) {
        if (!saving_list.empty()) {
            uint16_t* data = saving_list.front();
            saving_list.pop_front();
            return data;
        }
    }
    return nullptr; // Timeout or shutdown
}
