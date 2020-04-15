#pragma once

#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>


class Timer {
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> startTimestamps;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> stopTimestamps;
public:

    void start(const std::string& name) {
#ifdef DEBUG
        startTimestamps[name] = std::chrono::high_resolution_clock::now();
#endif
    }

    void stop(const std::string& name) {
#ifdef DEBUG
        stopTimestamps[name] = std::chrono::high_resolution_clock::now();
#endif
    }

    void print() {
#ifdef DEBUG
        std::cout << "-----\n";
        for (const auto& pair : stopTimestamps) {
            const auto& name = pair.first;
            auto elapsed = std::to_string(std::chrono::duration<double, std::milli>(stopTimestamps[name] - startTimestamps[name]).count()); 

            std::cout 
                << name
                << ": "
                << elapsed
                << "ms"
                << "\n";
        }
#endif
    }

};
