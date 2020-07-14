//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

template <typename T>
struct is_chrono_duration_type : std::false_type {};

template <typename T, typename U>
struct is_chrono_duration_type<std::chrono::duration<T, U>> : std::true_type {};

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

    template <typename T>
    double elapsed(const std::string& name) {
        static_assert(is_chrono_duration_type<T>::value, "Non supported type.");
        double duration_us = 0;
#ifdef DEBUG
        duration_us = std::chrono::duration_cast<T>(stopTimestamps[name] - startTimestamps[name]).count();
#endif
        return duration_us;
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
