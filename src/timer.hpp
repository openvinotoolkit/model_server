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
#include <string>
#include <unordered_map>

namespace ovms {

template <typename T>
struct is_chrono_duration_type : std::false_type {};

template <typename T, typename U>
struct is_chrono_duration_type<std::chrono::duration<T, U>> : std::true_type {};

typedef unsigned int SIZE_TYPE;
template <SIZE_TYPE N>
class Timer {
    std::array<std::chrono::high_resolution_clock::time_point, N> startTimestamps;
    std::array<std::chrono::high_resolution_clock::time_point, N> stopTimestamps;

public:
    void start(SIZE_TYPE i) {
        startTimestamps[i] = std::chrono::high_resolution_clock::now();
    }

    void stop(SIZE_TYPE i) {
        stopTimestamps[i] = std::chrono::high_resolution_clock::now();
    }

    template <typename T>
    double elapsed(SIZE_TYPE i) {
        static_assert(is_chrono_duration_type<T>::value, "Non supported type.");
        return std::chrono::duration_cast<T>(stopTimestamps[i] - startTimestamps[i]).count();
    }
};
}  // namespace ovms
