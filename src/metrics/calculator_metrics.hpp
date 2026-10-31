//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace ovms {

class MetricCounter;
class MetricRegistry;
template <typename T>
class MetricFamily;

/// A thread-safe custom counter handle. Updates after graph unload are ignored.
class CalculatorCounter {
    std::mutex mutex;
    std::shared_ptr<MetricFamily<MetricCounter>> family;
    std::unique_ptr<MetricCounter> metric;
    CalculatorCounter(MetricRegistry* registry, const std::string& name, const std::string& help,
        const std::string& graphName, const std::string& nodeName);
    void deactivate();
    friend class CalculatorMetrics;

public:
    ~CalculatorCounter();
    /// Add a finite, non-negative value. Invalid values throw std::invalid_argument.
    void increment(double value = 1.0);
};

/// One graph/node registration scope, shared by all instances of that calculator.
/// The registry must outlive the scope, or deactivate() must be called first.
class CalculatorMetrics {
    struct Registration {
        std::string help;
        std::shared_ptr<CalculatorCounter> counter;
    };
    std::mutex mutex;
    MetricRegistry* registry;
    const std::string graphName;
    const std::string nodeName;
    bool active = true;
    std::map<std::string, Registration> counters;

public:
    CalculatorMetrics(MetricRegistry* registry, const std::string& graphName, const std::string& nodeName);
    ~CalculatorMetrics();
    /// Register once (usually in Open). Repeated identical registration returns
    /// the same handle. Names must start with custom_ and end with _total.
    /// Disabled metrics return a valid no-op handle; malformed registrations throw.
    std::shared_ptr<CalculatorCounter> registerCounter(const std::string& name, const std::string& help);
    /// Remove this scope's series and make retained handles safe no-ops.
    void deactivate();
};

using CalculatorMetricsMap = std::map<std::string, std::shared_ptr<CalculatorMetrics>>;
inline constexpr const char* CALCULATOR_METRICS_TAG = "CALCULATOR_METRICS";
inline constexpr const char* CALCULATOR_METRICS_PACKET_NAME = "calculator_metrics";

}  // namespace ovms
