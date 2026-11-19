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
#include <variant>
#include <vector>

namespace ovms {

class MetricCounter;
class MetricGauge;
class MetricHistogram;
class MetricRegistry;

/// A thread-safe custom counter handle. The registry owns the series, which
/// persists across graph unload and reload.
class CalculatorCounter {
    std::unique_ptr<MetricCounter> metric;
    CalculatorCounter(MetricRegistry* registry, const std::string& name, const std::string& help,
        const std::string& graphName, const std::string& nodeName, bool* newlyRegistered);
    friend class CalculatorMetrics;

public:
    CalculatorCounter(const CalculatorCounter&) = delete;
    CalculatorCounter& operator=(const CalculatorCounter&) = delete;
    ~CalculatorCounter();
    /// Add a finite, non-negative value. Invalid values are logged and ignored.
    void increment(double value = 1.0);
};

/// A thread-safe custom gauge handle. The registry owns the series, which
/// persists across graph unload and reload.
class CalculatorGauge {
    std::unique_ptr<MetricGauge> metric;
    CalculatorGauge(MetricRegistry* registry, const std::string& name, const std::string& help,
        const std::string& graphName, const std::string& nodeName, bool* newlyRegistered);
    friend class CalculatorMetrics;

public:
    CalculatorGauge(const CalculatorGauge&) = delete;
    CalculatorGauge& operator=(const CalculatorGauge&) = delete;
    ~CalculatorGauge();
    /// Set a finite value. Invalid values are logged and ignored.
    void set(double value);
    /// Add a finite signed value, including zero. Invalid values are logged and ignored.
    void add(double value);
};

/// A thread-safe custom histogram handle. The registry owns the series, which
/// persists across graph unload and reload.
class CalculatorHistogram {
    std::unique_ptr<MetricHistogram> metric;
    CalculatorHistogram(MetricRegistry* registry, const std::string& name, const std::string& help,
        const std::string& graphName, const std::string& nodeName, const std::vector<double>& bucketBoundaries,
        bool* newlyRegistered);
    friend class CalculatorMetrics;

public:
    CalculatorHistogram(const CalculatorHistogram&) = delete;
    CalculatorHistogram& operator=(const CalculatorHistogram&) = delete;
    ~CalculatorHistogram();
    /// Record a finite observation. Invalid values are logged and ignored.
    void observe(double value);
};

/// One graph/node registration scope, shared by all instances of that calculator.
/// A scope created again for the same graph and node, for example after a graph
/// reload, continues the existing series. The registry must outlive updates made
/// through the scope's handles.
class CalculatorMetrics {
    using MetricHandle = std::variant<
        std::shared_ptr<CalculatorCounter>,
        std::shared_ptr<CalculatorGauge>,
        std::shared_ptr<CalculatorHistogram>>;

    struct Registration {
        std::string help;
        std::vector<double> bucketBoundaries;
        MetricHandle metricHandle;
    };
    std::mutex scopeMutex;
    MetricRegistry* const registry;
    const std::string graphName;
    const std::string nodeName;
    std::map<std::string, Registration> metrics;

    template <typename HandleType>
    std::shared_ptr<HandleType> findCompatibleRegistration(const std::string& name, const std::string& help,
        const std::vector<double>& bucketBoundaries = {}) const;

public:
    CalculatorMetrics(MetricRegistry* registry, const std::string& graphName, const std::string& nodeName);
    /// Get an existing handle or register one (usually in Open).
    /// Counters must be named custom_<name>_total;
    /// gauges and histograms must start with custom_. Names and help text must
    /// match across registrations. Disabled metrics return valid no-op handles.
    std::shared_ptr<CalculatorCounter> getOrRegisterCounter(const std::string& name, const std::string& help);
    std::shared_ptr<CalculatorGauge> getOrRegisterGauge(const std::string& name, const std::string& help);
    /// Histogram bucket boundaries must be finite and strictly increasing.
    std::shared_ptr<CalculatorHistogram> getOrRegisterHistogram(const std::string& name, const std::string& help,
        const std::vector<double>& bucketBoundaries);
};

using CalculatorMetricsMap = std::map<std::string, std::shared_ptr<CalculatorMetrics>>;
inline constexpr const char* CALCULATOR_METRICS_TAG = "CALCULATOR_METRICS";
inline constexpr const char* CALCULATOR_METRICS_PACKET_NAME = "calculator_metrics";

}  // namespace ovms
