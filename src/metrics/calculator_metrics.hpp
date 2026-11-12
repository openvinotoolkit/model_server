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
template <typename T>
class MetricFamily;
template <typename T>
class MetricSeries;

/// A thread-safe custom counter handle. Updates after graph unload are ignored.
class CalculatorCounter {
    std::shared_ptr<MetricSeries<MetricCounter>> series;
    CalculatorCounter(MetricRegistry* registry, const std::string& name, const std::string& help,
        const std::string& graphName, const std::string& nodeName);
    void deactivate();
    friend class CalculatorMetrics;

public:
    CalculatorCounter(const CalculatorCounter&) = delete;
    CalculatorCounter& operator=(const CalculatorCounter&) = delete;
    ~CalculatorCounter();
    /// Add a finite, non-negative value. Invalid values are logged and ignored.
    void increment(double value = 1.0);
};

/// A thread-safe custom gauge handle. Updates after graph unload are ignored.
class CalculatorGauge {
    std::shared_ptr<MetricSeries<MetricGauge>> series;
    CalculatorGauge(MetricRegistry* registry, const std::string& name, const std::string& help,
        const std::string& graphName, const std::string& nodeName);
    void deactivate();
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

/// A thread-safe custom histogram handle. Updates after graph unload are ignored.
class CalculatorHistogram {
    std::shared_ptr<MetricSeries<MetricHistogram>> series;
    CalculatorHistogram(MetricRegistry* registry, const std::string& name, const std::string& help,
        const std::string& graphName, const std::string& nodeName, const std::vector<double>& bucketBoundaries);
    void deactivate();
    friend class CalculatorMetrics;

public:
    CalculatorHistogram(const CalculatorHistogram&) = delete;
    CalculatorHistogram& operator=(const CalculatorHistogram&) = delete;
    ~CalculatorHistogram();
    /// Record a finite observation. Invalid values are logged and ignored.
    void observe(double value);
};

/// One graph/node registration scope, shared by all instances of that calculator.
/// The registry must outlive the scope, or deactivate() must be called first.
class CalculatorMetrics {
    using MetricHandle = std::variant<
        std::shared_ptr<CalculatorCounter>,
        std::shared_ptr<CalculatorGauge>,
        std::shared_ptr<CalculatorHistogram>>;

    struct Registration {
        std::string help;
        std::vector<double> bucketBoundaries;
        MetricHandle metricHandle;

        void deactivate() const;
    };
    std::mutex scopeMutex;
    MetricRegistry* registry;
    const std::string graphName;
    const std::string nodeName;
    bool active = true;
    std::map<std::string, Registration> metrics;

    template <typename HandleType>
    std::shared_ptr<HandleType> findCompatibleRegistration(const std::string& name, const std::string& help,
        const std::vector<double>& bucketBoundaries = {}) const;

public:
    CalculatorMetrics(MetricRegistry* registry, const std::string& graphName, const std::string& nodeName);
    ~CalculatorMetrics();
    /// Get an existing handle or register one (usually in Open).
    /// Counters must be named custom_<name>_total;
    /// gauges and histograms must start with custom_. Names and help text must
    /// match across registrations. Disabled metrics return valid no-op handles.
    std::shared_ptr<CalculatorCounter> getOrRegisterCounter(const std::string& name, const std::string& help);
    std::shared_ptr<CalculatorGauge> getOrRegisterGauge(const std::string& name, const std::string& help);
    /// Histogram bucket boundaries must be finite and strictly increasing.
    std::shared_ptr<CalculatorHistogram> getOrRegisterHistogram(const std::string& name, const std::string& help,
        const std::vector<double>& bucketBoundaries);
    /// Remove this scope's series and make retained handles safe no-ops.
    void deactivate();
};

using CalculatorMetricsMap = std::map<std::string, std::shared_ptr<CalculatorMetrics>>;
inline constexpr const char* CALCULATOR_METRICS_TAG = "CALCULATOR_METRICS";
inline constexpr const char* CALCULATOR_METRICS_PACKET_NAME = "calculator_metrics";

}  // namespace ovms
