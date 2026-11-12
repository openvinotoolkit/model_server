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
#include "calculator_metrics.hpp"

#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "../logging.hpp"
#include "metric.hpp"
#include "metric_family.hpp"
#include "metric_registry.hpp"
#include "metric_series.hpp"

namespace ovms {

namespace {

constexpr std::string_view CUSTOM_METRIC_PREFIX = "custom_";
constexpr std::string_view COUNTER_SUFFIX = "_total";

bool hasValidCustomMetricName(const std::string& name) {
    if (name.size() <= CUSTOM_METRIC_PREFIX.size() ||
        name.compare(0, CUSTOM_METRIC_PREFIX.size(), CUSTOM_METRIC_PREFIX) != 0) {
        return false;
    }
    for (char c : name) {
        const bool isAllowed = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9') || c == '_';
        if (!isAllowed) {
            return false;
        }
    }
    return true;
}

bool hasValidCustomCounterName(const std::string& name) {
    return hasValidCustomMetricName(name) &&
           name.size() > CUSTOM_METRIC_PREFIX.size() + COUNTER_SUFFIX.size() &&
           name.compare(name.size() - COUNTER_SUFFIX.size(), COUNTER_SUFFIX.size(), COUNTER_SUFFIX) == 0;
}

bool hasValidBucketBoundaries(const std::vector<double>& bucketBoundaries) {
    if (bucketBoundaries.empty()) {
        return false;
    }
    for (size_t i = 0; i < bucketBoundaries.size(); ++i) {
        if (!std::isfinite(bucketBoundaries[i]) ||
            (i > 0 && bucketBoundaries[i] <= bucketBoundaries[i - 1])) {
            return false;
        }
    }
    return true;
}

std::string conflictingRegistrationMessage(const std::string& name) {
    return "Custom metric '" + name +
           "' was registered with a different type, help text, or histogram buckets; "
           "all registrations for the same metric name must match";
}

}  // namespace

CalculatorCounter::CalculatorCounter(MetricRegistry* registry, const std::string& name, const std::string& help,
    const std::string& graphName, const std::string& nodeName) :
    series(registry ? std::make_shared<MetricSeries<MetricCounter>>(
                          registry->registerCustomFamily<MetricCounter>(name, help),
                          MetricLabels{{"name", graphName}, {"node", nodeName}})
                    : nullptr) {}

CalculatorCounter::~CalculatorCounter() {
    deactivate();
}

void CalculatorCounter::deactivate() {
    if (series) {
        series->deactivate();
    }
}

void CalculatorCounter::increment(double value) {
    if (!std::isfinite(value) || value < 0.0) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Ignoring custom counter increment; value must be finite and non-negative, got {}", value);
        return;
    }
    if (series) {
        series->update([value](MetricCounter& metric) { metric.increment(value); });
    }
}

CalculatorGauge::CalculatorGauge(MetricRegistry* registry, const std::string& name, const std::string& help,
    const std::string& graphName, const std::string& nodeName) :
    series(registry ? std::make_shared<MetricSeries<MetricGauge>>(
                          registry->registerCustomFamily<MetricGauge>(name, help),
                          MetricLabels{{"name", graphName}, {"node", nodeName}})
                    : nullptr) {}

CalculatorGauge::~CalculatorGauge() {
    deactivate();
}

void CalculatorGauge::deactivate() {
    if (series) {
        series->deactivate();
    }
}

void CalculatorGauge::set(double value) {
    if (!std::isfinite(value)) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Ignoring custom gauge set; value must be finite, got {}", value);
        return;
    }
    if (series) {
        series->update([value](MetricGauge& metric) { metric.set(value); });
    }
}

void CalculatorGauge::add(double value) {
    if (!std::isfinite(value)) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Ignoring custom gauge add; value must be finite, got {}", value);
        return;
    }
    if (series) {
        series->update([value](MetricGauge& metric) { metric.increment(value); });
    }
}

CalculatorHistogram::CalculatorHistogram(MetricRegistry* registry, const std::string& name, const std::string& help,
    const std::string& graphName, const std::string& nodeName, const std::vector<double>& bucketBoundaries) :
    series(registry ? std::make_shared<MetricSeries<MetricHistogram>>(
                          registry->registerCustomFamily<MetricHistogram>(name, help, bucketBoundaries),
                          MetricLabels{{"name", graphName}, {"node", nodeName}}, bucketBoundaries)
                    : nullptr) {}

CalculatorHistogram::~CalculatorHistogram() {
    deactivate();
}

void CalculatorHistogram::deactivate() {
    if (series) {
        series->deactivate();
    }
}

void CalculatorHistogram::observe(double value) {
    if (!std::isfinite(value)) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Ignoring custom histogram observation; value must be finite, got {}", value);
        return;
    }
    if (series) {
        series->update([value](MetricHistogram& metric) { metric.observe(value); });
    }
}

CalculatorMetrics::CalculatorMetrics(MetricRegistry* registry, const std::string& graphName, const std::string& nodeName) :
    registry(registry),
    graphName(graphName),
    nodeName(nodeName) {}

CalculatorMetrics::~CalculatorMetrics() {
    deactivate();
}

template <typename HandleType>
std::shared_ptr<HandleType> CalculatorMetrics::findCompatibleRegistration(
    const std::string& name, const std::string& help,
    const std::vector<double>& bucketBoundaries) const {
    const auto existing = metrics.find(name);
    if (existing == metrics.end()) {
        return nullptr;
    }
    const auto* metricHandle = std::get_if<std::shared_ptr<HandleType>>(&existing->second.metricHandle);
    if (!metricHandle || existing->second.help != help ||
        existing->second.bucketBoundaries != bucketBoundaries) {
        throw std::invalid_argument(conflictingRegistrationMessage(name));
    }
    return *metricHandle;
}

std::shared_ptr<CalculatorCounter> CalculatorMetrics::getOrRegisterCounter(const std::string& name, const std::string& help) {
    std::lock_guard<std::mutex> lock(scopeMutex);
    try {
        if (!active) {
            throw std::logic_error("Cannot register a custom counter after graph unload");
        }
        if (!hasValidCustomCounterName(name) || help.empty()) {
            throw std::invalid_argument("Custom counter requires a custom_<name>_total name and non-empty help");
        }
        if (auto existing = findCompatibleRegistration<CalculatorCounter>(name, help)) {
            return existing;
        }
        auto counter = std::shared_ptr<CalculatorCounter>(new CalculatorCounter(registry, name, help, graphName, nodeName));
        metrics.emplace(name, Registration{help, {}, counter});
        return counter;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Custom counter registration failed for graph {} node {} metric {}: {}",
            graphName, nodeName, name, e.what());
        throw;
    }
}

std::shared_ptr<CalculatorGauge> CalculatorMetrics::getOrRegisterGauge(const std::string& name, const std::string& help) {
    std::lock_guard<std::mutex> lock(scopeMutex);
    try {
        if (!active) {
            throw std::logic_error("Cannot register a custom gauge after graph unload");
        }
        if (!hasValidCustomMetricName(name) || help.empty()) {
            throw std::invalid_argument("Custom gauge requires a custom_ name and non-empty help");
        }
        if (auto existing = findCompatibleRegistration<CalculatorGauge>(name, help)) {
            return existing;
        }
        auto gauge = std::shared_ptr<CalculatorGauge>(new CalculatorGauge(registry, name, help, graphName, nodeName));
        metrics.emplace(name, Registration{help, {}, gauge});
        return gauge;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Custom gauge registration failed for graph {} node {} metric {}: {}",
            graphName, nodeName, name, e.what());
        throw;
    }
}

std::shared_ptr<CalculatorHistogram> CalculatorMetrics::getOrRegisterHistogram(const std::string& name, const std::string& help,
    const std::vector<double>& bucketBoundaries) {
    std::lock_guard<std::mutex> lock(scopeMutex);
    try {
        if (!active) {
            throw std::logic_error("Cannot register a custom histogram after graph unload");
        }
        if (!hasValidCustomMetricName(name) || help.empty() || !hasValidBucketBoundaries(bucketBoundaries)) {
            throw std::invalid_argument("Custom histogram requires a custom_ name, non-empty help, and finite strictly increasing buckets");
        }
        if (auto existing = findCompatibleRegistration<CalculatorHistogram>(name, help, bucketBoundaries)) {
            return existing;
        }
        auto histogram = std::shared_ptr<CalculatorHistogram>(
            new CalculatorHistogram(registry, name, help, graphName, nodeName, bucketBoundaries));
        metrics.emplace(name, Registration{help, bucketBoundaries, histogram});
        return histogram;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Custom histogram registration failed for graph {} node {} metric {}: {}",
            graphName, nodeName, name, e.what());
        throw;
    }
}

void CalculatorMetrics::Registration::deactivate() const {
    std::visit([](const auto& handle) { handle->deactivate(); }, metricHandle);
}

void CalculatorMetrics::deactivate() {
    std::lock_guard<std::mutex> lock(scopeMutex);
    if (!active) {
        return;
    }
    active = false;
    for (const auto& entry : metrics) {
        entry.second.deactivate();
    }
    registry = nullptr;
}

}  // namespace ovms
