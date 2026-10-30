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

#include "../logging.hpp"
#include "metric.hpp"
#include "metric_family.hpp"
#include "metric_registry.hpp"

namespace ovms {

CalculatorCounter::CalculatorCounter(MetricRegistry* registry, const std::string& name, const std::string& help,
    const std::string& graphName, const std::string& nodeName) {
    if (registry) {
        family = registry->registerCustomCounterFamily(name, help);
        metric = family->addMetric({{"name", graphName}, {"node", nodeName}});
    }
}

CalculatorCounter::~CalculatorCounter() {
    deactivate();
}

void CalculatorCounter::deactivate() {
    std::lock_guard<std::mutex> lock(mutex);
    if (metric) {
        family->remove(metric);
        metric.reset();
    }
}

void CalculatorCounter::increment(double value) {
    if (!std::isfinite(value) || value < 0.0) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Custom counter increment must be finite and non-negative, got {}", value);
        throw std::invalid_argument("Custom counter increment must be finite and non-negative");
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (metric) {
        metric->increment(value);
    }
}

CalculatorMetrics::CalculatorMetrics(MetricRegistry* registry, const std::string& graphName, const std::string& nodeName) :
    registry(registry),
    graphName(graphName),
    nodeName(nodeName) {}

CalculatorMetrics::~CalculatorMetrics() {
    deactivate();
}

std::shared_ptr<CalculatorCounter> CalculatorMetrics::registerCounter(const std::string& name, const std::string& help) {
    std::lock_guard<std::mutex> lock(mutex);
    try {
        if (!active) {
            throw std::logic_error("Cannot register a custom counter after graph unload");
        }
        bool validName = name.size() > 13 && name.compare(0, 7, "custom_") == 0 &&
                         name.compare(name.size() - 6, 6, "_total") == 0;
        for (char c : name) {
            validName = validName && ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                         (c >= '0' && c <= '9') || c == '_');
        }
        if (!validName || help.empty()) {
            throw std::invalid_argument("Custom counter requires a custom_<name>_total name and non-empty help");
        }
        const auto it = counters.find(name);
        if (it != counters.end()) {
            if (it->second.help != help) {
                throw std::invalid_argument("Custom counter registered with conflicting help");
            }
            return it->second.counter;
        }
        auto counter = std::shared_ptr<CalculatorCounter>(new CalculatorCounter(registry, name, help, graphName, nodeName));
        counters.emplace(name, Registration{help, counter});
        return counter;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Custom counter registration failed for graph {} node {} metric {}: {}",
            graphName, nodeName, name, e.what());
        throw;
    }
}

void CalculatorMetrics::deactivate() {
    std::lock_guard<std::mutex> lock(mutex);
    active = false;
    for (const auto& entry : counters) {
        entry.second.counter->deactivate();
    }
    registry = nullptr;
}

}  // namespace ovms
