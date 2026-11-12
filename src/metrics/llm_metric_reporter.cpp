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
#include "llm_metric_reporter.hpp"

#include <cmath>
#include <stdexcept>

#include "../logging.hpp"
#include "metric.hpp"
#include "metric_config.hpp"
#include "metric_family.hpp"
#include "metric_registry.hpp"
#include "metric_series.hpp"

namespace ovms {
namespace {

template <typename MetricType, typename Registration, size_t Count>
void registerEnabledMetrics(LLMMetricReporter& reporter, const MetricConfig& config,
    MetricRegistry& registry, const MetricLabels& labels,
    const Registration (&registrations)[Count]) {
    for (const auto& registration : registrations) {
        if (!config.isFamilyEnabled(registration.name)) {
            continue;
        }
        auto family = registry.createFamily<MetricType>(registration.name, registration.help);
        if (!family) {
            throw std::logic_error("Cannot create LLM metric family '" + registration.name + "'");
        }
        reporter.*(registration.destination) =
            std::make_shared<MetricSeries<MetricType>>(family, labels, registration.bucketBoundaries);
    }
}

template <typename Registration, size_t Count>
void removeRegisteredMetrics(LLMMetricReporter& reporter,
    const Registration (&registrations)[Count]) {
    for (const auto& registration : registrations) {
        auto& series = reporter.*(registration.destination);
        if (series) {
            series->deactivate();
            series.reset();
        }
    }
}

}  // namespace

LLMMetricReporter::LLMMetricReporter(const MetricConfig* config, MetricRegistry* registry,
    const std::string& graphName, const std::string& nodeName) {
    if (!registry || !config || !config->metricsEnabled) {
        return;
    }

    const MetricLabels labels{{"name", graphName}, {"node", nodeName}};
    try {
        registerEnabledMetrics<MetricGauge>(*this, *config, *registry, labels, GAUGE_REGISTRATIONS);
        registerEnabledMetrics<MetricCounter>(*this, *config, *registry, labels, COUNTER_REGISTRATIONS);
        registerEnabledMetrics<MetricHistogram>(*this, *config, *registry, labels, HISTOGRAM_REGISTRATIONS);
        cacheEnabled.store(cacheUtilization != nullptr, std::memory_order_relaxed);
        cacheSizeEnabled.store(cacheSize != nullptr, std::memory_order_relaxed);
        scheduledEnabled.store(scheduledRequests != nullptr, std::memory_order_relaxed);
        completedRequestMetricsEnabled.store(promptTokens || generatedTokens || ttft || tpot, std::memory_order_relaxed);
        completionMetricsEnabled.store(promptTokens || generatedTokens || ttft || tpot, std::memory_order_relaxed);
    } catch (const std::exception&) {
        removeMetrics();
        throw;
    }
}

LLMMetricReporter::~LLMMetricReporter() {
    deactivate();
}

void LLMMetricReporter::deactivate() {
    std::lock_guard<std::mutex> lock(mutex);
    cacheEnabled.store(false, std::memory_order_relaxed);
    cacheSizeEnabled.store(false, std::memory_order_relaxed);
    scheduledEnabled.store(false, std::memory_order_relaxed);
    completedRequestMetricsEnabled.store(false, std::memory_order_relaxed);
    completionMetricsEnabled.store(false, std::memory_order_relaxed);
    removeMetrics();
}

void LLMMetricReporter::removeMetrics() {
    removeRegisteredMetrics(*this, GAUGE_REGISTRATIONS);
    removeRegisteredMetrics(*this, COUNTER_REGISTRATIONS);
    removeRegisteredMetrics(*this, HISTOGRAM_REGISTRATIONS);
}

bool LLMMetricReporter::isScheduledRequestsEnabled() const {
    return scheduledEnabled.load(std::memory_order_relaxed);
}

bool LLMMetricReporter::hasCompletedRequestMetricsEnabled() const {
    return completedRequestMetricsEnabled.load(std::memory_order_relaxed);
}

bool LLMMetricReporter::hasAnyCompletionMetricsEnabled() const {
    return completionMetricsEnabled.load(std::memory_order_relaxed);
}

void LLMMetricReporter::setScheduledRequests(size_t requests) {
    std::lock_guard<std::mutex> lock(mutex);
    if (scheduledRequests) {
        scheduledRequests->update([requests](MetricGauge& metric) { metric.set(requests); });
    }
}

void LLMMetricReporter::observeCompletedRequest(const RequestMetricsSnapshot& metrics) {
    std::lock_guard<std::mutex> lock(mutex);
    if (promptTokens) {
        promptTokens->update([&metrics](MetricCounter& metric) { metric.increment(metrics.inputTokens); });
    }
    if (generatedTokens) {
        generatedTokens->update([&metrics](MetricCounter& metric) { metric.increment(metrics.outputTokens); });
    }
    if (ttft && (!std::isfinite(metrics.ttftMilliseconds) || metrics.ttftMilliseconds <= 0.0)) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Ignoring invalid GenAI TTFT in milliseconds: {}", metrics.ttftMilliseconds);
    } else if (ttft) {
        ttft->update([&metrics](MetricHistogram& metric) { metric.observe(metrics.ttftMilliseconds); });
    }
    if (tpot && (!std::isfinite(metrics.tpotMilliseconds) || metrics.tpotMilliseconds <= 0.0)) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Ignoring invalid GenAI TPOT in milliseconds: {}", metrics.tpotMilliseconds);
    } else if (tpot) {
        tpot->update([&metrics](MetricHistogram& metric) { metric.observe(metrics.tpotMilliseconds); });
    }
}

bool LLMMetricReporter::isCacheUtilizationEnabled() const {
    return cacheEnabled.load(std::memory_order_relaxed);
}

bool LLMMetricReporter::isCacheSizeEnabled() const {
    return cacheSizeEnabled.load(std::memory_order_relaxed);
}

void LLMMetricReporter::setCacheUtilization(double percent) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!cacheUtilization) {
        return;
    }
    if (!std::isfinite(percent) || percent < 0.0 || percent > 100.0) {
        SPDLOG_LOGGER_WARN(llm_executor_logger, "Ignoring invalid GenAI KV cache utilization: {}", percent);
        return;
    }
    cacheUtilization->update([percent](MetricGauge& metric) { metric.set(percent); });
}

void LLMMetricReporter::setCacheSizeBytes(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    if (cacheSize) {
        cacheSize->update([bytes](MetricGauge& metric) { metric.set(static_cast<double>(bytes)); });
    }
}

}  // namespace ovms
