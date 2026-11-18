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

#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>

#include "../logging.hpp"
#include "metric.hpp"
#include "metric_config.hpp"
#include "metric_family.hpp"
#include "metric_registry.hpp"

namespace ovms {
namespace {

constexpr double STOPPED_CACHE_UTILIZATION_PERCENT = 0.0;
constexpr size_t STOPPED_CACHE_SIZE_BYTES = 0;
constexpr size_t STOPPED_SCHEDULED_REQUESTS = 0;

template <typename MetricType, size_t Count>
using MetricFamilies = std::array<std::shared_ptr<MetricFamily<MetricType>>, Count>;

// Families are created before any series is added, so a registration failure
// leaves no partially registered series behind.
template <typename MetricType, typename Registration, size_t Count>
MetricFamilies<MetricType, Count> createEnabledFamilies(const MetricConfig& config,
    MetricRegistry& registry, const Registration (&registrations)[Count]) {
    MetricFamilies<MetricType, Count> families;
    for (size_t i = 0; i < Count; ++i) {
        const auto& registration = registrations[i];
        if (!config.isFamilyEnabled(registration.name)) {
            continue;
        }
        families[i] = registry.createFamily<MetricType>(registration.name, registration.help);
        if (!families[i]) {
            throw std::logic_error("Cannot create LLM metric family '" + registration.name + "'");
        }
    }
    return families;
}

// Adding a series whose labels are already registered returns the existing series.
template <typename MetricType, typename Registration, size_t Count>
void addSeries(LLMMetricReporter& reporter, const MetricFamilies<MetricType, Count>& families,
    const MetricLabels& labels, const Registration (&registrations)[Count]) {
    for (size_t i = 0; i < Count; ++i) {
        if (families[i]) {
            reporter.*(registrations[i].destination) =
                families[i]->addMetric(labels, registrations[i].bucketBoundaries);
        }
    }
}

}  // namespace

LLMMetricReporter::LLMMetricReporter(const MetricConfig* config, MetricRegistry* registry,
    const std::string& graphName, const std::string& nodeName) {
    if (!registry || !config || !config->metricsEnabled) {
        return;
    }

    const auto gaugeFamilies = createEnabledFamilies<MetricGauge>(*config, *registry, GAUGE_REGISTRATIONS);
    const auto counterFamilies = createEnabledFamilies<MetricCounter>(*config, *registry, COUNTER_REGISTRATIONS);
    const auto histogramFamilies = createEnabledFamilies<MetricHistogram>(*config, *registry, HISTOGRAM_REGISTRATIONS);
    const MetricLabels labels{{"name", graphName}, {"node", nodeName}};
    addSeries(*this, gaugeFamilies, labels, GAUGE_REGISTRATIONS);
    addSeries(*this, counterFamilies, labels, COUNTER_REGISTRATIONS);
    addSeries(*this, histogramFamilies, labels, HISTOGRAM_REGISTRATIONS);

    pipelineMetricsOwnership = &registry->getPipelineMetricsOwnership(graphName, nodeName);
    std::lock_guard<std::mutex> lock(pipelineMetricsOwnership->mutex);
    reporterId = ++pipelineMetricsOwnership->newestReporterId;
}

LLMMetricReporter::~LLMMetricReporter() = default;

bool LLMMetricReporter::isScheduledRequestsEnabled() const {
    return scheduledRequests != nullptr;
}

bool LLMMetricReporter::hasCompletedRequestMetricsEnabled() const {
    return promptTokens || generatedTokens || ttft || tpot;
}

bool LLMMetricReporter::hasAnyCompletionMetricsEnabled() const {
    return hasCompletedRequestMetricsEnabled();
}

void LLMMetricReporter::resetCacheAndScheduledRequestsGauges() {
    setCacheUtilization(STOPPED_CACHE_UTILIZATION_PERCENT);
    setCacheSizeBytes(STOPPED_CACHE_SIZE_BYTES);
    setScheduledRequests(STOPPED_SCHEDULED_REQUESTS);
}

void LLMMetricReporter::setPipelineMetric(const std::unique_ptr<MetricGauge>& gauge, double value) {
    if (!gauge) {
        return;
    }
    // The check and the write share one lock, so a reporter superseded in between cannot write.
    std::lock_guard<std::mutex> lock(pipelineMetricsOwnership->mutex);
    if (reporterId == pipelineMetricsOwnership->newestReporterId) {
        gauge->set(value);
    }
}

void LLMMetricReporter::setScheduledRequests(size_t requests) {
    setPipelineMetric(scheduledRequests, static_cast<double>(requests));
}

void LLMMetricReporter::observeCompletedRequest(const RequestMetricsSnapshot& metrics) {
    if (promptTokens) {
        promptTokens->increment(static_cast<double>(metrics.inputTokens));
    }
    if (generatedTokens) {
        generatedTokens->increment(static_cast<double>(metrics.outputTokens));
    }
    if (ttft && (!std::isfinite(metrics.ttftMilliseconds) || metrics.ttftMilliseconds <= 0.0)) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Ignoring invalid GenAI TTFT in milliseconds: {}", metrics.ttftMilliseconds);
    } else if (ttft) {
        ttft->observe(metrics.ttftMilliseconds);
    }
    if (tpot && (!std::isfinite(metrics.tpotMilliseconds) || metrics.tpotMilliseconds <= 0.0)) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Ignoring invalid GenAI TPOT in milliseconds: {}", metrics.tpotMilliseconds);
    } else if (tpot) {
        tpot->observe(metrics.tpotMilliseconds);
    }
}

bool LLMMetricReporter::isCacheUtilizationEnabled() const {
    return cacheUtilization != nullptr;
}

bool LLMMetricReporter::isCacheSizeEnabled() const {
    return cacheSize != nullptr;
}

void LLMMetricReporter::setCacheUtilization(double percent) {
    if (!cacheUtilization) {
        return;
    }
    if (!std::isfinite(percent) || percent < 0.0 || percent > 100.0) {
        SPDLOG_LOGGER_WARN(llm_executor_logger, "Ignoring invalid GenAI KV cache utilization: {}", percent);
        return;
    }
    setPipelineMetric(cacheUtilization, percent);
}

void LLMMetricReporter::setCacheSizeBytes(size_t bytes) {
    setPipelineMetric(cacheSize, static_cast<double>(bytes));
}

}  // namespace ovms
