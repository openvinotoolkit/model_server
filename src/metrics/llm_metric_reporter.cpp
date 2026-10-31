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

namespace ovms {

LLMMetricReporter::LLMMetricReporter(const MetricConfig* config, MetricRegistry* registry,
    const std::string& graphName, const std::string& nodeName) {
    if (!registry || !config || !config->metricsEnabled) {
        return;
    }

    const MetricLabels labels{{"name", graphName}, {"node", nodeName}};
    try {
        if (config->isFamilyEnabled(METRIC_NAME_LLM_KV_CACHE_UTILIZATION)) {
            cacheFamily = registry->createFamily<MetricGauge>(METRIC_NAME_LLM_KV_CACHE_UTILIZATION,
                "GenAI KV cache utilization in percent of the pipeline cache capacity.");
            if (!cacheFamily) {
                throw std::logic_error("Cannot create LLM KV cache metric family");
            }
            cacheUtilization = cacheFamily->addMetric(labels);
            cacheEnabled.store(true, std::memory_order_relaxed);
        }
        if (config->isFamilyEnabled(METRIC_NAME_LLM_PREFILL_SPEED)) {
            prefillFamily = registry->createFamily<MetricHistogram>(METRIC_NAME_LLM_PREFILL_SPEED,
                "Completed request input tokens divided by GenAI time to first token in seconds.");
            if (!prefillFamily) {
                throw std::logic_error("Cannot create LLM prefill speed metric family");
            }
            prefillSpeed = prefillFamily->addMetric(labels,
                {1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000});
            prefillEnabled.store(true, std::memory_order_relaxed);
        }
    } catch (const std::exception& e) {
        removeMetrics();
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to register LLM metrics for graph {} node {}: {}", graphName, nodeName, e.what());
        throw;
    }
}

LLMMetricReporter::~LLMMetricReporter() {
    deactivate();
}

void LLMMetricReporter::deactivate() {
    std::lock_guard<std::mutex> lock(mutex);
    cacheEnabled.store(false, std::memory_order_relaxed);
    prefillEnabled.store(false, std::memory_order_relaxed);
    removeMetrics();
}

void LLMMetricReporter::removeMetrics() {
    if (cacheUtilization) {
        cacheFamily->remove(cacheUtilization);
        cacheUtilization.reset();
    }
    if (prefillSpeed) {
        prefillFamily->remove(prefillSpeed);
        prefillSpeed.reset();
    }
}

bool LLMMetricReporter::isCacheUtilizationEnabled() const {
    return cacheEnabled.load(std::memory_order_relaxed);
}

bool LLMMetricReporter::isPrefillSpeedEnabled() const {
    return prefillEnabled.load(std::memory_order_relaxed);
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
    cacheUtilization->set(percent);
}

void LLMMetricReporter::observePrefillSpeed(double tokensPerSecond) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!prefillSpeed) {
        return;
    }
    if (!std::isfinite(tokensPerSecond) || tokensPerSecond <= 0.0) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Ignoring invalid TTFT-derived prefill speed: {}", tokensPerSecond);
        return;
    }
    prefillSpeed->observe(tokensPerSecond);
}

}  // namespace ovms
