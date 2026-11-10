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

const BucketBoundaries TTFT_BUCKETS{
    1, 5, 10, 20, 40, 60, 80, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000,
    20000, 40000, 80000, 160000, 640000};
const BucketBoundaries TPOT_BUCKETS{
    10, 20, 40, 60, 80, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000, 40000, 80000};

}  // namespace

LLMMetricReporter::LLMMetricReporter(const MetricConfig* config, MetricRegistry* registry,
    const std::string& graphName, const std::string& nodeName) {
    if (!registry || !config || !config->metricsEnabled) {
        return;
    }

    const MetricLabels labels{{"name", graphName}, {"node", nodeName}};
    try {
        if (config->isFamilyEnabled(METRIC_NAME_KV_CACHE_UTILIZATION)) {
            auto family = registry->createFamily<MetricGauge>(METRIC_NAME_KV_CACHE_UTILIZATION,
                "GenAI KV cache utilization in percent of the pipeline cache capacity.");
            if (!family) {
                throw std::logic_error("Cannot create LLM KV cache metric family");
            }
            cacheUtilization = std::make_shared<MetricSeries<MetricGauge>>(family, labels);
            cacheEnabled.store(true, std::memory_order_relaxed);
        }
        if (config->isFamilyEnabled(METRIC_NAME_KV_CACHE_SIZE)) {
            auto family = registry->createFamily<MetricGauge>(METRIC_NAME_KV_CACHE_SIZE,
                "GenAI-reported KV cache size in bytes.");
            if (!family) {
                throw std::logic_error("Cannot create KV cache size metric family");
            }
            cacheSize = std::make_shared<MetricSeries<MetricGauge>>(family, labels);
            cacheSizeEnabled.store(true, std::memory_order_relaxed);
        }
        if (config->isFamilyEnabled(METRIC_NAME_LLM_PROMPT_TOKENS)) {
            auto family = registry->createFamily<MetricCounter>(METRIC_NAME_LLM_PROMPT_TOKENS,
                "Input prompt tokens from successfully completed requests, as reported by GenAI.");
            if (!family) {
                throw std::logic_error("Cannot create LLM prompt tokens metric family");
            }
            promptTokens = std::make_shared<MetricSeries<MetricCounter>>(family, labels);
        }
        if (config->isFamilyEnabled(METRIC_NAME_LLM_SCHEDULED_REQUESTS)) {
            auto family = registry->createFamily<MetricGauge>(METRIC_NAME_LLM_SCHEDULED_REQUESTS,
                "Number of requests scheduled in GenAI's last generation step, or zero when idle.");
            if (!family) {
                throw std::logic_error("Cannot create LLM scheduled requests metric family");
            }
            scheduledRequests = std::make_shared<MetricSeries<MetricGauge>>(family, labels);
            scheduledEnabled.store(true, std::memory_order_relaxed);
        }
        if (config->isFamilyEnabled(METRIC_NAME_LLM_GENERATED_TOKENS)) {
            auto family = registry->createFamily<MetricCounter>(METRIC_NAME_LLM_GENERATED_TOKENS,
                "Generated output tokens from successfully completed requests; excludes partial output from failed or cancelled requests.");
            if (!family) {
                throw std::logic_error("Cannot create LLM generated tokens metric family");
            }
            generatedTokens = std::make_shared<MetricSeries<MetricCounter>>(family, labels);
        }
        if (config->isFamilyEnabled(METRIC_NAME_LLM_TTFT)) {
            auto family = registry->createFamily<MetricHistogram>(METRIC_NAME_LLM_TTFT,
                "GenAI time to first token in milliseconds for successfully completed requests, excluding VLM embedding preparation.");
            if (!family) {
                throw std::logic_error("Cannot create LLM TTFT metric family");
            }
            ttft = std::make_shared<MetricSeries<MetricHistogram>>(family, labels, TTFT_BUCKETS);
        }
        if (config->isFamilyEnabled(METRIC_NAME_LLM_TPOT)) {
            auto family = registry->createFamily<MetricHistogram>(METRIC_NAME_LLM_TPOT,
                "GenAI time per output token in milliseconds for successfully completed requests.");
            if (!family) {
                throw std::logic_error("Cannot create LLM TPOT metric family");
            }
            tpot = std::make_shared<MetricSeries<MetricHistogram>>(family, labels, TPOT_BUCKETS);
        }
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
    if (cacheUtilization) {
        cacheUtilization->deactivate();
        cacheUtilization.reset();
    }
    if (cacheSize) {
        cacheSize->deactivate();
        cacheSize.reset();
    }
    if (scheduledRequests) {
        scheduledRequests->deactivate();
        scheduledRequests.reset();
    }
    if (generatedTokens) {
        generatedTokens->deactivate();
        generatedTokens.reset();
    }
    if (promptTokens) {
        promptTokens->deactivate();
        promptTokens.reset();
    }
    if (ttft) {
        ttft->deactivate();
        ttft.reset();
    }
    if (tpot) {
        tpot->deactivate();
        tpot.reset();
    }
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

void LLMMetricReporter::observeCompletedRequest(size_t inputTokens, size_t outputTokens, double ttftMilliseconds, double tpotMilliseconds) {
    std::lock_guard<std::mutex> lock(mutex);
    if (promptTokens) {
        promptTokens->update([inputTokens](MetricCounter& metric) { metric.increment(inputTokens); });
    }
    if (generatedTokens) {
        generatedTokens->update([outputTokens](MetricCounter& metric) { metric.increment(outputTokens); });
    }
    if (ttft && (!std::isfinite(ttftMilliseconds) || ttftMilliseconds <= 0.0)) {
#if (MEDIAPIPE_DISABLE == 0)
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Ignoring invalid GenAI TTFT in milliseconds: {}", ttftMilliseconds);
#endif
    } else if (ttft) {
        ttft->update([ttftMilliseconds](MetricHistogram& metric) { metric.observe(ttftMilliseconds); });
    }
    if (tpot && (!std::isfinite(tpotMilliseconds) || tpotMilliseconds <= 0.0)) {
#if (MEDIAPIPE_DISABLE == 0)
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Ignoring invalid GenAI TPOT in milliseconds: {}", tpotMilliseconds);
#endif
    } else if (tpot) {
        tpot->update([tpotMilliseconds](MetricHistogram& metric) { metric.observe(tpotMilliseconds); });
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
#if (MEDIAPIPE_DISABLE == 0)
        SPDLOG_LOGGER_WARN(llm_executor_logger, "Ignoring invalid GenAI KV cache utilization: {}", percent);
#endif
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
