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

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "metric_config.hpp"

namespace ovms {

class MetricRegistry;
class MetricGauge;
class MetricHistogram;
class MetricCounter;
template <typename T>
class MetricFamily;
template <typename T>
class MetricSeries;

struct RequestMetricsSnapshot {
    size_t inputTokens;
    size_t outputTokens;
    double ttftMilliseconds;
    double tpotMilliseconds;
};

// Owns the metric series for one LLM pipeline. The registry must outlive it.
class LLMMetricReporter {
    std::mutex mutex;
    std::atomic<bool> cacheEnabled{false};
    std::atomic<bool> cacheSizeEnabled{false};
    std::atomic<bool> scheduledEnabled{false};
    std::atomic<bool> completedRequestMetricsEnabled{false};
    std::atomic<bool> completionMetricsEnabled{false};
    std::shared_ptr<MetricSeries<MetricGauge>> cacheUtilization;
    std::shared_ptr<MetricSeries<MetricGauge>> cacheSize;
    std::shared_ptr<MetricSeries<MetricGauge>> scheduledRequests;
    std::shared_ptr<MetricSeries<MetricCounter>> promptTokens;
    std::shared_ptr<MetricSeries<MetricCounter>> generatedTokens;
    std::shared_ptr<MetricSeries<MetricHistogram>> ttft;
    std::shared_ptr<MetricSeries<MetricHistogram>> tpot;

    template <typename MetricType>
    struct MetricRegistration {
        const std::string& name;
        const char* help;
        std::shared_ptr<MetricSeries<MetricType>> LLMMetricReporter::* destination;
        std::vector<double> bucketBoundaries;
    };

    inline static const std::vector<double> BUCKET_BOUNDARIES_1_TO_640K{
        1, 5, 10, 20, 40, 60, 80, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000,
        20000, 40000, 80000, 160000, 640000};
    inline static const std::vector<double> BUCKET_BOUNDARIES_10_TO_80K{
        10, 20, 40, 60, 80, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000, 40000, 80000};

    inline static const MetricRegistration<MetricGauge> GAUGE_REGISTRATIONS[]{
        {METRIC_NAME_KV_CACHE_UTILIZATION,
            "KV cache utilization in percent of the pipeline cache capacity.",
            &LLMMetricReporter::cacheUtilization, {}},
        {METRIC_NAME_KV_CACHE_SIZE,
            "KV cache size in bytes.",
            &LLMMetricReporter::cacheSize, {}},
        {METRIC_NAME_LLM_SCHEDULED_REQUESTS,
            "Number of requests scheduled in GenAI's last generation step, or zero when idle.",
            &LLMMetricReporter::scheduledRequests, {}}};
    inline static const MetricRegistration<MetricCounter> COUNTER_REGISTRATIONS[]{
        {METRIC_NAME_LLM_PROMPT_TOKENS,
            "Input prompt tokens from successfully completed requests, as reported by GenAI.",
            &LLMMetricReporter::promptTokens, {}},
        {METRIC_NAME_LLM_GENERATED_TOKENS,
            "Generated output tokens from successfully completed requests; excludes partial output from failed or cancelled requests.",
            &LLMMetricReporter::generatedTokens, {}}};
    inline static const MetricRegistration<MetricHistogram> HISTOGRAM_REGISTRATIONS[]{
        {METRIC_NAME_LLM_TTFT,
            "GenAI time to first token in milliseconds for successfully completed requests, excluding VLM embedding preparation.",
            &LLMMetricReporter::ttft, BUCKET_BOUNDARIES_1_TO_640K},
        {METRIC_NAME_LLM_TPOT,
            "GenAI time per output token in milliseconds for successfully completed requests.",
            &LLMMetricReporter::tpot, BUCKET_BOUNDARIES_10_TO_80K}};

    void removeMetrics();

public:
    LLMMetricReporter(const MetricConfig* config, MetricRegistry* registry,
        const std::string& graphName, const std::string& nodeName);
    ~LLMMetricReporter();
    LLMMetricReporter(const LLMMetricReporter&) = delete;
    LLMMetricReporter& operator=(const LLMMetricReporter&) = delete;

    bool isCacheUtilizationEnabled() const;
    bool isCacheSizeEnabled() const;
    bool isScheduledRequestsEnabled() const;
    bool hasCompletedRequestMetricsEnabled() const;
    bool hasAnyCompletionMetricsEnabled() const;
    /// Stop publishing and remove series before replacing or unloading a pipeline.
    void deactivate();
    /// Publish GenAI's cache occupancy in percent, independently of scraping.
    void setCacheUtilization(double percent);
    /// Publish GenAI's KV cache size in bytes.
    void setCacheSizeBytes(size_t bytes);
    /// Publish GenAI's last-step scheduled request count, or zero when the executor is idle.
    void setScheduledRequests(size_t requests);
    /// Record successful completion only; partial output from failed/cancelled requests is not counted.
    /// TTFT and TPOT are GenAI request timing values in milliseconds.
    void observeCompletedRequest(const RequestMetricsSnapshot& metrics);
};

}  // namespace ovms
