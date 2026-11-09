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

namespace ovms {

class MetricConfig;
class MetricRegistry;
class MetricGauge;
class MetricHistogram;
class MetricCounter;
template <typename T>
class MetricFamily;
template <typename T>
class MetricSeries;

// Owns the metric series for one LLM pipeline. The registry must outlive it.
class LLMMetricReporter {
    std::mutex mutex;
    std::atomic<bool> cacheEnabled{false};
    std::atomic<bool> cacheSizeEnabled{false};
    std::atomic<bool> prefillEnabled{false};
    std::atomic<bool> scheduledEnabled{false};
    std::atomic<bool> completedRequestMetricsEnabled{false};
    std::atomic<bool> completionMetricsEnabled{false};
    std::shared_ptr<MetricSeries<MetricGauge>> cacheUtilization;
    std::shared_ptr<MetricSeries<MetricGauge>> cacheSize;
    std::shared_ptr<MetricSeries<MetricHistogram>> prefillSpeed;
    std::shared_ptr<MetricSeries<MetricGauge>> scheduledRequests;
    std::shared_ptr<MetricSeries<MetricCounter>> generatedTokens;
    std::shared_ptr<MetricSeries<MetricHistogram>> ttft;

    void removeMetrics();

public:
    LLMMetricReporter(const MetricConfig* config, MetricRegistry* registry,
        const std::string& graphName, const std::string& nodeName);
    ~LLMMetricReporter();
    LLMMetricReporter(const LLMMetricReporter&) = delete;
    LLMMetricReporter& operator=(const LLMMetricReporter&) = delete;

    bool isCacheUtilizationEnabled() const;
    bool isCacheSizeEnabled() const;
    bool isPrefillSpeedEnabled() const;
    bool isScheduledRequestsEnabled() const;
    bool hasCompletedRequestMetricsEnabled() const;
    bool hasAnyCompletionMetricsEnabled() const;
    /// Stop publishing and remove series before replacing or unloading a pipeline.
    void deactivate();
    /// Publish GenAI's cache occupancy in percent, independently of scraping.
    void setCacheUtilization(double percent);
    /// Publish GenAI's KV cache size in bytes.
    void setCacheSizeBytes(size_t bytes);
    /// Record one completed request's TTFT-derived speed in tokens per second.
    void observePrefillSpeed(double tokensPerSecond);
    /// Publish GenAI's last-step scheduled request count, or zero when the executor is idle.
    void setScheduledRequests(size_t requests);
    /// Record successful completion only; partial output from failed/cancelled requests is not counted.
    /// TTFT is GenAI's request latency in milliseconds, excluding VLM embedding preparation.
    void observeCompletedRequest(size_t outputTokens, double ttftMilliseconds);
};

}  // namespace ovms
