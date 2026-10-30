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
#include <memory>
#include <mutex>
#include <string>

namespace ovms {

class MetricConfig;
class MetricRegistry;
class MetricGauge;
class MetricHistogram;
template <typename T>
class MetricFamily;

// Owns the metric series for one LLM pipeline. The registry must outlive it.
class LLMMetricReporter {
    std::mutex mutex;
    std::atomic<bool> cacheEnabled{false};
    std::atomic<bool> prefillEnabled{false};
    std::shared_ptr<MetricFamily<MetricGauge>> cacheFamily;
    std::shared_ptr<MetricFamily<MetricHistogram>> prefillFamily;
    std::unique_ptr<MetricGauge> cacheUtilization;
    std::unique_ptr<MetricHistogram> prefillSpeed;

    void removeMetrics();

public:
    LLMMetricReporter(const MetricConfig* config, MetricRegistry* registry,
        const std::string& graphName, const std::string& nodeName);
    ~LLMMetricReporter();
    LLMMetricReporter(const LLMMetricReporter&) = delete;
    LLMMetricReporter& operator=(const LLMMetricReporter&) = delete;

    bool isCacheUtilizationEnabled() const;
    bool isPrefillSpeedEnabled() const;
    /// Stop publishing and remove series before replacing or unloading a pipeline.
    void deactivate();
    /// Publish GenAI's cache occupancy in percent, independently of scraping.
    void setCacheUtilization(double percent);
    /// Record one completed request's TTFT-derived speed in tokens per second.
    void observePrefillSpeed(double tokensPerSecond);
};

}  // namespace ovms
