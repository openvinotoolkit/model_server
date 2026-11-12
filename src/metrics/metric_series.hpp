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

#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "metric.hpp"
#include "metric_family.hpp"

namespace ovms {

/// Owns one labeled registry series and synchronizes updates with removal.
template <typename MetricType>
class MetricSeries {
    std::mutex mutex;
    std::shared_ptr<MetricFamily<MetricType>> family;
    std::unique_ptr<MetricType> metric;

public:
    MetricSeries(std::shared_ptr<MetricFamily<MetricType>> family,
        const MetricLabels& labels, const BucketBoundaries& bucketBoundaries = {}) :
        family(std::move(family)) {
        if (!this->family) {
            throw std::invalid_argument("Cannot create metric series without a registered family");
        }
        metric = this->family->addMetric(labels, bucketBoundaries);
    }

    template <typename UpdateFn>
    void update(UpdateFn&& updateMetric) {
        std::lock_guard<std::mutex> lock(mutex);
        if (metric) {
            updateMetric(*metric);
        }
    }

    void deactivate() {
        std::lock_guard<std::mutex> lock(mutex);
        if (metric) {
            family->remove(metric);
            metric.reset();
        }
    }
};

}  // namespace ovms
