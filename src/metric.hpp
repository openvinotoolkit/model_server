//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include <map>
#include <string>
#include <vector>

namespace prometheus {
class Counter;
class Gauge;
class Histogram;
}  // namespace prometheus

namespace ovms {

template <typename T>
class MetricFamily;

class Metric {
public:
    using Labels = std::map<std::string, std::string>;
    using BucketBoundaries = std::vector<double>;

    Metric(const Labels& labels);

    const Labels& getLabels() const;

protected:
    bool enabled = true;

private:
    Labels labels;
};

class MetricCounter : public Metric {
public:
    MetricCounter(const Labels& labels, prometheus::Counter& counterImpl);

    // API
    void increment(double value = 1.0f);

private:
    // Prometheus internals
    prometheus::Counter& counterImpl;

    friend class MetricFamily<MetricCounter>;
};

class MetricGauge : public Metric {
public:
    MetricGauge(const Labels& labels, prometheus::Gauge& gaugeImpl);

    // API
    void increment(double value = 1.0f);
    void decrement(double value = 1.0f);

private:
    // Prometheus internals
    prometheus::Gauge& gaugeImpl;

    friend class MetricFamily<MetricGauge>;
};

class MetricHistogram : public Metric {
    BucketBoundaries bucketBoundaries;

public:
    MetricHistogram(const Labels& labels, const BucketBoundaries& bucketBoundaries, prometheus::Histogram& histogramImpl);

    // API
    void observe(double value);

private:
    // Prometheus internals
    prometheus::Histogram& histogramImpl;

    friend class MetricFamily<MetricHistogram>;
};

}  // namespace ovms
