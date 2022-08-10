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

class Metric {
public:
    using Labels = std::map<std::string, std::string>;
    using BucketBoundaries = std::vector<double>;

private:
    Labels labels;

public:
    Metric(const Labels& labels);

    bool hasLabel(const std::string& label) const;

    // virtual void increment() = 0;
    // virtual void decrement() = 0;
    // virtual void observe(double value) = 0;
};

class MetricCounter : public Metric {
public:
    MetricCounter(const Labels& labels, prometheus::Counter& counterImpl);

    void increment();

private:
    // Prometheus internals
    prometheus::Counter& counterImpl;
};

class MetricGauge : public Metric {
public:
    MetricGauge(const Labels& labels, prometheus::Gauge& gaugeImpl);

    void increment();
    void decrement();

private:
    // Prometheus internals
    prometheus::Gauge& gaugeImpl;
};

class MetricHistogram : public Metric {
    BucketBoundaries bucketBoundaries;

public:
    MetricHistogram(const Labels& labels, const BucketBoundaries& bucketBoundaries, prometheus::Histogram& histogramImpl);

    void observe(double value);

private:
    // Prometheus internals
    prometheus::Histogram& histogramImpl;
};

}  // namespace ovms
