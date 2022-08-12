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
#include "metric.hpp"

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>

namespace ovms {

Metric::Metric(const Labels& labels) :
    labels(labels) {}

const Metric::Labels& Metric::getLabels() const {
    return this->labels;
}

/// Counter implementation

MetricCounter::MetricCounter(const Labels& labels, prometheus::Counter& counterImpl) :
    Metric(labels),
    counterImpl(counterImpl) {}

void MetricCounter::increment(double value) {
    if (this->removed) {
        return;
    }
    this->counterImpl.Increment(value);
}

/// Gauge implementation

MetricGauge::MetricGauge(const Labels& labels, prometheus::Gauge& gaugeImpl) :
    Metric(labels),
    gaugeImpl(gaugeImpl) {}

void MetricGauge::increment(double value) {
    if (this->removed) {
        return;
    }
    this->gaugeImpl.Increment(value);
}

void MetricGauge::decrement(double value) {
    if (this->removed) {
        return;
    }
    this->gaugeImpl.Decrement(value);
}

/// Histogram implementation

MetricHistogram::MetricHistogram(const Labels& labels, const BucketBoundaries& bucketBoundaries, prometheus::Histogram& histogramImpl) :
    Metric(labels),
    bucketBoundaries(bucketBoundaries),
    histogramImpl(histogramImpl) {}

void MetricHistogram::observe(double value) {
    if (this->removed) {
        return;
    }
    this->histogramImpl.Observe(value);
}

}  // namespace ovms
