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

MetricCounter::MetricCounter(prometheus::Counter& counterImpl) :
    counterImpl(counterImpl) {}

void MetricCounter::increment(double value) {
    this->counterImpl.Increment(value);
}

MetricGauge::MetricGauge(prometheus::Gauge& gaugeImpl) :
    gaugeImpl(gaugeImpl) {}

void MetricGauge::increment(double value) {
    this->gaugeImpl.Increment(value);
}

void MetricGauge::decrement(double value) {
    this->gaugeImpl.Decrement(value);
}

void MetricGauge::set(double value) {
    this->gaugeImpl.Set(value);
}

MetricHistogram::MetricHistogram(prometheus::Histogram& histogramImpl) :
    histogramImpl(histogramImpl) {}

void MetricHistogram::observe(double value) {
    this->histogramImpl.Observe(value);
}

}  // namespace ovms
