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

namespace ovms {

Metric::Metric(const std::map<std::string, std::string>& labels) :
    labels(labels) {}

bool Metric::hasLabel(const std::string& label) const {
    return this->labels.find(label) != this->labels.end();
}

MetricCounter::MetricCounter(const std::map<std::string, std::string>& labels, prometheus::Counter& counterImpl) :
    Metric(labels),
    counterImpl(counterImpl) {}

void MetricCounter::increment() {
    this->counterImpl.Increment();
}

void MetricCounter::decrement() {
    throw std::logic_error("cannot decrement counter");
}

MetricGauge::MetricGauge(const std::map<std::string, std::string>& labels, prometheus::Gauge& gaugeImpl) :
    Metric(labels),
    gaugeImpl(gaugeImpl) {}

void MetricGauge::increment() {
    this->gaugeImpl.Increment();
}

void MetricGauge::decrement() {
    this->gaugeImpl.Decrement();
}

}  // namespace ovms
