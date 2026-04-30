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

#include <string>

namespace prometheus {
class Counter;
class Gauge;
class Histogram;
}  // namespace prometheus

namespace ovms {

#define INCREMENT_IF_ENABLED(metric) \
    if (metric) {                    \
        metric->increment();         \
    }
#define DECREMENT_IF_ENABLED(metric) \
    if (metric) {                    \
        metric->decrement();         \
    }
#define SET_IF_ENABLED(metric, val) \
    if (metric) {                   \
        metric->set(val);           \
    }
#define OBSERVE_IF_ENABLED(metric, val) \
    if (metric) {                       \
        metric->observe(val);           \
    }

template <typename T>
class MetricFamily;

class MetricCounter {
private:
    MetricCounter(prometheus::Counter& counterImpl);
    MetricCounter(const MetricCounter&) = delete;
    MetricCounter(MetricCounter&&) = delete;
    MetricCounter& operator=(const MetricCounter&) = delete;

public:
    void increment(double value = 1.0f);

private:
    prometheus::Counter& counterImpl;

    friend class MetricFamily<MetricCounter>;
};

class MetricGauge {
public:
    MetricGauge(prometheus::Gauge& gaugeImpl);
    MetricGauge(const MetricGauge&) = delete;
    MetricGauge(MetricCounter&&) = delete;
    MetricGauge& operator=(const MetricGauge&) = delete;

    void increment(double value = 1.0f);
    void decrement(double value = 1.0f);
    void set(double value = 1.0f);

private:
    prometheus::Gauge& gaugeImpl;

    friend class MetricFamily<MetricGauge>;
};

class MetricHistogram {
public:
    MetricHistogram(prometheus::Histogram& histogramImpl);
    MetricHistogram(const MetricHistogram&) = delete;
    MetricHistogram(MetricCounter&&) = delete;
    MetricHistogram& operator=(const MetricHistogram&) = delete;

    void observe(double value);

private:
    prometheus::Histogram& histogramImpl;

    friend class MetricFamily<MetricHistogram>;
};

// Increments upon destruction, however can be disabled to do so.
class MetricCounterGuard {
    bool active = true;
    MetricCounter* metric;

public:
    MetricCounterGuard(MetricCounter* metric) :
        metric(metric) {
    }
    void disable() { active = false; }
    ~MetricCounterGuard() {
        if (active) {
            INCREMENT_IF_ENABLED(metric);
        }
    }
};

// Increments upon construction, decrements upon destruction.
class MetricGaugeGuard {
    MetricGauge* metric;

public:
    MetricGaugeGuard(MetricGauge* metric) :
        metric(metric) {
        INCREMENT_IF_ENABLED(metric);
    }
    ~MetricGaugeGuard() {
        DECREMENT_IF_ENABLED(metric);
    }
};

}  // namespace ovms
