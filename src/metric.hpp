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

namespace prometheus {
class Counter;
class Gauge;
}  // namespace prometheus

namespace ovms {

class Metric {
    std::map<std::string, std::string> labels;

public:
    Metric(const std::map<std::string, std::string>& labels);

    bool hasLabel(const std::string& label) const;

    virtual void increment() = 0;
    virtual void decrement() = 0;
};

class MetricCounter : public Metric {
public:
    MetricCounter(const std::map<std::string, std::string>& labels, prometheus::Counter& counterImpl);

    void increment() override;
    void decrement() override;

private:
    // Prometheus internals
    prometheus::Counter& counterImpl;
};

class MetricGauge : public Metric {
public:
    MetricGauge(const std::map<std::string, std::string>& labels, prometheus::Gauge& gaugeImpl);

    void increment() override;
    void decrement() override;

private:
    // Prometheus internals
    prometheus::Gauge& gaugeImpl;
};

}  // namespace ovms
