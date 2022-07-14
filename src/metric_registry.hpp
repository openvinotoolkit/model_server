//*****************************************************************************
// Copyright 2018-2020 Intel Corporation
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
#include <memory>
#include <string>
#include <vector>

#include <prometheus/counter.h>
#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

#include "modelversion.hpp"

namespace ovms {

enum class MetricKind {
    COUNTER,
    GAUGE,
    HISTOGRAM
};

class Metric {
public:
    virtual void increment(double val) = 0;
    virtual void decrement(double val) = 0;
    virtual void set(double val) = 0;
};

class MetricCounter : public Metric {
    prometheus::Counter& counter;

public:
    MetricCounter(prometheus::Counter& counter) :
        counter(counter) {}
    void increment(double val) override {
        this->counter.Increment(val);
    }
    void decrement(double val) override {
        throw std::logic_error("cannot decrement a counter");
    }
    void set(double val) override {
        throw std::logic_error("cannot set a counter");
    }
};

class MetricRegistry;
class MetricFamily {
    MetricRegistry& registry;

    MetricKind kind;
    std::string name, description;

    std::vector<std::shared_ptr<Metric>> metrics;

public:
    MetricFamily(MetricKind kind, const std::string& name, const std::string& description, MetricRegistry& registry) :
        registry(registry),
        kind(kind),
        name(name),
        description(description) {}

    std::shared_ptr<Metric> add(std::map<std::string, std::string> labels);
};

class MetricRegistry {
    std::shared_ptr<prometheus::Registry> registry;

    std::shared_ptr<MetricFamily> inferOk;
    std::shared_ptr<MetricFamily> inferFail;

    MetricRegistry() :
        registry(std::make_shared<prometheus::Registry>()),
        inferOk(this->createFamily(MetricKind::COUNTER, "infer_ok", "Success requests")),
        inferFail(this->createFamily(MetricKind::COUNTER, "infer_fail", "Failed requests")) {
    }

    MetricRegistry(const MetricRegistry&) = delete;
    MetricRegistry(MetricRegistry&&) = delete;

    std::vector<std::shared_ptr<MetricFamily>> families;

public:
    static MetricRegistry& getInstance() {
        static MetricRegistry instance;
        return instance;
    }

    std::shared_ptr<prometheus::Registry> getRegistry() {
        return this->registry;
    }

    std::shared_ptr<MetricFamily> createFamily(
        MetricKind kind,
        const std::string& name,
        const std::string& description) {
        return this->families.emplace_back(std::make_shared<MetricFamily>(
            kind, name, description, *this));
    }

    MetricFamily& getInferOkFamily() { return *this->inferOk; }
    MetricFamily& getInferFailFamily() { return *this->inferFail; }

    std::string serializeToString() const {
        prometheus::TextSerializer serializer;
        return serializer.Serialize(this->registry->Collect());
    }
};

}  // namespace ovms
