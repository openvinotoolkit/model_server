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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

namespace ovms {

class Metric {
    std::unordered_map<std::string, std::string> labels;
public:
    Metric(const std::unordered_map<std::string, std::string>& labels)
        : labels(labels) {}

    bool hasLabel(const std::string& label) const { return this->labels.find(label) != this->labels.end(); }
};

class MetricFamily {
    std::string name, description;
    std::vector<std::shared_ptr<Metric>> metrics;
public:
    MetricFamily(const std::string& name, const std::string& description, prometheus::Registry& registryImplRef)
        : name(name), description(description), registryImplRef(registryImplRef) {}
    
    const std::string& getDesc() const { return description; }

    std::shared_ptr<Metric> addMetric(const std::unordered_map<std::string, std::string>& labels) {
        return this->metrics.emplace_back(std::make_shared<Metric>(labels));
    }
private:
    // Prometheus internals
    prometheus::Registry& registryImplRef;
};

class MetricRegistry {
    std::vector<std::shared_ptr<MetricFamily>> families;
public:
    MetricRegistry() {}

    std::shared_ptr<MetricFamily> createFamily(const std::string& name, const std::string& description) {
        return this->families.emplace_back(std::make_shared<MetricFamily>(name, description, this->registryImpl));
    }

    std::string collect() const {
        prometheus::TextSerializer serializer;
        return serializer.Serialize(this->registryImpl.Collect());
    }
private:
    // Prometheus internals
    prometheus::Registry registryImpl;
};

}  // namespace ovms
