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
#include "metric_registry.hpp"

#include <utility>

#include <prometheus/family.h>
#include <prometheus/text_serializer.h>

#include "metric.hpp"
#include "metric_family.hpp"

namespace ovms {

MetricRegistry::MetricRegistry() = default;

std::shared_ptr<MetricFamily<MetricCounter>> MetricRegistry::registerCustomCounterFamily(const std::string& name, const std::string& help) {
    std::lock_guard<std::mutex> lock(customFamiliesMutex);
    const auto existing = customCounterFamilies.find(name);
    if (existing != customCounterFamilies.end()) {
        if (existing->second.first != help) {
            throw std::invalid_argument("Custom counter registered with conflicting help");
        }
        return existing->second.second;
    }
    auto family = createFamily<MetricCounter>(name, help);
    if (!family) {
        throw std::invalid_argument("Cannot register custom counter family");
    }
    customCounterFamilies.emplace(name, std::make_pair(help, family));
    return family;
}

std::string MetricRegistry::collect() const {
    prometheus::TextSerializer serializer;
    return serializer.Serialize(this->registryImpl.Collect());
}

template <>
bool MetricRegistry::remove(std::shared_ptr<MetricFamily<MetricCounter>> family) {
    return this->registryImpl.Remove(*static_cast<prometheus::Family<prometheus::Counter>*>(family->familyImplRef));
}

template <>
bool MetricRegistry::remove(std::shared_ptr<MetricFamily<MetricGauge>> family) {
    return this->registryImpl.Remove(*static_cast<prometheus::Family<prometheus::Gauge>*>(family->familyImplRef));
}

template <>
bool MetricRegistry::remove(std::shared_ptr<MetricFamily<MetricHistogram>> family) {
    return this->registryImpl.Remove(*static_cast<prometheus::Family<prometheus::Histogram>*>(family->familyImplRef));
}

}  // namespace ovms
