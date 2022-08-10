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
#include <vector>

#include <prometheus/registry.h>

#include "metric_kind.hpp"

namespace ovms {

class MetricFamilyBase;
template <typename T>
class MetricFamily;

class MetricRegistry {
    std::vector<std::shared_ptr<MetricFamilyBase>> families;

public:
    MetricRegistry();

    template <typename T>
    std::shared_ptr<MetricFamily<T>> createFamily(MetricKind kind, const std::string& name, const std::string& description) {
        std::shared_ptr<MetricFamilyBase> family = std::make_shared<MetricFamily<T>>(kind, name, description, this->registryImpl);
        this->families.emplace_back(family);
        return std::dynamic_pointer_cast<MetricFamily<T>>(family);
    }

    // Returns all collected metrics in "Prometheus Text Exposition Format".
    std::string collect() const;

private:
    // Prometheus internals
    prometheus::Registry registryImpl;
};

}  // namespace ovms
