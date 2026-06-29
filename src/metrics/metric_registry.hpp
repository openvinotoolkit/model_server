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
#include <stdexcept>
#include <prometheus/registry.h>

namespace ovms {

template <typename MetricType>
class MetricFamily;

class MetricRegistry {
public:
    MetricRegistry();
    MetricRegistry(const MetricRegistry&) = delete;
    MetricRegistry(MetricRegistry&&) = delete;
    MetricRegistry& operator=(const MetricRegistry&) = delete;

    template <typename MetricType>
    std::shared_ptr<MetricFamily<MetricType>> createFamily(const std::string& name, const std::string& description) {
        try {
            return std::shared_ptr<MetricFamily<MetricType>>(
                new MetricFamily<MetricType>(name, description, this->registryImpl));
        } catch (std::invalid_argument&) {
            return nullptr;
        }
    }

    template <typename MetricType>
    bool remove(std::shared_ptr<MetricFamily<MetricType>> family);

    // Returns all collected metrics in "Prometheus Text Exposition Format".
    std::string collect() const;

private:
    prometheus::Registry registryImpl;
};

}  // namespace ovms
