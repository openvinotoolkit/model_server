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

#include <prometheus/text_serializer.h>

#include "metric_family.hpp"

namespace ovms {

MetricRegistry::MetricRegistry() = default;

// std::shared_ptr<MetricFamily> MetricRegistry::createFamily(MetricKind kind, const std::string& name, const std::string& description) {
//     return this->families.emplace_back(std::make_shared<MetricFamily>(kind, name, description, this->registryImpl));
// }

std::string MetricRegistry::collect() const {
    prometheus::TextSerializer serializer;
    return serializer.Serialize(this->registryImpl.Collect());
}

}  // namespace ovms
