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
#include "metric_registry.hpp"

namespace ovms {

std::shared_ptr<Metric> MetricFamily::add(std::map<std::string, std::string> labels) {
    if (this->kind == MetricKind::COUNTER) {
        auto& promFamily = prometheus::BuildCounter()
                               .Name(this->name)
                               .Help(this->description)
                               .Register(*this->registry.getRegistry());
        return metrics.emplace_back(std::make_shared<MetricCounter>(promFamily.Add(labels)));
    } else {
        throw std::logic_error("not implemented");
    }
    return nullptr;
}

}  // namespace ovms
