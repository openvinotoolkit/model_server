//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "model_version_policy.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>

namespace ovms {

std::shared_ptr<ModelVersionPolicy> ModelVersionPolicy::getDefaultVersionPolicy() {
    return std::make_shared<LatestModelVersionPolicy>(1);
}

AllModelVersionPolicy::operator std::string() const {
    return std::string("all");
}

std::vector<model_version_t> SpecificModelVersionPolicy::filter(std::vector<model_version_t> versions) const {
    std::vector<model_version_t> result;
    std::sort(versions.begin(), versions.end());
    std::set_intersection(
        versions.begin(),
        versions.end(),
        specificVersions.begin(),
        specificVersions.end(),
        std::back_inserter(result));
    return result;
}

SpecificModelVersionPolicy::operator std::string() const {
    std::stringstream versionStream;
    versionStream << "specific: ";
    std::copy(specificVersions.begin(), specificVersions.end(), std::ostream_iterator<model_version_t>(versionStream, " "));
    return versionStream.str();
}

std::vector<model_version_t> LatestModelVersionPolicy::filter(std::vector<model_version_t> versions) const {
    std::vector<model_version_t> result;
    std::sort(versions.begin(), versions.end(), std::greater<model_version_t>());
    for (size_t i = 0; i < numVersions && i < versions.size(); i++) {
        result.push_back(versions[i]);
    }
    return result;
}

LatestModelVersionPolicy::operator std::string() const {
    return std::string("latest: ") + std::to_string(numVersions);
}

}  // namespace ovms
