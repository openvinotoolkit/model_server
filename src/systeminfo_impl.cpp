//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include "systeminfo_impl.hpp"

#include <ios>

#include "logging.hpp"
#include "status.hpp"
#include "stringutils.hpp"

namespace ovms {
uint16_t getCoreCountImpl(const std::string& cpusets) {
    auto sets = tokenize(cpusets, ',');
    uint16_t totalCount = 0;
    for (auto& set : sets) {
        if (std::string::npos == set.find('-')) {
            if (stou32(set).has_value()) {
                totalCount += 1;
            } else {
                return 1;
            }
        } else {
            auto bounds = tokenize(set, '-');
            // this handles both single minus numbers as well as two range sets
            if (bounds.size() != 2) {
                return 1;
            }
            auto rbound = stou32(bounds[1]);
            if (!rbound.has_value()) {
                return 1;
            }
            auto lbound = stou32(bounds[0]);
            if (!lbound.has_value()) {
                return 1;
            }
            if (rbound <= lbound) {
                return 1;
            }
            uint16_t setCount = rbound.value() - lbound.value() + 1;
            totalCount += setCount;
        }
    }
    return totalCount;
}
Status getCPUSetFile(std::ifstream& ifs, const std::string& filename) {
    ifs.open(filename, std::ios_base::in);
    if (ifs.fail()) {
        SPDLOG_ERROR("Failed to open file: {}", filename);
        return StatusCode::FILESYSTEM_ERROR;
    }
    return StatusCode::OK;
}
}  // namespace ovms
