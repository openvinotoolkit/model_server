//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <cxxopts.hpp>

#include "graph_cli_parser.hpp"

namespace ovms {

struct HFSettingsImpl;
struct RerankGraphSettingsImpl;
class Status;

class RerankGraphCLIParser : public GraphCLIParser {
public:
    RerankGraphCLIParser() = default;
    cxxopts::ParseResult parse(const std::vector<std::string>& unmatchedOptions);
    void prepare(HFSettingsImpl& hfSettings, const std::string& modelName);

    void printHelp();
    void createOptions();

private:
    static RerankGraphSettingsImpl& defaultGraphSettings();
};

}  // namespace ovms
