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
#include <variant>

#include <cxxopts.hpp>

#include "graph_export/graph_cli_parser.hpp"
#include "graph_export/rerank_graph_cli_parser.hpp"
#include "graph_export/embeddings_graph_cli_parser.hpp"

namespace ovms {

struct ServerSettingsImpl;
struct ModelsSettingsImpl;

class CLIParser {
    std::unique_ptr<cxxopts::Options> options;
    std::unique_ptr<cxxopts::ParseResult> result;
    std::variant<GraphCLIParser, RerankGraphCLIParser, EmbeddingsGraphCLIParser> graphOptionsParser;

public:
    CLIParser() = default;
    void parse(int argc, char** argv);
    void prepare(ServerSettingsImpl*, ModelsSettingsImpl*);

protected:
    void prepareServer(ServerSettingsImpl& serverSettings);
    void prepareModel(ModelsSettingsImpl& modelsSettings, HFSettingsImpl& hfSettings);
    void prepareGraph(HFSettingsImpl& hfSettings, const std::string& modelName, const std::string& modelPath);
    void prepareGraphStart(HFSettingsImpl& hfSettings, ModelsSettingsImpl& modelsSettings);
    bool isHFPullOrPullAndStart(bool isPull, bool isSourceModel, bool isModelRepository, bool isTask);
};

}  // namespace ovms
