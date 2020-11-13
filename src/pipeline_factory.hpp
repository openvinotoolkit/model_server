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
#pragma once

#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "pipeline.hpp"
#include "pipelinedefinition.hpp"
#include "status.hpp"

namespace ovms {

class ModelManager;

class PipelineFactory {
    std::map<std::string, std::unique_ptr<PipelineDefinition>> definitions;
    mutable std::shared_mutex definitionsMtx;

public:
    Status createDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections,
        ModelManager& manager);

    bool definitionExists(const std::string& name) const {
        std::shared_lock lock(definitionsMtx);
        return definitions.count(name) == 1;
    }

    Status create(std::unique_ptr<Pipeline>& pipeline,
        const std::string& name,
        const tensorflow::serving::PredictRequest* request,
        tensorflow::serving::PredictResponse* response,
        ModelManager& manager) const;

    PipelineDefinition* findDefinitionByName(const std::string& name) const {
        std::shared_lock lock(definitionsMtx);
        auto it = definitions.find(name);
        if (it == std::end(definitions)) {
            return nullptr;
        } else {
            return it->second.get();
        }
    }
};

}  // namespace ovms
