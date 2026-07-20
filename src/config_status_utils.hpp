//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <string>
#include <vector>

#include "execution_context.hpp"
#include "modelversion.hpp"
#include "modelversionstatus.hpp"

namespace ovms {
class ModelInstanceProvider;
class ServableNameChecker;
class Status;

struct ModelVersionStatusDetails {
    model_version_t version;
    ModelVersionState state;
    ModelVersionStatusErrorCode errorCode;
    std::string errorMessage;
};

using ModelsStatuses = std::map<std::string, std::vector<ModelVersionStatusDetails>>;

Status getAllModelsStatuses(ModelsStatuses& modelsStatuses, ModelInstanceProvider& modelProvider, ServableNameChecker& servableChecker, ExecutionContext context);
Status serializeModelsStatuses2Json(const ModelsStatuses& modelsStatuses, std::string& output);

}  // namespace ovms
