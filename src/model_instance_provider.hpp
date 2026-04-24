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
#include <memory>
#include <string>

#include "modelversion.hpp"

namespace ovms {

class Model;
class ModelInstance;
class ModelInstanceUnloadGuard;
struct NotifyReceiver;
class Status;
class TensorInfo;
using tensor_map_t = std::map<std::string, std::shared_ptr<const TensorInfo>>;

class ModelInstanceProvider {
public:
    virtual ~ModelInstanceProvider() = default;
    virtual Status getModelInstance(
        const std::string& modelName,
        model_version_t modelVersionId,
        std::shared_ptr<ModelInstance>& modelInstance,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) const = 0;
    virtual const std::shared_ptr<Model> findModelByName(const std::string& name) const = 0;
    virtual const std::shared_ptr<ModelInstance> findModelInstance(const std::string& name, model_version_t version = 0) const = 0;
    virtual bool subscribeToModel(const std::string& name, model_version_t version, NotifyReceiver& receiver) = 0;
    virtual void unsubscribeFromModel(const std::string& name, model_version_t version, NotifyReceiver& receiver) = 0;
    virtual Status getModelInputsInfo(const std::string& name, model_version_t version, tensor_map_t& info) const = 0;
    virtual Status getModelOutputsInfo(const std::string& name, model_version_t version, tensor_map_t& info) const = 0;
    virtual Status hasAutoModelParameters(const std::string& name, model_version_t version, bool& batchAuto, bool& shapeAuto) const = 0;
};

}  // namespace ovms
