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
#include "model.h"

namespace ovms {

Status Model::addVersion(   const std::string& name,
                            const std::string& path,
                            const std::string& backend,
                            const int64_t version,
                            const size_t batchSize,
                            const std::vector<size_t>& shape) {
    std::shared_ptr<ModelVersion> modelVersion = std::make_shared<ModelVersion>();
    auto status = modelVersion->loadModel(path, backend, version, batchSize, shape);
    if (status != Status::OK) {
        return status;
    }
    this->name = name;
    modelVersions.push_back(std::move(modelVersion));
    
    return Status::OK;
}

Status Model::dropVersion(const ModelVersion& modelVersion) {
    return Status::OK;
}

} // namespace ovms
