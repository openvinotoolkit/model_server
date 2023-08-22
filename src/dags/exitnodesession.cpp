//*****************************************************************************
// Copyright 2021-2022 Intel Corporation
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
#include "exitnodesession.hpp"

#include <memory>

#include "../capi_frontend/inferenceresponse.hpp"
#include "gatherexitnodeinputhandler.hpp"
#include "nodesessionmetadata.hpp"

namespace ovms {

template <typename ResponseType>
ExitNodeSession<ResponseType>::ExitNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ResponseType* response) :
    NodeSession(metadata, nodeName, inputsCount, collapsingDetails) {
    if (collapsingDetails.collapsedSessionNames.size() != 0) {
        this->inputHandler = std::make_unique<GatherExitNodeInputHandler<ResponseType>>(inputsCount, collapsingDetails, response);
    }
}

template <typename ResponseType>
const TensorMap& ExitNodeSession<ResponseType>::getInputTensors() const {
    return this->inputHandler->getInputs();
}

template <typename ResponseType>
ExitNodeSession<ResponseType>::~ExitNodeSession() = default;

template class ExitNodeSession<InferenceResponse>;
template ExitNodeSession<tensorflow::serving::PredictResponse>::ExitNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, tensorflow::serving::PredictResponse* response);
template ExitNodeSession<::KFSResponse>::ExitNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ::KFSResponse* response);

template const TensorMap& ExitNodeSession<tensorflow::serving::PredictResponse>::getInputTensors() const;
template const TensorMap& ExitNodeSession<::KFSResponse>::getInputTensors() const;

}  // namespace ovms
