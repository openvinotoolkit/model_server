//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <openvino/openvino.hpp>

#include "../precision.hpp"
#include "../shape.hpp"
#include "nodeinputhandler.hpp"
#include "nodesessionmetadata.hpp"
#include "session_id.hpp"

namespace ovms {

using shard_map_t = std::unordered_map<session_id_t, ov::Tensor>;

class CollapseDetails;

class GatherNodeInputHandler : public NodeInputHandler {
    std::unordered_map<std::string, shard_map_t> shardsStorage;
    std::unique_ptr<CollapseDetails> collapsingDetails;

public:
    // TODO: Investigate why windows does not see this symbol
    GatherNodeInputHandler(uint32_t inputsMissingCount, const CollapseDetails& collapsingDetails) :
        NodeInputHandler(inputsMissingCount),
        collapsingDetails(std::make_unique<CollapseDetails>(collapsingDetails)) {
        remainingDependencies = std::accumulate(
            collapsingDetails.collapsedSessionSizes.begin(),
            collapsingDetails.collapsedSessionSizes.end(),
            remainingDependencies,
            std::multiplies<session_id_t>());
    }
    Status setInput(const std::string& inputName, TensorWithSource& tensor, session_id_t shardId) override;
    Status notifyFinishedDependency() override;

protected:
    virtual Status prepareConsolidatedTensor(ov::Tensor& tensorOut, const std::string& name, ov::element::Type_t precision, const ov::Shape& shape) const;
};
}  // namespace ovms
