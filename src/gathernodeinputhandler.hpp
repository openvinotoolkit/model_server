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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <openvino/openvino.hpp>

#include "nodeinputhandler.hpp"
#include "session_id.hpp"

namespace ovms {

using shard_map_t = std::unordered_map<session_id_t, std::shared_ptr<ov::runtime::Tensor>>;

class CollapseDetails;

class GatherNodeInputHandler : public NodeInputHandler {
    std::unordered_map<std::string, shard_map_t> shardsStorage;
    std::unique_ptr<CollapseDetails> collapsingDetails;

public:
    GatherNodeInputHandler(uint32_t inputsMissingCount, const CollapseDetails& collapsingDetails);
    Status setInput(const std::string& inputName, std::shared_ptr<ov::runtime::Tensor>& blobPtr, session_id_t shardId) override;
    Status notifyFinishedDependency() override;
};
}  // namespace ovms
