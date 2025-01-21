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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "../logging.hpp"
#include "../ovms.h"  // NOLINT
#include "../regularovtensorfactory.hpp"
#include "../tensorinfo.hpp"
#include "node.hpp"

namespace ovms {
class IOVTensorFactory;
extern const std::string ENTRY_NODE_NAME;

template <typename RequestType>
class EntryNode : public Node {
    const RequestType* request;
    const tensor_map_t inputsInfo;
    const tensor_map_t outputsInfo;  // specifying outputs is not supported for DAGS
    std::unordered_map<int, std::shared_ptr<IOVTensorFactory>> factories;

public:
    EntryNode(const RequestType* request,
        const tensor_map_t& inputsInfo,
        std::optional<int32_t> demultiplyCount = std::nullopt) :
        Node(ENTRY_NODE_NAME, demultiplyCount),
        request(request),
        inputsInfo(inputsInfo) {
        factories.emplace(OVMS_BUFFERTYPE_CPU, std::make_shared<RegularOVTensorFactory>());
    }

    Status execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) override;

    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) override;

protected:
    Status fetchResults(TensorWithSourceMap& outputs);
    Status createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName) override;

public:
    // Entry nodes have no dependency
    void addDependency(Node&, const Aliases&) override {
        throw std::logic_error("This node cannot have dependency");
    }

    Status isInputBinary(const std::string& name, bool& isBinary) const;

    const Status validate();
};

}  // namespace ovms
