#pragma once
//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "../ovms.h"  // NOLINT
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "src/mediapipe_internal/ovmscalculator.pb.h"
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib

namespace mediapipe {
namespace ovms {

using InferenceOutput = std::map<std::string, ov::Tensor>;
using InferenceInput = std::map<std::string, ov::Tensor>;

// TODO
// * why std::map
class OVMSInferenceAdapter {
    OVMS_Server* cserver{nullptr};
    const std::string servableName;
    uint32_t servableVersion;

public:
    std::unordered_map<std::string, std::string> inputTagToName;
    std::unordered_map<std::string, std::string> outputNameToTag;

public:
    OVMSInferenceAdapter(const std::string& servableName, uint32_t servableVersion = 0);
    virtual ~OVMSInferenceAdapter();
    virtual InferenceOutput infer(const InferenceInput& input);
    virtual void loadModel(const std::shared_ptr<const ov::Model>& model, ov::Core& core,
        const std::string& device, const ov::AnyMap& compilationConfig);
    virtual ov::Shape getInputShape(const std::string& inputName) const;  // TODO
    virtual std::vector<std::string> getInputNames();                // TODO
    virtual std::vector<std::string> getOutputNames();                    // TODO
                                                                                        //    virtual const ov::AnyMap& getModelConfig() const = 0; // TODO
    virtual const std::string& getModelConfig() const; // TODO
};
}  // namespace ovms
}  // namespace mediapipe
