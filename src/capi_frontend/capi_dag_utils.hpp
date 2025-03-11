//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include <utility>

#include <openvino/openvino.hpp>

#include "../ovms.h"
#include "../dags/pipelinedefinitionstatus.hpp"

namespace ovms {
class InferenceResponse;
class Status;
OVMS_ServableState convertToServableState(ovms::PipelineDefinitionStateCode code);

Status prepareConsolidatedTensorImpl(InferenceResponse* response, const std::string& name, ov::element::Type_t precision, const ov::Shape& shape, char*& bufferOut, size_t size);
}  // namespace ovms
