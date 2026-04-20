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
#include "../tfs_frontend/tfs_utils.hpp"

#pragma warning(push)
#pragma warning(disable : 4624 6001 6385 6386 6326 6011 4457 6308 6387 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "exitnodesession_impl.hpp"

namespace ovms {

template ExitNodeSession<tensorflow::serving::PredictResponse>::ExitNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, tensorflow::serving::PredictResponse* response);
template const TensorMap& ExitNodeSession<tensorflow::serving::PredictResponse>::getInputTensors() const;

}  // namespace ovms
