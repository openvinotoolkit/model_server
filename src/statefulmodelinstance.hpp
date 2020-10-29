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

#include "modelinstance.hpp"

namespace ovms {


/**
     * @brief This class contains all the information about inference engine model
     */
class StatefulModelInstance : public ModelInstance {
private:
     static constexpr std::array<const char*, 2> SPECIAL_INPUT_NAMES{"sequence_id", "sequence_control_input"};
protected:
     // Override validateNumberOfInputs method to handle special inputs
    const Status validateNumberOfInputs(const tensorflow::serving::PredictRequest* request, const size_t expectedNumberOfInputs);
};
}  // namespace ovms
