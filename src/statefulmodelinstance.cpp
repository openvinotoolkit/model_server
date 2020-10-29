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
#include "statefulmodelinstance.hpp"


using namespace InferenceEngine;

namespace ovms {
const Status StatefulModelInstance::validateNumberOfInputs(const tensorflow::serving::PredictRequest* request, const size_t expectedNumberOfInputs) {
    // Begin with number of inputs required by the model and increase it with special inputs for sequence handling
    auto completeInputsNumber = expectedNumberOfInputs;
    for (auto specialInputName : SPECIAL_INPUT_NAMES) {
        if (request->inputs().count(specialInputName))
            completeInputsNumber++;
    }
    return ModelInstance::validateNumberOfInputs(request, completeInputsNumber);
}
}