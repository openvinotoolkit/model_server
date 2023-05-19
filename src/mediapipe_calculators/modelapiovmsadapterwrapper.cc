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
#include "modelapiovmsadapterwrapper.hpp"

#include <memory>

#include <adapters/inference_adapter.h>  // TODO fix path  model_api/model_api/cpp/adapters/include/adapters/inference_adapter.h

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "src/mediapipe_calculators/ovmscalculator.pb.h"
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {
namespace ovms {

AdapterWrapper::AdapterWrapper(::InferenceAdapter* adapter) :
    adapter(adapter) {
    LOG(ERROR) << "Wrapper constr";
}
AdapterWrapper::~AdapterWrapper() {
    LOG(ERROR) << "Wrapper destr";
}
}  // namespace ovms
}  // namespace mediapipe
