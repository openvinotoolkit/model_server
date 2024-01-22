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

#include <string>

#include <rapidjson/schema.h>

#include "status.hpp"

namespace ovms {
extern const std::string MODELS_CONFIG_SCHEMA;
extern const std::string MODEL_CONFIG_DEFINITION2;
extern const char* MODELS_MAPPING_SCHEMA;
extern const std::string MEDIAPIPE_SUBCONFIG_SCHEMA;

Status validateJsonAgainstSchema(rapidjson::Document& json, const char* schema, bool detailedError = false);
}  // namespace ovms
