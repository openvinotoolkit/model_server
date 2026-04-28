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
#pragma once

#include <set>
#include <string>

#include "src/port/rapidjson_document.hpp"

namespace ovms {

class CustomNodeLibraryManager;
class DagResourceManager;
class MetricConfig;
class MetricRegistry;
class ModelInstanceProvider;
class PipelineFactory;
class ServableNameChecker;
class Status;

/**
 * @brief Loads pipelines configuration from JSON document, creates pipeline definitions and adds them to pipeline factory.
 */
Status loadPipelinesConfig(
    rapidjson::Document& configJson,
    PipelineFactory& factory,
    ModelInstanceProvider& modelInstanceProvider,
    ServableNameChecker& nameChecker,
    DagResourceManager& resourceMgr,
    const CustomNodeLibraryManager& customNodeLibraryManager,
    MetricRegistry* metricRegistry,
    const MetricConfig* metricConfig);

}  // namespace ovms
