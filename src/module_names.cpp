//****************************************************************************
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
#include "module_names.hpp"

namespace ovms {
const std::string PROFILER_MODULE_NAME = "ProfilerModule";
const std::string GRPC_SERVER_MODULE_NAME = "GRPCServerModule";
const std::string HTTP_SERVER_MODULE_NAME = "HTTPServerModule";
const std::string SERVABLE_MANAGER_MODULE_NAME = "ServableManagerModule";
const std::string HF_MODEL_PULL_MODULE_NAME = "HfModelPullModule";
const std::string METRICS_MODULE_NAME = "MetricsModule";
const std::string PYTHON_INTERPRETER_MODULE_NAME = "PythonInterpreterModule";
const std::string CAPI_MODULE_NAME = "C-APIModule";
const std::string SERVABLES_CONFIG_MANAGER_MODULE_NAME = {"ServablesConfigManagerModule"};
}  // namespace ovms
