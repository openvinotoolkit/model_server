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
// CAPI frontend-specific headers — must precede _impl.hpp includes
// so that two-phase lookup resolves dependent names at instantiation.
#include "src/capi_frontend/capi_dag_utils.hpp"
#include "src/capi_frontend/capi_request_utils.hpp"
#include "src/capi_frontend/capi_utils.hpp"
#include "src/capi_frontend/deserialization.hpp"
#include "src/capi_frontend/inferencerequest.hpp"
#include "src/capi_frontend/inferenceresponse.hpp"
#include "src/capi_frontend/serialization.hpp"

#include "entry_node_impl.hpp"
#include "exit_node_impl.hpp"
#include "exitnodesession_impl.hpp"
#include "pipelinedefinition_create_impl.hpp"
#include "pipeline_factory_create_impl.hpp"

namespace ovms {

const std::string ENTRY_NODE_NAME = "request";
const std::string EXIT_NODE_NAME = "response";
const std::string DEFAULT_PIPELINE_NAME = "";

template class EntryNode<InferenceRequest>;

template class ExitNode<InferenceResponse>;

template class ExitNodeSession<InferenceResponse>;

template Status PipelineDefinition::create<InferenceRequest, InferenceResponse>(
    std::unique_ptr<Pipeline>& pipeline,
    const InferenceRequest* request,
    InferenceResponse* response,
    ModelInstanceProvider& provider);

template Status PipelineFactory::create<InferenceRequest, InferenceResponse>(
    std::unique_ptr<Pipeline>& pipeline,
    const std::string& name,
    const InferenceRequest* request,
    InferenceResponse* response,
    ModelInstanceProvider& provider) const;

}  // namespace ovms
