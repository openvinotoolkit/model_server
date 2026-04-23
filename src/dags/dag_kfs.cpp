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
// KFS frontend-specific headers — must precede _impl.hpp includes
// so that two-phase lookup resolves dependent names at instantiation.
// kfs_utils.hpp must precede deserialization.hpp (getBinaryInput/getBinaryInputsSize).
#include "src/kfs_frontend/kfs_utils.hpp"
#include "src/kfs_frontend/kfs_request_utils.hpp"
#include "src/kfs_frontend/deserialization.hpp"
#include "src/kfs_frontend/serialization.hpp"

#include "entry_node_impl.hpp"
#include "exit_node_impl.hpp"
#include "exitnodesession_impl.hpp"
#include "pipelinedefinition_create_impl.hpp"
#include "pipeline_factory_create_impl.hpp"

namespace ovms {

template class EntryNode<::KFSRequest>;

template class ExitNode<::KFSResponse>;

template class ExitNodeSession<::KFSResponse>;

template Status PipelineDefinition::create<::KFSRequest, ::KFSResponse>(
    std::unique_ptr<Pipeline>& pipeline,
    const ::KFSRequest* request,
    ::KFSResponse* response,
    ModelInstanceProvider& modelInstanceProvider);

template Status PipelineFactory::create<::KFSRequest, ::KFSResponse>(
    std::unique_ptr<Pipeline>& pipeline,
    const std::string& name,
    const ::KFSRequest* request,
    ::KFSResponse* response,
    ModelInstanceProvider& modelInstanceProvider) const;

}  // namespace ovms
