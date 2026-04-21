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
// TFS frontend-specific headers — must precede _impl.hpp includes
// so that two-phase lookup resolves dependent names at instantiation.
// tfs_utils.hpp must precede deserialization.hpp (getBinaryInput/getBinaryInputsSize).
#include "src/tfs_frontend/tfs_utils.hpp"
#include "src/tfs_frontend/tfs_request_utils.hpp"
#include "src/tfs_frontend/deserialization.hpp"
#include "src/tfs_frontend/serialization.hpp"

#pragma warning(push)
#pragma warning(disable : 4624 6001 6385 6386 6326 6011 4457 6308 6387 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "entry_node_impl.hpp"
#include "exit_node_impl.hpp"
#include "exitnodesession_impl.hpp"
#include "pipelinedefinition_create_impl.hpp"
#include "pipeline_factory_create_impl.hpp"

namespace ovms {

template class EntryNode<tensorflow::serving::PredictRequest>;

template class ExitNode<tensorflow::serving::PredictResponse>;

template class ExitNodeSession<tensorflow::serving::PredictResponse>;

template Status PipelineDefinition::create<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>(
    std::unique_ptr<Pipeline>& pipeline,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    ModelInstanceProvider& provider);

template Status PipelineFactory::create<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>(
    std::unique_ptr<Pipeline>& pipeline,
    const std::string& name,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    ModelInstanceProvider& provider) const;

}  // namespace ovms
