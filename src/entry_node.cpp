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
#include "entry_node.hpp"

#include <functional>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop

namespace ovms {

Status EntryNode::fetchResults(BlobMap& outputs) {
    // Fill outputs map with tensorflow predict request inputs. Fetch only those that are required in following nodes
    for (const auto& node : this->next) {
        for (const auto& pair : node.get().getMappingByDependency(*this)) {
            const auto& output_name = pair.first;
            if (outputs.count(output_name) == 1) {
                continue;
            }

            if (request->inputs().count(output_name) == 0) {
                std::stringstream ss;
                ss << "Required input: " << output_name;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[Node: {}] Missing input with specific name", getName(), details);
                return Status(StatusCode::INVALID_MISSING_INPUT, details);
            }
            const auto& tensor_proto = request->inputs().at(output_name);
            InferenceEngine::Blob::Ptr blob;
            SPDLOG_DEBUG("[Node: {}] Deserializing input: {}", getName(), output_name);
            auto status = deserialize(tensor_proto, blob);
            if (!status.ok()) {
                return status;
            }

            outputs[output_name] = blob;

            SPDLOG_DEBUG("[Node: {}]: blob with name {} has been prepared", getName(), output_name);
        }
    }

    return StatusCode::OK;
}

Status EntryNode::deserialize(const tensorflow::TensorProto& proto, InferenceEngine::Blob::Ptr& blob) {
    InferenceEngine::TensorDesc description;
    if (proto.tensor_content().size() == 0) {
        const std::string details = "Tensor content size can't be 0";
        SPDLOG_DEBUG("[Node: {}] {}", getName(), details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }

    // Assuming content is in proto.tensor_content

    InferenceEngine::SizeVector shape;
    for (int i = 0; i < proto.tensor_shape().dim_size(); i++) {
        shape.emplace_back(proto.tensor_shape().dim(i).size());
    }

    description.setDims(shape);

    size_t tensor_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    if (proto.tensor_content().size() != tensor_count * tensorflow::DataTypeSize(proto.dtype())) {
        std::stringstream ss;
        ss << "Expected: " << tensor_count * tensorflow::DataTypeSize(proto.dtype()) << "; Actual: " << proto.tensor_content().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Node {}] Invalid size of tensor proto - {}", getName(), details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }

    // description.setLayout();  // Layout info is stored in model instance. If we find out it is required, then need to be set right before inference.
    try {
        switch (proto.dtype()) {
        case tensorflow::DataType::DT_FLOAT:
            description.setPrecision(InferenceEngine::Precision::FP32);
            blob = InferenceEngine::make_shared_blob<float>(description, (float*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_UINT8:
            description.setPrecision(InferenceEngine::Precision::U8);
            blob = InferenceEngine::make_shared_blob<uint8_t>(description, (uint8_t*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_INT8:
            description.setPrecision(InferenceEngine::Precision::I8);
            blob = InferenceEngine::make_shared_blob<int8_t>(description, (int8_t*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_INT16:
            description.setPrecision(InferenceEngine::Precision::I16);
            blob = InferenceEngine::make_shared_blob<int16_t>(description, (int16_t*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_INT32:
            description.setPrecision(InferenceEngine::Precision::I32);
            blob = InferenceEngine::make_shared_blob<int32_t>(description, (int32_t*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_HALF:
        case tensorflow::DataType::DT_UINT16:
        case tensorflow::DataType::DT_INT64:
        default: {
            std::stringstream ss;
            ss << "Actual: " << TensorInfo::getDataTypeAsString(proto.dtype());
            const std::string details = ss.str();
            SPDLOG_DEBUG("[Node: {}] Unsupported deserialization precision - {}", getName(), details);
            return Status(StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, details);
        }
        }
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("[Node: {}] Exception thrown during deserialization from make_shared_blob; {}; exception message: {}",
            getName(), status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("[Node: {}] Exception thrown during deserialization from make_shared_blob; {}; exception message: {}",
            getName(), status.string(), e.what());
        return status;
    }

    return StatusCode::OK;
}
}  // namespace ovms
