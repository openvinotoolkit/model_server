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
#include "rest_parser.hpp"

#include <functional>

namespace ovms {

RestParser::RestParser(const tensor_map_t& tensors) {
    for (const auto& kv : tensors) {
        const auto& name = kv.first;
        const auto& tensor = kv.second;
        auto& input = (*requestProto.mutable_inputs())[name];
        input.set_dtype(tensor->getPrecisionAsDataType());
        input.mutable_tensor_content()->reserve(std::accumulate(
                                                    tensor->getShape().begin(),
                                                    tensor->getShape().end(),
                                                    1,
                                                    std::multiplies<size_t>()) *
                                                DataTypeSize(tensor->getPrecisionAsDataType()));
    }
}

bool RestParser::parseArray(rapidjson::Value& doc, int dim, tensorflow::TensorProto& proto) {
    if (!doc.IsArray()) {
        return false;
    }
    if (doc.GetArray().Size() == 0) {
        return false;
    }
    if (!setDimOrValidate(proto, dim, doc.GetArray().Size())) {
        return false;
    }
    if (doc.GetArray()[0].IsArray()) {
        for (auto& itr : doc.GetArray()) {
            if (!parseArray(itr, dim + 1, proto)) {
                return false;
            }
        }
        return true;
    } else {
        for (auto& value : doc.GetArray()) {
            if (!addValue(proto, value)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool RestParser::parseInstance(rapidjson::Value& doc) {
    if (doc.GetObject().MemberCount() == 0) {
        return false;
    }
    for (auto& itr : doc.GetObject()) {
        auto& proto = (*requestProto.mutable_inputs())[itr.name.GetString()];
        increaseBatchSize(proto);
        if (!parseArray(itr.value, 1, proto)) {
            return false;
        }
    }
    return true;
}

bool RestParser::isBatchSizeEqualForAllInputs() const {
    int64_t size = 0;
    for (const auto& kv : requestProto.inputs()) {
        if (size == 0) {
            size = kv.second.tensor_shape().dim(0).size();
        } else if (kv.second.tensor_shape().dim(0).size() != size) {
            return false;
        }
    }
    return true;
}

Status RestParser::parseRowFormat(rapidjson::Value& node) {
    order = Order::ROW;
    if (!node.IsArray()) {
        return StatusCode::REST_INSTANCES_NOT_AN_ARRAY;
    }
    if (node.GetArray().Size() == 0) {
        return StatusCode::REST_NO_INSTANCES_FOUND;
    }
    if (node.GetArray()[0].IsObject()) {
        // named format
        for (auto& instance : node.GetArray()) {
            if (!instance.IsObject()) {
                return StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT;
            }
            if (!this->parseInstance(instance)) {
                return StatusCode::REST_COULD_NOT_PARSE_INSTANCE;
            }
        }
    } else if (node.GetArray()[0].IsArray()) {
        // no named format
        if (requestProto.inputs_size() != 1) {
            return StatusCode::REST_INPUT_NOT_PREALLOCATED;
        }
        if (!parseArray(node, 0, requestProto.mutable_inputs()->begin()->second)) {
            return StatusCode::REST_COULD_NOT_PARSE_INSTANCE;
        } else {
            format = Format::NONAMED;
            return StatusCode::OK;
        }
    } else {
        return StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED;
    }
    if (!isBatchSizeEqualForAllInputs()) {
        return StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER;
    }
    format = Format::NAMED;
    return StatusCode::OK;
}

Status RestParser::parseColumnFormat(rapidjson::Value& node) {
    order = Order::COLUMN;
    // no named format
    if (node.IsArray()) {
        if (requestProto.inputs_size() != 1) {
            return StatusCode::REST_INPUT_NOT_PREALLOCATED;
        }
        if (!parseArray(node, 0, requestProto.mutable_inputs()->begin()->second)) {
            return StatusCode::REST_COULD_NOT_PARSE_INPUT;
        }
        format = Format::NONAMED;
        return StatusCode::OK;
    }
    // named format
    if (!node.IsObject()) {
        return StatusCode::REST_INPUTS_NOT_AN_OBJECT;
    }
    if (node.GetObject().MemberCount() == 0) {
        return StatusCode::REST_NO_INPUTS_FOUND;
    }
    for (auto& kv : node.GetObject()) {
        auto& proto = (*requestProto.mutable_inputs())[kv.name.GetString()];
        if (!parseArray(kv.value, 0, proto)) {
            return StatusCode::REST_COULD_NOT_PARSE_INPUT;
        }
    }
    format = Format::NAMED;
    return StatusCode::OK;
}

Status RestParser::parse(const char* json) {
    rapidjson::Document doc;
    if (doc.Parse(json).HasParseError()) {  // TODO: Use ParseStream?
        return StatusCode::JSON_INVALID;
    }
    if (!doc.IsObject()) {
        return StatusCode::REST_BODY_IS_NOT_AN_OBJECT;
    }
    auto instancesItr = doc.FindMember("instances");
    auto inputsItr = doc.FindMember("inputs");
    if (instancesItr != doc.MemberEnd() && inputsItr != doc.MemberEnd()) {
        return StatusCode::REST_PREDICT_UNKNOWN_ORDER;
    }
    if (instancesItr != doc.MemberEnd()) {
        return parseRowFormat(instancesItr->value);
    }
    if (inputsItr != doc.MemberEnd()) {
        return parseColumnFormat(inputsItr->value);
    }
    return StatusCode::REST_PREDICT_UNKNOWN_ORDER;
}

void RestParser::increaseBatchSize(tensorflow::TensorProto& proto) {
    if (proto.tensor_shape().dim_size() < 1) {
        proto.mutable_tensor_shape()->add_dim()->set_size(0);
    }
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(proto.tensor_shape().dim(0).size() + 1);
}

bool RestParser::setDimOrValidate(tensorflow::TensorProto& proto, int dim, int size) {
    if (proto.tensor_shape().dim_size() > dim) {
        return proto.tensor_shape().dim(dim).size() == size;
    } else {
        while (proto.tensor_shape().dim_size() <= dim) {
            proto.mutable_tensor_shape()->add_dim()->set_size(0);
        }
        proto.mutable_tensor_shape()->mutable_dim(dim)->set_size(size);
        return true;
    }
}

template <typename T>
bool addToTensorContent(tensorflow::TensorProto& proto, T value) {
    if (sizeof(T) != DataTypeSize(proto.dtype())) {
        return false;
    }
    proto.mutable_tensor_content()->append(reinterpret_cast<const char*>(&value), sizeof(T));
    return true;
}

template <typename T>
bool addToTensorContent(tensorflow::TensorProto& proto, const rapidjson::Value& value) {
    if (value.IsDouble()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetDouble()));
    }
    if (value.IsInt64()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetInt64()));
    }
    if (value.IsUint64()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetUint64()));
    }
    if (value.IsInt()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetInt()));
    }
    if (value.IsUint()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetUint()));
    }
    return false;
}

bool addToHalfVal(tensorflow::TensorProto& proto, const rapidjson::Value& value) {
    if (value.IsDouble()) {
        proto.add_half_val(value.GetDouble());
        return true;
    }
    if (value.IsInt64()) {
        proto.add_half_val(value.GetInt64());
        return true;
    }
    if (value.IsUint64()) {
        proto.add_half_val(value.GetUint64());
        return true;
    }
    if (value.IsInt()) {
        proto.add_half_val(value.GetInt());
        return true;
    }
    if (value.IsUint()) {
        proto.add_half_val(value.GetUint());
        return true;
    }
    return false;
}

bool addToIntVal(tensorflow::TensorProto& proto, const rapidjson::Value& value) {
    if (value.IsDouble()) {
        proto.add_int_val(value.GetDouble());
        return true;
    }
    if (value.IsInt64()) {
        proto.add_int_val(value.GetInt64());
        return true;
    }
    if (value.IsUint64()) {
        proto.add_int_val(value.GetUint64());
        return true;
    }
    if (value.IsInt()) {
        proto.add_int_val(value.GetInt());
        return true;
    }
    if (value.IsUint()) {
        proto.add_int_val(value.GetUint());
        return true;
    }
    return false;
}

bool RestParser::addValue(tensorflow::TensorProto& proto, const rapidjson::Value& value) {
    if (!value.IsNumber()) {
        return false;
    }
    switch (proto.dtype()) {
    case tensorflow::DataType::DT_FLOAT:
        return addToTensorContent<float>(proto, value);
    case tensorflow::DataType::DT_HALF:
        return addToHalfVal(proto, value);  // TODO(CVS-34457): Deserialize always from tensor_content, otherwise try from HalfVal/IntVal/etc
    case tensorflow::DataType::DT_DOUBLE:
        return addToTensorContent<double>(proto, value);
    case tensorflow::DataType::DT_INT32:
        return addToTensorContent<int32_t>(proto, value);
    case tensorflow::DataType::DT_INT16:
        return addToTensorContent<int16_t>(proto, value);
    case tensorflow::DataType::DT_UINT16:
        return addToIntVal(proto, value);  // TODO(CVS-34457): Deserialize always from tensor_content, otherwise try from HalfVal/IntVal/etc
    case tensorflow::DataType::DT_INT8:
        return addToTensorContent<int8_t>(proto, value);
    case tensorflow::DataType::DT_UINT8:
        return addToTensorContent<uint8_t>(proto, value);
    case tensorflow::DataType::DT_INT64:
        return addToTensorContent<int64_t>(proto, value);
    case tensorflow::DataType::DT_UINT32:
        return addToTensorContent<uint32_t>(proto, value);
    case tensorflow::DataType::DT_UINT64:
        return addToTensorContent<uint64_t>(proto, value);
    default:
        return false;
    }
}

}  // namespace ovms
