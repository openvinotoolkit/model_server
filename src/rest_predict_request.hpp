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

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <functional>
#include <memory>

#include <rapidjson/document.h>
#include <spdlog/spdlog.h>

#include "tensorinfo.hpp"
#include "modelconfig.hpp"
#include "status.hpp"

namespace ovms {

/**
 * @brief Request order types
 */
enum class Order {
    UNKNOWN,
    ROW,
    COLUMN
};

/**
 * @brief Request format types
 */
enum class Format {
    UNKNOWN,
    NAMED,
    NONAMED
};

/**
 * @brief This class represents shape encapsulation with utility methods used during predict request parsing.
 */
class Shape {
    /**
     * @brief Shape
     */
    shape_t shape;

    /**
     * @brief Sets dimension to certain size, increases number of dimensions if [dim] dimension does not exist.
     * 
     * @param dim Dimension to set or add
     * @param size Size for dimension
     */
    void setDim(size_t dim, size_t size) {
        while (!hasDim(dim)) {
            shape.emplace_back(0);
        }
        shape[dim] = size;
    }

public:
    /**
     * @brief Sets dimension to certain size, check whether dimension already existed.
     * 
     * @param dim Dimension to set or add
     * @param size Size for dimension
     * 
     * @return returns false if dimension already existed and with different size, true otherwise
     */
    bool setDimOrValidate(size_t dim, size_t size) {
        if (hasDim(dim)) {
            return getDim(dim) == size;
        } else {
            setDim(dim, size);
            return true;
        }
    }

    /**
     * @brief Increment 0th-dimension of shape.
     */
    void increaseBatchSize() {
        if (shape.size() == 0) {
            shape.emplace_back(0);
        }
        shape[0]++;
    }

    /**
     * Checks whether [dim] dimension exists.
     * 
     * @return true or false
     */
    bool hasDim(size_t dim) const {
        return shape.size() > dim;
    }

    /**
     * @brief Retrieves [dim] dimension size.
     * 
     * @return size of requested dimension
     */
    size_t getDim(size_t dim) const {
        return shape[dim];
    }

    /**
     * Gets shape
     */
    const shape_t& get() const { return shape; }
};

/**
 * @brief This class represents input with raw data in vector prepared to be passed to OpenVINO.
 */
template<typename T>
struct Input {
    using Data = std::vector<T>;

    /**
     * @brief Shape of input
     */
    Shape shape;

    /**
     * @brief Vector of data with specified data type as template parameter
     */
    Data data;

    /**
     * @brief Parses rapidjson::Value for numeric value and casts to appropriate data type specified as template parameter.
     * 
     * @param value rapidjson Node
     * 
     * @return false indicates there was an error during parsing, true otherwise
     */
    bool push(rapidjson::Value& value) {
        if (!value.IsNumber()) {
            return false;
        }
        if (value.IsDouble()) {
            data.emplace_back(static_cast<T>(value.GetDouble()));
            return true;
        }
        if (value.IsInt64()) {
            data.emplace_back(static_cast<T>(value.GetInt64()));
            return true;
        }
        if (value.IsUint64()) {
            data.emplace_back(static_cast<T>(value.GetUint64()));
            return true;
        }
        if (value.IsInt()) {
            data.emplace_back(static_cast<T>(value.GetInt()));
            return true;
        }
        if (value.IsUint()) {
            data.emplace_back(static_cast<T>(value.GetUint()));
            return true;
        }
        return false;
    }
};

/**
 * @brief This class encapsulates http request body string parsing to inputs ready to be passed to OpenVINO.
 */
template<typename T>
class RestPredictRequest {
    using Inputs = std::unordered_map<std::string, Input<T>>;

    /**
     * @brief Request order
     */
    Order order = Order::UNKNOWN;

    /**
     * @brief Request format
     */
    Format format = Format::UNKNOWN;

    /**
     * @brief Parsed inputs
     */
    Inputs inputs;

    /**
     * @brief Parses rapidjson Node for arrays or numeric values on certain level of nesting.
     * 
     * @param doc rapidjson Node
     * @param dim level of nesting
     * @param input destination for parsing numeric values
     * 
     * @return false if processing failed, true when succeeded
     * 
     * Rapid json node expected to be passed in following structure:
     * [
     *     [...],
     *     [...],
     *     ...
     * ]
     */
    bool parseArray(rapidjson::Value& doc, int dim, Input<T>& input) {
        if (!doc.IsArray()) {
            return false;
        }
        if (doc.GetArray().Size() == 0) {
            return false;
        }
        if (!input.shape.setDimOrValidate(dim, doc.GetArray().Size())) {
            return false;
        }
        if (doc.GetArray()[0].IsArray()) {
            for (auto& itr : doc.GetArray()) {
                if (!parseArray(itr, dim+1, input)) {
                    return false;
                }
            }
            return true;
        } else {
            for (auto& value : doc.GetArray()) {
                if (!input.push(value)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    /**
     * @brief Parses rapidjson Node for inputs in a string(name)=>array(data) format
     * 
     * @param doc rapidjson Node
     * 
     * @return false if processing failed, true when succeeded
     * 
     * Rapid json node expected to be passed in following structure:
     * {
     *     "input1": [[...], [...], ...],
     *     "input2": [[...], [...], ...],
     *     ...
     * }
     */
    bool parseInstance(rapidjson::Value& doc) {
        if (doc.GetObject().MemberCount() == 0) {
            return false;
        }
        for (auto& itr : doc.GetObject()) {
            auto& input = inputs[itr.name.GetString()];

            input.shape.increaseBatchSize();

            if (!parseArray(itr.value, 1, input)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Checks whether all inputs have equal batch size, 0th-dimension
     * 
     * @return true or false
     */
    bool isBatchSizeEqualForAllInputs() const {
        size_t size = 0;
        for (const auto& input : inputs) {
            if (size == 0) {
                size = input.second.shape.get()[0];
            } else if (input.second.shape.get()[0] != size) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Parses row format: list of objects, each object corresponding to one batch with one or multiple inputs.
     *        When no named format is detected, instance is treated as array of single input batches with no name.
     * 
     * @param node rapidjson Node
     * 
     * @return Status indicating if processing succeeded, error code otherwise
     *
     * Rapid json node expected to be passed in following structure:
     * [{inputs...}, {inputs...}, {inputs...}, ...]
     * or:
     * [no named input data batches...]
     */
    Status parseRowFormat(rapidjson::Value& node) {
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
            if (inputs.size() != 1) {
                return StatusCode::REST_INPUT_NOT_PREALLOCATED;
            }
            if (!parseArray(node, 0, inputs.begin()->second)) {
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

    /**
     * @brief Parses column format: object of input:batches key value pairs.
     *        When no named format is detected, instance is treated as array of single input batches with no name.
     * 
     * @param node rapidjson Node
     * 
     * @return Status indicating if processing succeeded, error code otherwise
     * 
     * Rapid json node expected to be passed in following structure:
     * {"inputA": [...], "inputB": [...], ...}
     * or:
     * [no named input data batches...]
     */
    Status parseColumnFormat(rapidjson::Value& node) {
        order = Order::COLUMN;
        // no named format
        if (node.IsArray()) {
            if (inputs.size() != 1) {
                return StatusCode::REST_INPUT_NOT_PREALLOCATED;
            }
            if (!parseArray(node, 0, inputs.begin()->second)) {
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
            auto& input = inputs[kv.name.GetString()];
            if (!parseArray(kv.value, 0, input)) {
                return StatusCode::REST_COULD_NOT_PARSE_INPUT;
            }
        }
        format = Format::NAMED;
        return StatusCode::OK;
    }

public:
    /**
     * @brief Default constructor
     */
    RestPredictRequest() = default;

    /**
     * @brief Constructor for preallocating memory for vector beforehand. Size is calculated from tensor shape required by backend.
     * 
     * @param tensors Tensor map with model input parameters
     */
    RestPredictRequest(const tensor_map_t& tensors) {
        for (const auto& kv : tensors) {
            const auto& name = kv.first;
            const auto& tensor = kv.second;
            auto& input = inputs[name];
            input.data.reserve(std::accumulate(
                tensor->getShape().begin(),
                tensor->getShape().end(),
                1,
                std::multiplies<size_t>()));
        }
    }

    /**
     * @brief Gets parsed inputs
     * 
     * @return inputs
     */
    const Inputs& getInputs() const { return inputs; }

    /**
     * @brief Gets request order
     */
    const Order getOrder() const { return order; }

    /**
     * @brief Gets request format
     */
    const Format getFormat() const { return format; }

    /*
    {
        "signature_name": "serving_default",
        "instances": [
            {...}, {...}, {...}, ...
        ]
    }
    */
    /**
     * @brief Parses http request body string
     * 
     * @param json request string
     * 
     * @return Status indicating error code or success
     * 
     * JSON expected to be passed in following structure:
     * {
     *     "signature_name": "serving_default",
     *     "instances": [
     *         {...}, {...}, {...}, ...
     *     ]
     * }
     */
    Status parse(const char* json) {
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
};


}  // namespace ovms
