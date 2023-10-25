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
#include "mediapipegraphexecutor.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../deserialization.hpp"
#include "../execution_context.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../modelmanager.hpp"
#include "../serialization.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "pythonnoderesource.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#include "mediapipegraphdefinition.hpp"  // for version in response
#include "opencv2/opencv.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wcomment"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#pragma GCC diagnostic pop

namespace ovms {

const ::mediapipe::Timestamp DEFAULT_STARTING_STREAM_TIMESTAMP = ::mediapipe::Timestamp(0);

static Status getRequestInput(google::protobuf::internal::RepeatedPtrIterator<const inference::ModelInferRequest_InferInputTensor>& itr, const std::string& requestedName, const KFSRequest& request) {
    auto requestInputItr = std::find_if(request.inputs().begin(), request.inputs().end(), [&requestedName](const ::KFSRequest::InferInputTensor& tensor) { return tensor.name() == requestedName; });
    if (requestInputItr == request.inputs().end()) {
        std::stringstream ss;
        ss << "Required input: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Missing input with specific name - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MISSING_INPUT, details);
    }
    itr = requestInputItr;
    return StatusCode::OK;
}

static mediapipe::Tensor::ElementType KFSPrecisionToMPPrecision(const KFSDataType& kfsDatatype) {
    static std::unordered_map<KFSDataType, mediapipe::Tensor::ElementType> precisionMap{
        //        {"FP64", mediapipe::Tensor::ElementType::},
        {"FP32", mediapipe::Tensor::ElementType::kFloat32},
        {"FP16", mediapipe::Tensor::ElementType::kFloat16},
        //        {"INT64", mediapipe::Tensor::ElementType::},
        {"INT32", mediapipe::Tensor::ElementType::kInt32},
        //        {"INT16", mediapipe::Tensor::ElementType::},
        {"INT8", mediapipe::Tensor::ElementType::kInt8},
        //        {"UINT64", mediapipe::Tensor::ElementType::},
        //        {"UINT32", mediapipe::Tensor::ElementType::},
        //        {"UINT16", mediapipe::Tensor::ElementType::},
        {"UINT8", mediapipe::Tensor::ElementType::kUInt8},
        {"BOOL", mediapipe::Tensor::ElementType::kBool}
        //        {"", ov::element::Type_t::, mediapipe::Tensor::ElementType::kChar}
    };
    auto it = precisionMap.find(kfsDatatype);
    if (it == precisionMap.end()) {
        return mediapipe::Tensor::ElementType::kNone;
    }
    return it->second;
}

const KFSDataType EMPTY_PREC = "";

static const KFSDataType& MPPrecisionToKFSPrecision(::mediapipe::Tensor::ElementType precision) {
    static std::unordered_map<mediapipe::Tensor::ElementType, KFSDataType> precisionMap{
        //        {mediapipe::Tensor::ElementType::, "FP64"},
        {mediapipe::Tensor::ElementType::kFloat32, "FP32"},
        {mediapipe::Tensor::ElementType::kFloat16, "FP16"},
        //        {mediapipe::Tensor::ElementType::, "INT64"},
        {mediapipe::Tensor::ElementType::kInt32, "INT32"},
        //        {mediapipe::Tensor::ElementType::, "INT16"},
        {mediapipe::Tensor::ElementType::kInt8, "INT8"},
        //        {mediapipe::Tensor::ElementType::, "UINT64"},
        //        {mediapipe::Tensor::ElementType::, "UINT32"},
        //        {mediapipe::Tensor::ElementType::, "UINT16"},
        {mediapipe::Tensor::ElementType::kUInt8, "UINT8"},
        {mediapipe::Tensor::ElementType::kBool, "BOOL"}
        //        {"", ov::element::Type_t::, mediapipe::Tensor::ElementType::kChar}
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        SPDLOG_WARN("Unsupported precision passed from Mediapipe graph");
        return EMPTY_PREC;
    }
    return it->second;
}

#define SET_DATA_FROM_MP_TENSOR(TENSOR, VIEW_TYPE)                                                     \
    switch ((TENSOR)->element_type()) {                                                                \
    case mediapipe::Tensor::ElementType::kFloat32:                                                     \
    case mediapipe::Tensor::ElementType::kFloat16:                                                     \
        data = reinterpret_cast<void*>(const_cast<float*>((TENSOR)->VIEW_TYPE().buffer<float>()));     \
        break;                                                                                         \
    case mediapipe::Tensor::ElementType::kUInt8:                                                       \
        data = reinterpret_cast<void*>(const_cast<uint8_t*>((TENSOR)->VIEW_TYPE().buffer<uint8_t>())); \
        break;                                                                                         \
    case mediapipe::Tensor::ElementType::kInt8:                                                        \
        data = reinterpret_cast<void*>(const_cast<int8_t*>((TENSOR)->VIEW_TYPE().buffer<int8_t>()));   \
        break;                                                                                         \
    case mediapipe::Tensor::ElementType::kInt32:                                                       \
        data = reinterpret_cast<void*>(const_cast<int32_t*>((TENSOR)->VIEW_TYPE().buffer<int32_t>())); \
        break;                                                                                         \
    case mediapipe::Tensor::ElementType::kBool:                                                        \
        data = reinterpret_cast<void*>(const_cast<bool*>((TENSOR)->VIEW_TYPE().buffer<bool>()));       \
        break;                                                                                         \
    default:                                                                                           \
        data = reinterpret_cast<void*>(const_cast<void*>((TENSOR)->VIEW_TYPE().buffer<void>()));       \
    }

#define HANDLE_DESERIALIZATION_EXCEPTION(TYPE_STRING)                                                       \
    catch (const std::exception& e) {                                                                       \
        std::stringstream ss;                                                                               \
        ss << "Exception:"                                                                                  \
           << e.what()                                                                                      \
           << "; caught during " TYPE_STRING " deserialization from KServe request tensor";                 \
        std::string details = ss.str();                                                                     \
        SPDLOG_DEBUG(details);                                                                              \
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details));                                       \
    }                                                                                                       \
    catch (...) {                                                                                           \
        std::stringstream ss;                                                                               \
        ss << "Unknown exception caught during " TYPE_STRING " deserialization from KServe request tensor"; \
        std::string details = ss.str();                                                                     \
        SPDLOG_DEBUG(details);                                                                              \
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details));                                       \
    }
static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<mediapipe::Tensor>& outTensor) {
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        return status;
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);
    try {
        auto datatype = KFSPrecisionToMPPrecision(requestInputItr->datatype());
        if (datatype == mediapipe::Tensor::ElementType::kNone) {
            std::stringstream ss;
            ss << "Not supported precision for Mediapipe tensor deserialization: " << requestInputItr->datatype();
            const std::string details = ss.str();
            SPDLOG_DEBUG(details);
            return Status(StatusCode::INVALID_PRECISION, std::move(details));
        }
        std::vector<int> rawShape;
        for (int i = 0; i < requestInputItr->shape().size(); i++) {
            if (requestInputItr->shape()[i] <= 0) {
                std::stringstream ss;
                ss << "Negative or zero dimension size is not acceptable: " << tensorShapeToString(requestInputItr->shape()) << "; input name: " << requestedName;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
            rawShape.emplace_back(requestInputItr->shape()[i]);
        }
        mediapipe::Tensor::Shape tensorShape{rawShape};
        outTensor = std::make_unique<mediapipe::Tensor>(datatype, tensorShape);
        void* data;
        SET_DATA_FROM_MP_TENSOR(outTensor, GetCpuWriteView);
        std::memcpy(data, bufferLocation.data(), bufferLocation.size());
    }
    HANDLE_DESERIALIZATION_EXCEPTION("Mediapipe tensor")
    return StatusCode::OK;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<tensorflow::Tensor>& outTensor) {
    using tensorflow::Tensor;
    using tensorflow::TensorShape;
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        return status;
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);
    try {
        auto datatype = getPrecisionAsDataType(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
        if (datatype == TFSDataType::DT_INVALID) {
            std::stringstream ss;
            ss << "Not supported precision for Tensorflow tensor deserialization: " << requestInputItr->datatype();
            const std::string details = ss.str();
            SPDLOG_DEBUG(details);
            return Status(StatusCode::INVALID_PRECISION, std::move(details));
        }
        TensorShape tensorShape;
        std::vector<int64_t> rawShape;
        for (int i = 0; i < requestInputItr->shape().size(); i++) {
            if (requestInputItr->shape()[i] < 0) {
                std::stringstream ss;
                ss << "Negative dimension size is not acceptable: " << tensorShapeToString(requestInputItr->shape()) << "; input name: " << requestedName;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
            rawShape.emplace_back(requestInputItr->shape()[i]);
        }
        int64_t dimsCount = rawShape.size();
        auto abslStatus = tensorflow::TensorShapeUtils::MakeShape(rawShape.data(), dimsCount, &tensorShape);
        if (!abslStatus.ok()) {
            auto stringViewAbslMessage = abslStatus.message();
            return Status(StatusCode::UNKNOWN_ERROR, std::string{stringViewAbslMessage});
        }
        abslStatus = TensorShape::BuildTensorShapeBase(rawShape, static_cast<tensorflow::TensorShapeBase<TensorShape>*>(&tensorShape));
        if (!abslStatus.ok()) {
            auto stringViewAbslMessage = abslStatus.message();
            return Status(StatusCode::UNKNOWN_ERROR, std::string{stringViewAbslMessage});
        }
        outTensor = std::make_unique<tensorflow::Tensor>(datatype, tensorShape);
        if (outTensor->TotalBytes() != bufferLocation.size()) {
            std::stringstream ss;
            ss << "Mediapipe deserialization content size mismatch; allocated TF Tensor: " << outTensor->TotalBytes() << " bytes vs KServe buffer: " << bufferLocation.size() << " bytes";
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
        void* tftensordata = outTensor->data();
        std::memcpy(tftensordata, bufferLocation.data(), bufferLocation.size());
    }
    HANDLE_DESERIALIZATION_EXCEPTION("Tensorflow tensor")
    return StatusCode::OK;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, TfLiteTensor** outTensor, const std::unique_ptr<tflite::Interpreter>& interpreter, size_t tfLiteTensorIdx) {
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        return status;
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);
    try {
        auto datatype = getPrecisionAsTfLiteDataType(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
        if (datatype == kTfLiteNoType) {
            std::stringstream ss;
            ss << "Not supported precision for Tensorflow Lite tensor deserialization: " << requestInputItr->datatype();
            const std::string details = ss.str();
            SPDLOG_DEBUG(details);
            return Status(StatusCode::INVALID_PRECISION, std::move(details));
        }
        std::vector<int> rawShape;
        for (int i = 0; i < requestInputItr->shape().size(); i++) {
            if (requestInputItr->shape()[i] < 0) {
                std::stringstream ss;
                ss << "Negative dimension size is not acceptable: " << tensorShapeToString(requestInputItr->shape()) << "; input name: " << requestedName;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
            rawShape.emplace_back(requestInputItr->shape()[i]);
        }
        interpreter->SetTensorParametersReadWrite(
            tfLiteTensorIdx,
            datatype,
            requestedName.c_str(),
            rawShape,
            TfLiteQuantization());
        // we shouldn't need to allocate tensors since we use buffer from request
        auto tensor = interpreter->tensor(tfLiteTensorIdx);
        tensor->data.data = reinterpret_cast<void*>(const_cast<char*>((bufferLocation.data())));
        *outTensor = tensor;
    }
    HANDLE_DESERIALIZATION_EXCEPTION("Tensorflow Lite tensor")
    return StatusCode::OK;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<ov::Tensor>& outTensor) {
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        return status;
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);
    try {
        ov::Shape shape;
        for (int i = 0; i < requestInputItr->shape().size(); i++) {
            if (requestInputItr->shape()[i] < 0) {
                std::stringstream ss;
                ss << "Negative dimension size is not acceptable: " << tensorShapeToString(requestInputItr->shape()) << "; input name: " << requestedName;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
            shape.push_back(requestInputItr->shape()[i]);
        }
        ov::element::Type precision = ovmsPrecisionToIE2Precision(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
        size_t expectElementsCount = ov::shape_size(shape.begin(), shape.end());
        size_t expectedBytes = precision.size() * expectElementsCount;
        if (expectedBytes != bufferLocation.size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedBytes << " bytes; Actual: " << bufferLocation.size() << " bytes; input name: " << requestedName;
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", request.model_name(), request.model_version(), details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
        if (expectedBytes == 0) {
            outTensor = std::make_unique<ov::Tensor>(precision, shape);  // OpenVINO does not accept nullptr as data ptr
        } else {
            outTensor = std::make_unique<ov::Tensor>(precision, shape, const_cast<void*>((const void*)bufferLocation.data()));
        }
    }
    HANDLE_DESERIALIZATION_EXCEPTION("OpenVINO tensor")
    return StatusCode::OK;
}

static mediapipe::ImageFormat::Format KFSDatatypeToImageFormat(const std::string& datatype, const size_t numberOfChannels) {
    if (datatype == "FP32") {
        if (numberOfChannels == 1) {
            return mediapipe::ImageFormat::VEC32F1;
        }
        if (numberOfChannels == 2) {
            return mediapipe::ImageFormat::VEC32F2;
        }
        // if(numberOfChannels == 4){
        //     return mediapipe::ImageFormat::VEC32F4;
        // }
    }
    if (datatype == "UINT8" || datatype == "INT8") {
        if (numberOfChannels == 1) {
            return mediapipe::ImageFormat::GRAY8;
        }
        if (numberOfChannels == 3) {
            return mediapipe::ImageFormat::SRGB;
        }
        if (numberOfChannels == 4) {
            return mediapipe::ImageFormat::SRGBA;
        }
    }
    if (datatype == "UINT16" || datatype == "INT16") {
        if (numberOfChannels == 1) {
            return mediapipe::ImageFormat::GRAY16;
        }
        if (numberOfChannels == 3) {
            return mediapipe::ImageFormat::SRGB48;
        }
        if (numberOfChannels == 4) {
            return mediapipe::ImageFormat::SRGBA64;
        }
    }
    if (datatype == "FP16") {
        if (numberOfChannels == 1) {
            return mediapipe::ImageFormat::GRAY16;
        }
        if (numberOfChannels == 3) {
            return mediapipe::ImageFormat::SRGB48;
        }
        if (numberOfChannels == 4) {
            return mediapipe::ImageFormat::SRGBA64;
        }
    }
    return mediapipe::ImageFormat::UNKNOWN;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<mediapipe::ImageFrame>& outTensor) {
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        SPDLOG_DEBUG("Getting Input failed");
        return status;
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);

    if (requestInputItr->shape().size() != 3) {
        std::stringstream ss;
        ss << "Invalid Mediapipe Image input shape size. Expected: 3; Actual: " << requestInputItr->shape().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG(details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    int64_t numberOfRows = requestInputItr->shape()[0];
    if (numberOfRows <= 0) {
        std::stringstream ss;
        ss << "Invalid Mediapipe Image input height. Expected greater than 0; Actual: " << numberOfRows << "; Expected layout - HWC.";
        const std::string details = ss.str();
        SPDLOG_DEBUG(details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    int64_t numberOfCols = requestInputItr->shape()[1];
    if (numberOfCols <= 0) {
        std::stringstream ss;
        ss << "Invalid Mediapipe Image input width. Expected greater than 0; Actual: " << numberOfCols << "; Expected layout - HWC.";
        const std::string details = ss.str();
        SPDLOG_DEBUG(details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    int64_t numberOfChannels = requestInputItr->shape()[2];
    if (numberOfChannels <= 0) {
        std::stringstream ss;
        ss << "Invalid Mediapipe Image input number of channels. Expected greater than 0; Actual: " << numberOfChannels << "; Expected layout - HWC.";
        const std::string details = ss.str();
        SPDLOG_DEBUG(details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }
    size_t elementSize = KFSDataTypeSize(requestInputItr->datatype());
    size_t expectedSize = numberOfChannels * numberOfCols * numberOfRows * elementSize;
    if (bufferLocation.size() != expectedSize) {
        std::stringstream ss;
        ss << "Invalid Mediapipe Image input buffer size. Actual: " << bufferLocation.size() << "Expected: " << expectedSize;
        const std::string details = ss.str();
        SPDLOG_DEBUG(details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }
    auto imageFormat = KFSDatatypeToImageFormat(requestInputItr->datatype(), numberOfChannels);
    if (imageFormat == mediapipe::ImageFormat::UNKNOWN) {
        SPDLOG_DEBUG("Invalid KFS request datatype, conversion to Mediapipe ImageFrame format failed.");
        return Status(StatusCode::INVALID_INPUT_FORMAT, "Invalid KFS request datatype, conversion to Mediapipe ImageFrame format failed.");
    }
    try {
        outTensor = std::make_unique<mediapipe::ImageFrame>(
            imageFormat,
            numberOfCols,
            numberOfRows,
            numberOfCols * numberOfChannels * elementSize,
            reinterpret_cast<uint8_t*>((const_cast<char*>(bufferLocation.data()))),
            mediapipe::ImageFrame::PixelDataDeleter::kNone);
    }
    HANDLE_DESERIALIZATION_EXCEPTION("Mediapipe ImageFrame")
    return StatusCode::OK;
}

MediapipeGraphExecutor::MediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
    stream_types_mapping_t inputTypes,
    stream_types_mapping_t outputTypes,
    std::vector<std::string> inputNames, std::vector<std::string> outputNames,
    const std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>>& pythonNodeResources) :
    name(name),
    version(version),
    config(config),
    inputTypes(std::move(inputTypes)),
    outputTypes(std::move(outputTypes)),
    inputNames(std::move(inputNames)),
    outputNames(std::move(outputNames)),
    pythonNodeResources(pythonNodeResources),
    currentStreamTimestamp(DEFAULT_STARTING_STREAM_TIMESTAMP) {}

namespace {
enum : unsigned int {
    INITIALIZE_GRAPH,
    RUN_GRAPH,
    ADD_INPUT_PACKET,
    FETCH_OUTPUT,
    ALL_FETCH,
    TOTAL,
    TIMER_END
};
}  // namespace

constexpr size_t STARTING_TIMESTAMP = 0;

static std::map<std::string, mediapipe::Packet> createInputSidePackets(const KFSRequest* request) {
    std::map<std::string, mediapipe::Packet> inputSidePackets;
    for (const auto& [name, valueChoice] : request->parameters()) {
        if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kStringParam) {
            inputSidePackets[name] = mediapipe::MakePacket<std::string>(valueChoice.string_param()).At(mediapipe::Timestamp(STARTING_TIMESTAMP));
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            inputSidePackets[name] = mediapipe::MakePacket<int64_t>(valueChoice.int64_param()).At(mediapipe::Timestamp(STARTING_TIMESTAMP));
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kBoolParam) {
            inputSidePackets[name] = mediapipe::MakePacket<bool>(valueChoice.bool_param()).At(mediapipe::Timestamp(STARTING_TIMESTAMP));
        } else {
            SPDLOG_DEBUG("Handling parameters of different types than: bool, string, int64 is not supported");
        }
    }
    return inputSidePackets;
}

template <typename T>
static Status createPacketAndPushIntoGraph(const std::string& name, const KFSRequest& request, ::mediapipe::CalculatorGraph& graph) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
    if (request.raw_input_contents().size() == 0) {
        const std::string details = "Invalid message structure - raw_input_content is empty";
        SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    if (request.raw_input_contents().size() != request.inputs().size()) {
        std::stringstream ss;
        ss << "Size of raw_input_contents: " << request.raw_input_contents().size() << " is different than number of inputs: " << request.inputs().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    std::unique_ptr<T> input_tensor;
    auto status = deserializeTensor(name, request, input_tensor);
    if (!status.ok()) {
        SPDLOG_DEBUG("Failed to deserialize tensor: {}", name);
        return status;
    }
    auto absStatus = graph.AddPacketToInputStream(
        name, ::mediapipe::Adopt<T>(input_tensor.release()).At(::mediapipe::Timestamp(STARTING_TIMESTAMP)));
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to add stream: {} packet to mediapipe graph: {} with error: {}",
            name, request.model_name(), absStatus.message(), absStatus.raw_code());
        return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM, std::move(absMessage));
    }
    absStatus = graph.CloseInputStream(name);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to close stream: {} of mediapipe graph: {} with error: {}",
            name, request.model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR, std::move(absMessage));
    }
    return StatusCode::OK;
}

static Status createPacketAndPushIntoGraphTfLiteTensor(const std::string& name, const KFSRequest& request, ::mediapipe::CalculatorGraph& graph, const std::unique_ptr<tflite::Interpreter>& interpreter, size_t tfLiteTensorIdx) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
    if (request.raw_input_contents().size() == 0) {
        const std::string details = "Invalid message structure - raw_input_content is empty";
        SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    if (request.raw_input_contents().size() != request.inputs().size()) {
        std::stringstream ss;
        ss << "Size of raw_input_contents: " << request.raw_input_contents().size() << " is different than number of inputs: " << request.inputs().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    TfLiteTensor* input_tensor = nullptr;
    auto status = deserializeTensor(name, request, &input_tensor, interpreter, tfLiteTensorIdx);
    if (!status.ok() || (!input_tensor)) {
        SPDLOG_DEBUG("Failed to deserialize tensor: {}", name);
        return status;
    }
    auto absStatus = graph.AddPacketToInputStream(  // potential need for fix when TfLite tensors enabled
        name, ::mediapipe::Adopt<TfLiteTensor>(input_tensor).At(::mediapipe::Timestamp(STARTING_TIMESTAMP)));
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to add stream: {} packet to mediapipe graph: {} with error: {}",
            name, request.model_name(), absStatus.message(), absStatus.raw_code());
        return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM, std::move(absMessage));
    }
    absStatus = graph.CloseInputStream(name);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to close stream: {} of mediapipe graph: {} with error: {}",
            name, request.model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR, std::move(absMessage));
    }
    return StatusCode::OK;
}

template <>
Status createPacketAndPushIntoGraph<KFSRequest*>(const std::string& name, const KFSRequest& request, ::mediapipe::CalculatorGraph& graph) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Request to passthrough:\"{}\"", name);
    auto absStatus = graph.AddPacketToInputStream(
        name, ::mediapipe::MakePacket<const KFSRequest*>(&request).At(::mediapipe::Timestamp(STARTING_TIMESTAMP)));
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to add stream: {} packet to mediapipe graph: {} with error: {}",
            name, request.model_name(), absStatus.message(), absStatus.raw_code());
        return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM, std::move(absMessage));
    }
    absStatus = graph.CloseInputStream(name);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to close stream: {} of mediapipe graph: {} with error: {}",
            name, request.model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR, std::move(absMessage));
    }
    return StatusCode::OK;
}

#define HANDLE_PACKET_RECEIVAL_EXCEPTIONS()                           \
    catch (const std::exception& e) {                                 \
        std::stringstream ss;                                         \
        ss << "Failed to get packet"                                  \
           << outputStreamName                                        \
           << " with exception: "                                     \
           << e.what();                                               \
        std::string details{ss.str()};                                \
        SPDLOG_DEBUG(details);                                        \
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details)); \
    }                                                                 \
    catch (...) {                                                     \
        std::stringstream ss;                                         \
        ss << "Failed to get packet"                                  \
           << outputStreamName                                        \
           << " with exception.";                                     \
        std::string details{ss.str()};                                \
        SPDLOG_DEBUG(details);                                        \
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details)); \
    }

template <typename T>
static Status receiveAndSerializePacket(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName);

template <>
Status receiveAndSerializePacket<tensorflow::Tensor>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<tensorflow::Tensor>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        output->set_datatype(
            ovmsPrecisionToKFSPrecision(
                TFSPrecisionToOvmsPrecision(
                    received.dtype())));
        output->clear_shape();
        for (const auto& dim : received.shape()) {
            output->add_shape(dim.size);
        }
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(received.data()), received.TotalBytes());
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

template <>
Status receiveAndSerializePacket<::mediapipe::Tensor>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        const ::mediapipe::Tensor& received = packet.Get<::mediapipe::Tensor>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        output->set_datatype(MPPrecisionToKFSPrecision(received.element_type()));
        output->clear_shape();
        for (const auto& dim : received.shape().dims) {
            output->add_shape(dim);
        }
        void* data;
        SET_DATA_FROM_MP_TENSOR(&received, GetCpuReadView);
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(data), received.bytes());
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

template <>
Status receiveAndSerializePacket<ov::Tensor>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<ov::Tensor>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        output->set_datatype(
            ovmsPrecisionToKFSPrecision(
                ovElementTypeToOvmsPrecision(
                    received.get_element_type())));
        output->clear_shape();
        for (const auto& dim : received.get_shape()) {
            output->add_shape(dim);
        }
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(received.data()), received.get_byte_size());
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

template <>
Status receiveAndSerializePacket<KFSResponse*>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<KFSResponse*>();
        if (received == nullptr) {
            std::stringstream ss;
            ss << "Received nullptr KFSResponse for: "
               << outputStreamName;
            std::string details{ss.str()};
            SPDLOG_DEBUG(details);
            return Status(StatusCode::UNKNOWN_ERROR, std::move(details));
        }
        response = std::move(*received);
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

static KFSDataType convertImageFormatToKFSDataType(const mediapipe::ImageFormat::Format& imageFormat) {
    static std::unordered_map<mediapipe::ImageFormat::Format, KFSDataType> ImageFormatKFSDatatypeMap{
        {mediapipe::ImageFormat::GRAY8, "UINT8"},
        {mediapipe::ImageFormat::SRGB, "UINT8"},
        {mediapipe::ImageFormat::SRGBA, "UINT8"},
        {mediapipe::ImageFormat::GRAY16, "UINT16"},
        {mediapipe::ImageFormat::SRGB48, "UINT16"},
        {mediapipe::ImageFormat::SRGBA64, "UINT16"},
        {mediapipe::ImageFormat::VEC32F1, "FP32"},
        {mediapipe::ImageFormat::VEC32F2, "FP32"}};
    auto it = ImageFormatKFSDatatypeMap.find(imageFormat);
    if (it == ImageFormatKFSDatatypeMap.end()) {
        SPDLOG_DEBUG("Converting Mediapipe::ImageFrame format to KFS datatype failed. Datatype will be set to default - UINT8");
        return "UINT8";
    }
    return it->second;
}

template <>
Status receiveAndSerializePacket<mediapipe::ImageFrame>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        const auto& received = packet.Get<mediapipe::ImageFrame>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        KFSDataType datatype = convertImageFormatToKFSDataType(received.Format());
        output->set_datatype(datatype);
        output->clear_shape();
        output->add_shape(received.Height());
        output->add_shape(received.Width());
        output->add_shape(received.NumberOfChannels());
        cv::Mat image = mediapipe::formats::MatView(&received);

        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(image.data), image.cols * image.rows * image.channels() * image.elemSize1());
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

Status MediapipeGraphExecutor::infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const {
    Timer<TIMER_END> timer;
    SPDLOG_DEBUG("Start KServe request mediapipe graph: {} execution", request->model_name());
    ::mediapipe::CalculatorGraph graph;
    auto absStatus = graph.Initialize(this->config);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("KServe request for mediapipe graph: {} initialization failed with message: {}", request->model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
    }

    std::unordered_map<std::string, ::mediapipe::OutputStreamPoller> outputPollers;
    for (auto& name : this->outputNames) {
        if (name.empty()) {
            SPDLOG_DEBUG("Creating Mediapipe graph outputs name failed for: {}", name);
            return StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR;
        }
        auto absStatusOrPoller = graph.AddOutputStreamPoller(name);
        if (!absStatusOrPoller.ok()) {
            const std::string absMessage = absStatusOrPoller.status().ToString();
            SPDLOG_DEBUG("Failed to add mediapipe graph output stream poller: {} with error: {}", request->model_name(), absMessage);
            return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR, std::move(absMessage));
        }
        outputPollers.emplace(name, std::move(absStatusOrPoller).value());
    }

    std::map<std::string, mediapipe::Packet> inputSidePackets{createInputSidePackets(request)};
    absStatus = graph.StartRun(inputSidePackets);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to start mediapipe graph: {} with error: {}", request->model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_START_ERROR, std::move(absMessage));
    }
    if (static_cast<int>(this->inputNames.size()) != request->inputs().size()) {
        std::stringstream ss;
        ss << "Expected: " << this->inputNames.size() << "; Actual: " << request->inputs().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of inputs - {}", request->model_name(), version, details);
        return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
    }

    ::mediapipe::Packet packet;
    std::set<std::string> outputPollersWithReceivedPacket;

    ovms::Status status;
    size_t tfLiteTensors = std::count_if(this->inputTypes.begin(), this->inputTypes.end(), [](const auto& pair) {
        return pair.second == mediapipe_packet_type_enum::TFLITETENSOR;
    });
    std::unique_ptr<tflite::Interpreter> interpreter = nullptr;
    if (tfLiteTensors) {
        interpreter = std::make_unique<tflite::Interpreter>();
        interpreter->AddTensors(tfLiteTensors);
    }
    size_t tfLiteTensorIdx = 0;
    size_t insertedStreamPackets = 0;
    for (auto& name : this->inputNames) {
        if (this->inputTypes.at(name) == mediapipe_packet_type_enum::KFS_REQUEST) {
            SPDLOG_DEBUG("Request processing KFS passthrough: {}", name);
            status = createPacketAndPushIntoGraph<KFSRequest*>(name, *request, graph);
        } else if (this->inputTypes.at(name) == mediapipe_packet_type_enum::TFTENSOR) {
            SPDLOG_DEBUG("Request processing TF tensor: {}", name);
            status = createPacketAndPushIntoGraph<tensorflow::Tensor>(name, *request, graph);
        } else if (this->inputTypes.at(name) == mediapipe_packet_type_enum::TFLITETENSOR) {
            SPDLOG_DEBUG("Request processing TfLite tensor: {}", name);
            status = createPacketAndPushIntoGraphTfLiteTensor(name, *request, graph, interpreter, tfLiteTensorIdx);
            ++tfLiteTensorIdx;
        } else if (this->inputTypes.at(name) == mediapipe_packet_type_enum::MPTENSOR) {
            SPDLOG_DEBUG("Request processing MP tensor: {}", name);
            status = createPacketAndPushIntoGraph<mediapipe::Tensor>(name, *request, graph);
        } else if (this->inputTypes.at(name) == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
            SPDLOG_DEBUG("Request processing Mediapipe ImageFrame: {}", name);
            status = createPacketAndPushIntoGraph<mediapipe::ImageFrame>(name, *request, graph);
        } else if ((this->inputTypes.at(name) == mediapipe_packet_type_enum::OVTENSOR) ||
                   (this->inputTypes.at(name) == mediapipe_packet_type_enum::UNKNOWN)) {
            SPDLOG_DEBUG("Request processing OVTensor: {}", name);
            status = createPacketAndPushIntoGraph<ov::Tensor>(name, *request, graph);
        }
        if (!status.ok()) {
            return status;
        }
        ++insertedStreamPackets;
    }
    if (this->inputNames.size() > insertedStreamPackets) {
        SPDLOG_DEBUG("Not all input packets created. Expected: {}, Actual: {}. Aborting execution of mediapipe graph: {}",
            this->inputNames.size(),
            insertedStreamPackets,
            this->name);
        return Status(StatusCode::INTERNAL_ERROR, "Not all input packets created");
    }
    // receive outputs
    for (auto& [outputStreamName, poller] : outputPollers) {
        size_t receivedOutputs = 0;
        SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
        while (poller.Next(&packet)) {
            SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
            if (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::KFS_RESPONSE) {
                SPDLOG_DEBUG("Response processing packet type KFSPass name: {}", outputStreamName);
                status = receiveAndSerializePacket<KFSResponse*>(packet, *response, outputStreamName);
            } else if (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::TFTENSOR) {
                SPDLOG_DEBUG("Response processing packet type TF Tensor name: {}", outputStreamName);
                status = receiveAndSerializePacket<tensorflow::Tensor>(packet, *response, outputStreamName);
            } else if (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::TFLITETENSOR) {
                SPDLOG_DEBUG("Response processing packet type TFLite Tensor name: {}", outputStreamName);
                std::string details{"Response processing packet type TFLite Tensor is not supported"};
                return Status(StatusCode::NOT_IMPLEMENTED, std::move(details));
            } else if (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::MPTENSOR) {
                SPDLOG_DEBUG("Response processing packet type MP Tensor name: {}", outputStreamName);
                status = receiveAndSerializePacket<mediapipe::Tensor>(packet, *response, outputStreamName);
            } else if (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
                SPDLOG_DEBUG("Response processing Mediapipe Image Frame: {}", outputStreamName);
                status = receiveAndSerializePacket<mediapipe::ImageFrame>(packet, *response, outputStreamName);
            } else if ((this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::OVTENSOR) ||
                       (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::UNKNOWN)) {
                SPDLOG_DEBUG("Response processing packet type:  OVTensor name: {}", outputStreamName);
                status = receiveAndSerializePacket<ov::Tensor>(packet, *response, outputStreamName);
            } else {
                return Status(StatusCode::UNKNOWN_ERROR, "Unreachable code");
            }
            if (!status.ok()) {
                return status;
            }
            outputPollersWithReceivedPacket.insert(outputStreamName);
            ++receivedOutputs;
        }
        SPDLOG_TRACE("Received all: {} packets for: {}", receivedOutputs, outputStreamName);
    }
    absStatus = graph.WaitUntilDone();
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Mediapipe failed to execute: {}", absMessage);
        return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, absMessage);
    }
    if (outputPollers.size() != outputPollersWithReceivedPacket.size()) {
        SPDLOG_DEBUG("Mediapipe failed to execute. Failed to receive all output packets");
        return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unknown error during mediapipe execution");
    }
    SPDLOG_DEBUG("Received all output stream packets for graph: {}", request->model_name());
    response->set_model_name(request->model_name());
    response->set_id(request->id());
    response->set_model_version(request->model_version());
    return StatusCode::OK;
}

#define MP_RETURN_ON_FAIL(code, message, errorCode)              \
    {                                                            \
        auto absStatus = code;                                   \
        if (!absStatus.ok()) {                                   \
            const std::string absMessage = absStatus.ToString(); \
            SPDLOG_DEBUG("{} {}", message, absMessage);          \
            return Status(errorCode, std::move(absMessage));     \
        }                                                        \
    }

#define OVMS_RETURN_ON_FAIL(code, message)                   \
    {                                                        \
        auto status = code;                                  \
        if (!status.ok()) {                                  \
            SPDLOG_DEBUG("{} {}", message, status.string()); \
            return status;                                   \
        }                                                    \
    }

#define OVMS_RETURN_MP_ERROR_ON_FAIL(code, message)                     \
    {                                                                   \
        auto status = code;                                             \
        if (!status.ok()) {                                             \
            SPDLOG_DEBUG("{} {}", message, status.string());            \
            return absl::Status(absl::StatusCode::kCancelled, message); \
        }                                                               \
    }

#define OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(code, message)          \
    {                                                                 \
        auto status = code;                                           \
        if (!status.ok()) {                                           \
            ::inference::ModelStreamInferResponse resp;               \
            std::stringstream ss;                                     \
            ss << status.string() << "; " << message;                 \
            *resp.mutable_error_message() = ss.str();                 \
            if (!stream.Write(resp)) {                                \
                SPDLOG_DEBUG("Writing error to disconnected client"); \
            }                                                         \
        }                                                             \
    }

template <typename T>
class HolderWithRequestOwnership : public ::mediapipe::packet_internal::Holder<T> {
    std::shared_ptr<const ::inference::ModelInferRequest> req;

public:
    explicit HolderWithRequestOwnership(const T* ptr, const std::shared_ptr<const ::inference::ModelInferRequest>& req) :
        ::mediapipe::packet_internal::Holder<T>(ptr),
        req(req) {}
};

// TODO: Add support for other types CVS-122328/CVS-122329
Status MediapipeGraphExecutor::partialDeserialize(std::shared_ptr<const ::inference::ModelInferRequest> request, ::mediapipe::CalculatorGraph& graph) {
    // Deserialize optional manual timestamp
    if (!request->id().empty()) {
        auto requestTimestamp = stoi64(request->id());  // TODO: Decide if deserialize from id or int parameter
        if (requestTimestamp.has_value()) {
            // Cannot create with error checking since error check = abseil death test
            currentStreamTimestamp = ::mediapipe::Timestamp::CreateNoErrorChecking(requestTimestamp.value());
        } else {
            auto status = Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, "Invalid timestamp format in request id field");
            SPDLOG_DEBUG(status.string());
            return status;
        }
    }

    // Validate current timestamp (can be manual or automatic at this point)
    if (!currentStreamTimestamp.IsRangeValue()) {
        SPDLOG_DEBUG("Timestamp not in range: {}", currentStreamTimestamp.DebugString());
        return Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, currentStreamTimestamp.DebugString());
    }

    // Deserialize each input separately
    for (const auto& input : request->inputs()) {
        std::unique_ptr<ov::Tensor> tensor;
        OVMS_RETURN_ON_FAIL(deserializeTensor(input.name(), *request, tensor), "ov::Tensor deserialization");
        MP_RETURN_ON_FAIL(graph.AddPacketToInputStream(
                              input.name(),
                              ::mediapipe::packet_internal::Create(
                                  new HolderWithRequestOwnership<ov::Tensor>(
                                      tensor.release(),
                                      request))
                                  .At(currentStreamTimestamp)),
            "adding packet to input stream", StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
    }

    // Increment timestamp automatically for next request
    currentStreamTimestamp = currentStreamTimestamp.NextAllowedInStream();
    return StatusCode::OK;
}

Status MediapipeGraphExecutor::inferStream(const ::inference::ModelInferRequest& firstRequest, ::grpc::ServerReaderWriterInterface<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest>& stream) {
    try {
        // Init
        ::mediapipe::CalculatorGraph graph;
        MP_RETURN_ON_FAIL(graph.Initialize(this->config), "graph initialization", StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);

        // Installing observers
        for (const auto& name : this->outputNames) {
            MP_RETURN_ON_FAIL(graph.ObserveOutputStream(name, [&stream, &name](const ::mediapipe::Packet& packet) -> absl::Status {
                try {
                    ::inference::ModelStreamInferResponse resp;
                    // TODO: Add support for other types CVS-122327/CVS-122329
                    OVMS_RETURN_MP_ERROR_ON_FAIL(receiveAndSerializePacket<ov::Tensor>(packet, *resp.mutable_infer_response(), name), "ov::Tensor serialization");  // TODO: Missing test
                    resp.mutable_infer_response()->set_id(std::to_string(packet.Timestamp().Value()));
                    if (!stream.Write(resp)) {
                        return absl::Status(absl::StatusCode::kCancelled, "client disconnected");
                    }
                    return absl::OkStatus();
                } catch (...) {
                    return absl::Status(absl::StatusCode::kCancelled, "error in serialization");
                }
            }),
                "output stream observer installation", StatusCode::INTERNAL_ERROR);  // Should never happen for validated graphs
        }

        // Launch
        MP_RETURN_ON_FAIL(graph.StartRun({}), "graph start", StatusCode::MEDIAPIPE_GRAPH_START_ERROR);  // TODO: Input side packets / missing test

        // Deserialize first request
        OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(this->partialDeserialize(
                                                  std::shared_ptr<const ::inference::ModelInferRequest>(&firstRequest,
                                                      // Custom deleter to avoid deallocation by custom holder
                                                      // Conversion to shared_ptr is required for unified deserialization method
                                                      // for first and subsequent requests
                                                      [](const ::inference::ModelInferRequest*) {}),
                                                  graph),
            "partial deserialization of first request");

        // Read loop
        // Here we create ModelInferRequest with shared ownership,
        // and move it down to custom packet holder to ensure
        // lifetime is extended to lifetime of deserialized Packets.
        auto req = std::make_shared<::inference::ModelInferRequest>();
        while (stream.Read(req.get())) {
            OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(this->partialDeserialize(req, graph), "partial deserialization of subsequent requests");
            if (graph.HasError()) {
                SPDLOG_DEBUG("Graph encountered an error, stopping the execution");
                break;
            }
            req = std::make_shared<::inference::ModelInferRequest>();
        }

        // Close input streams
        MP_RETURN_ON_FAIL(graph.CloseAllPacketSources(), "closing all packet sources", StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR);

        // Wait until done
        MP_RETURN_ON_FAIL(graph.WaitUntilDone(), "waiting until done", StatusCode::MEDIAPIPE_EXECUTION_ERROR);

        return StatusCode::OK;
    } catch (...) {
        return Status(StatusCode::UNKNOWN_ERROR, "Exception while processing MediaPipe graph");  // To be displayed in method level above
    }
}

}  // namespace ovms
