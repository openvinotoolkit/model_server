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

#include <functional>
#include <iostream>
#include <limits>
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
#include "../predict_request_validation_utils.hpp"
#include "../serialization.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "../timer.hpp"
#include "../version.hpp"
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
#include "opencv2/opencv.hpp"

#if (PYTHON_DISABLE == 0)
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../python/ovms_py_tensor.hpp"
#include "../python/python_backend.hpp"
#include "../python/pythonnoderesources.hpp"
namespace py = pybind11;
#endif

namespace ovms {
using namespace request_validation_utils;
using ::mediapipe::Timestamp;
const Timestamp DEFAULT_STARTING_STREAM_TIMESTAMP = Timestamp(0);

#define MP_RETURN_ON_FAIL(code, message, errorCode)              \
    {                                                            \
        auto absStatus = code;                                   \
        if (!absStatus.ok()) {                                   \
            const std::string absMessage = absStatus.ToString(); \
            SPDLOG_DEBUG("{} {}", message, absMessage);          \
            return Status(errorCode, std::move(absMessage));     \
        }                                                        \
    }

#define OVMS_RETURN_ON_FAIL(code) \
    {                             \
        auto status = code;       \
        if (!status.ok()) {       \
            return status;        \
        }                         \
    }

#define OVMS_RETURN_MP_ERROR_ON_FAIL(code, message)                     \
    {                                                                   \
        auto status = code;                                             \
        if (!status.ok()) {                                             \
            SPDLOG_DEBUG("{} {}", message, status.string());            \
            return absl::Status(absl::StatusCode::kCancelled, message); \
        }                                                               \
    }

#define OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(code, message)                 \
    {                                                                        \
        auto status = code;                                                  \
        if (!status.ok()) {                                                  \
            ::inference::ModelStreamInferResponse resp;                      \
            std::stringstream ss;                                            \
            ss << status.string() << "; " << message;                        \
            *resp.mutable_error_message() = ss.str();                        \
            if (!streamSynchronizedWrite(stream, streamWriterMutex, resp)) { \
                SPDLOG_DEBUG("Writing error to disconnected client");        \
            }                                                                \
        }                                                                    \
    }

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
static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<mediapipe::Tensor>& outTensor, PythonBackend* pythonBackend) {
    auto requestInputItr = request.inputs().begin();
    OVMS_RETURN_ON_FAIL(getRequestInput(requestInputItr, requestedName, request));
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

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<tensorflow::Tensor>& outTensor, PythonBackend* pythonBackend) {
    using tensorflow::Tensor;
    using tensorflow::TensorShape;
    auto requestInputItr = request.inputs().begin();
    OVMS_RETURN_ON_FAIL(getRequestInput(requestInputItr, requestedName, request));
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

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<ov::Tensor>& outTensor, PythonBackend* pythonBackend) {
    auto requestInputItr = request.inputs().begin();
    OVMS_RETURN_ON_FAIL(getRequestInput(requestInputItr, requestedName, request));
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
        if (numberOfChannels == 4) {
            return mediapipe::ImageFormat::VEC32F4;
        }
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

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<mediapipe::ImageFrame>& outTensor, PythonBackend* pythonBackend) {
    auto requestInputItr = request.inputs().begin();
    OVMS_RETURN_ON_FAIL(getRequestInput(requestInputItr, requestedName, request));
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
        ss << "Invalid Mediapipe Image input buffer size. Actual: " << bufferLocation.size() << "; Expected: " << expectedSize;
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

#if (PYTHON_DISABLE == 0)
static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<PyObjectWrapper<py::object>>& outTensor, PythonBackend* pythonBackend) {
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        return status;
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);
    try {
        std::vector<py::ssize_t> shape;
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

        auto formatIt = datatypeToBufferFormat.find(requestInputItr->datatype());
        if (formatIt != datatypeToBufferFormat.end()) {
            // If datatype is known, we check if a valid buffer can be created with provided data
            size_t itemsize = bufferFormatToItemsize.at(formatIt->second);
            size_t expectedBufferSize = 1;

            bool expectedBufferSizeValid = computeExpectedBufferSizeReturnFalseIfOverflow<py::ssize_t>(shape, itemsize, expectedBufferSize);
            if (!expectedBufferSizeValid) {
                const std::string details = "Provided shape and datatype declare too large buffer.";
                SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_CONTENT_SIZE, details);
            }

            if (bufferLocation.size() != expectedBufferSize) {
                std::stringstream ss;
                ss << "Invalid Python tensor buffer size. Actual: " << bufferLocation.size() << "; Expected: " << expectedBufferSize;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_CONTENT_SIZE, details);
            }
        }

        auto ok = pythonBackend->createOvmsPyTensor(
            requestedName,
            const_cast<void*>((const void*)bufferLocation.data()),
            shape,
            requestInputItr->datatype(),
            bufferLocation.size(),
            outTensor);

        if (!ok) {
            SPDLOG_DEBUG("Error creating Python tensor from data");
            return StatusCode::UNKNOWN_ERROR;
        }
    }
    HANDLE_DESERIALIZATION_EXCEPTION("Ovms Python tensor")
    return StatusCode::OK;
}
#endif

MediapipeGraphExecutor::MediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
    stream_types_mapping_t inputTypes,
    stream_types_mapping_t outputTypes,
    std::vector<std::string> inputNames, std::vector<std::string> outputNames,
    const PythonNodeResourcesMap& pythonNodeResourcesMap,
    PythonBackend* pythonBackend) :
    name(name),
    version(version),
    config(config),
    inputTypes(std::move(inputTypes)),
    outputTypes(std::move(outputTypes)),
    inputNames(std::move(inputNames)),
    outputNames(std::move(outputNames)),
    pythonNodeResourcesMap(pythonNodeResourcesMap),
    pythonBackend(pythonBackend),
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
const std::string MediapipeGraphExecutor::TIMESTAMP_PARAMETER_NAME = "OVMS_MP_TIMESTAMP";

const std::string PYTHON_SESSION_SIDE_PACKET_TAG = "py";

static std::map<std::string, mediapipe::Packet> createInputSidePackets(const KFSRequest* request) {
    std::map<std::string, mediapipe::Packet> inputSidePackets;
    for (const auto& [name, valueChoice] : request->parameters()) {
        SPDLOG_DEBUG("Found: {}; parameter in request for: {};", name, request->model_name());
        if (name == MediapipeGraphExecutor::TIMESTAMP_PARAMETER_NAME) {
            SPDLOG_DEBUG("Ignored: {}; parameter in request for: {}; Paremeter is reserved for MediaPipe input packet timestamps", name, request->model_name());
            continue;
        }
        if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kStringParam) {
            inputSidePackets[name] = mediapipe::MakePacket<std::string>(valueChoice.string_param());
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            inputSidePackets[name] = mediapipe::MakePacket<int64_t>(valueChoice.int64_param());
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kBoolParam) {
            inputSidePackets[name] = mediapipe::MakePacket<bool>(valueChoice.bool_param());
        } else {
            SPDLOG_DEBUG("Handling parameters of different types than: bool, string, int64 is not supported");
        }
    }
    return inputSidePackets;
}

template <typename T, template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& name, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const Timestamp& timestamp, PythonBackend* pythonBackend) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
    if (request->raw_input_contents().size() == 0) {
        const std::string details = "Invalid message structure - raw_input_content is empty";
        SPDLOG_DEBUG("[servable name: {} version: {}] {}", request->model_name(), request->model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    if (request->raw_input_contents().size() != request->inputs().size()) {
        std::stringstream ss;
        ss << "Size of raw_input_contents: " << request->raw_input_contents().size() << " is different than number of inputs: " << request->inputs().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request->model_name(), request->model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    std::unique_ptr<T> inputTensor;
    OVMS_RETURN_ON_FAIL(deserializeTensor(name, *request, inputTensor, pythonBackend));
    MP_RETURN_ON_FAIL(graph.AddPacketToInputStream(
                          name,
                          ::mediapipe::packet_internal::Create(
                              new Holder<T>(inputTensor.release(), request))
                              .At(timestamp)),
        std::string("failed to add packet to stream: ") + name, StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
    return StatusCode::OK;
}

template <template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& name, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const Timestamp& timestamp, PythonBackend* pythonBackend) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Request to passthrough:\"{}\"", name);
    const KFSRequest* lvaluePtr = request.get();
    MP_RETURN_ON_FAIL(graph.AddPacketToInputStream(
                          name,
                          ::mediapipe::packet_internal::Create(
                              new Holder<const KFSRequest*>(lvaluePtr, request))
                              .At(timestamp)),
        std::string("failed to add packet to stream: ") + name, StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
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

#if (PYTHON_DISABLE == 0)
template <>
Status receiveAndSerializePacket<PyObjectWrapper<py::object>>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        const PyObjectWrapper<py::object>& pyOutput = packet.Get<PyObjectWrapper<py::object>>();
        auto* output = response.add_outputs();
        output->set_name(pyOutput.getProperty<std::string>("name"));
        output->set_datatype(pyOutput.getProperty<std::string>("datatype"));
        output->clear_shape();
        for (const auto& dim : pyOutput.getProperty<std::vector<py::ssize_t>>("shape")) {
            output->add_shape(dim);
        }
        void* ptr = pyOutput.getProperty<void*>("ptr");
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(ptr), pyOutput.getProperty<py::ssize_t>("size"));
        return StatusCode::OK;
    } catch (const pybind11::error_already_set& e) {
        std::stringstream ss;
        ss << "Failed to get packet " << outputStreamName << " due to Python object unpacking error: " << e.what();
        std::string details{ss.str()};
        SPDLOG_DEBUG(details);
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details));
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}
#endif

// Two types of holders
// One (HolderWithRequestOwnership) is required for streaming where it is OVMS who creates the request but it is not the packet type and we have to clean up
// Second (HolderWithNoRequestOwnership) is required for unary-unary where it is gRPC who creates the request and musn't clean up
// Specializations are for special case when the request itsef is the packet and we need to ensure there is no double free
template <typename T>
class HolderWithRequestOwnership : public ::mediapipe::packet_internal::Holder<T> {
    std::shared_ptr<const KFSRequest> req;

public:
    explicit HolderWithRequestOwnership(const T* barePtr, const std::shared_ptr<const KFSRequest>& req) :
        ::mediapipe::packet_internal::Holder<T>(barePtr),
        req(req) {}
};
template <>
class HolderWithRequestOwnership<const KFSRequest*> : public ::mediapipe::packet_internal::ForeignHolder<const KFSRequest*> {
    const KFSRequest* hiddenPtr = nullptr;
    std::shared_ptr<const KFSRequest> req;

public:
    explicit HolderWithRequestOwnership(const KFSRequest* barePtr, const std::shared_ptr<const KFSRequest>& req) :
        ::mediapipe::packet_internal::ForeignHolder<const KFSRequest*>(&hiddenPtr),
        hiddenPtr(barePtr),
        req(req) {}
};

template <typename T>
class HolderWithNoRequestOwnership : public ::mediapipe::packet_internal::Holder<T> {
public:
    explicit HolderWithNoRequestOwnership(const T* barePtr, const std::shared_ptr<const KFSRequest>& req) :
        ::mediapipe::packet_internal::Holder<T>(barePtr) {}
};
template <>
class HolderWithNoRequestOwnership<const KFSRequest*> : public ::mediapipe::packet_internal::ForeignHolder<const KFSRequest*> {
public:
    const KFSRequest* hiddenPtr = nullptr;
    explicit HolderWithNoRequestOwnership(const KFSRequest* barePtr, const std::shared_ptr<const KFSRequest>& req) :
        ::mediapipe::packet_internal::ForeignHolder<const KFSRequest*>(&hiddenPtr),
        hiddenPtr(barePtr) {}
};

template <template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& inputName, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const Timestamp& timestamp, const stream_types_mapping_t& inputTypes, PythonBackend* pythonBackend) {
    auto inputPacketType = inputTypes.at(inputName);
    ovms::Status status;
    if (inputPacketType == mediapipe_packet_type_enum::KFS_REQUEST) {
        SPDLOG_DEBUG("Request processing KFS passthrough: {}", inputName);
        status = createPacketAndPushIntoGraph<Holder>(inputName, request, graph, timestamp, nullptr);
    } else if (inputPacketType == mediapipe_packet_type_enum::TFTENSOR) {
        SPDLOG_DEBUG("Request processing TF tensor: {}", inputName);
        status = createPacketAndPushIntoGraph<tensorflow::Tensor, Holder>(inputName, request, graph, timestamp, nullptr);
    } else if (inputPacketType == mediapipe_packet_type_enum::MPTENSOR) {
        SPDLOG_DEBUG("Request processing MP tensor: {}", inputName);
        status = createPacketAndPushIntoGraph<mediapipe::Tensor, Holder>(inputName, request, graph, timestamp, nullptr);
    } else if (inputPacketType == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
        SPDLOG_DEBUG("Request processing Mediapipe ImageFrame: {}", inputName);
        status = createPacketAndPushIntoGraph<mediapipe::ImageFrame, Holder>(inputName, request, graph, timestamp, nullptr);
#if (PYTHON_DISABLE == 0)
    } else if (inputPacketType == mediapipe_packet_type_enum::OVMS_PY_TENSOR) {
        SPDLOG_DEBUG("Request processing OVMS Python input: {}", inputName);
        status = createPacketAndPushIntoGraph<PyObjectWrapper<py::object>, Holder>(inputName, request, graph, timestamp, pythonBackend);
#endif
    } else if ((inputPacketType == mediapipe_packet_type_enum::OVTENSOR) ||
               (inputPacketType == mediapipe_packet_type_enum::UNKNOWN)) {
        SPDLOG_DEBUG("Request processing OVTensor: {}", inputName);
        status = createPacketAndPushIntoGraph<ov::Tensor, Holder>(inputName, request, graph, timestamp, nullptr);
    }
    return status;
}

Status MediapipeGraphExecutor::infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const {
    Timer<TIMER_END> timer;
    SPDLOG_DEBUG("Start unary KServe request mediapipe graph: {} execution", request->model_name());
    ::mediapipe::CalculatorGraph graph;
    MP_RETURN_ON_FAIL(graph.Initialize(this->config), std::string("failed initialization of MediaPipe graph: ") + request->model_name(), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
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
    std::map<std::string, mediapipe::Packet> sideInputPackets{createInputSidePackets(request)};
#if (PYTHON_DISABLE == 0)
    if (sideInputPackets.count(PYTHON_SESSION_SIDE_PACKET_TAG)) {
        const std::string absMessage = "Incoming input side packet: " + PYTHON_SESSION_SIDE_PACKET_TAG + " is special reserved name and cannot be used";
        SPDLOG_DEBUG("Failed to insert predefined input side packet: {} with error: {}", PYTHON_SESSION_SIDE_PACKET_TAG, absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
    }
    sideInputPackets[PYTHON_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<PythonNodeResourcesMap>(this->pythonNodeResourcesMap).At(mediapipe::Timestamp(STARTING_TIMESTAMP));
#endif
    MP_RETURN_ON_FAIL(graph.StartRun(sideInputPackets), std::string("start MediaPipe graph: ") + request->model_name(), StatusCode::MEDIAPIPE_GRAPH_START_ERROR);
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
    size_t insertedStreamPackets = 0;
    std::shared_ptr<const KFSRequest> requestWithNoOwnership(request, [](const KFSRequest* r) {});
    for (auto& inputName : this->inputNames) {
        OVMS_RETURN_ON_FAIL(createPacketAndPushIntoGraph<HolderWithNoRequestOwnership>(inputName, requestWithNoOwnership, graph, this->currentStreamTimestamp, this->inputTypes, pythonBackend));
        ++insertedStreamPackets;
    }
    if (this->inputNames.size() > insertedStreamPackets) {
        SPDLOG_DEBUG("Not all input packets created. Expected: {}, Actual: {}. Aborting execution of mediapipe graph: {}",
            this->inputNames.size(),
            insertedStreamPackets,
            this->name);
        return Status(StatusCode::INTERNAL_ERROR, "Not all input packets created");
    }
    // we wait idle since some calculators could hold ownership on packet content while nodes further down the graph
    // can be still processing those. Closing packet sources triggers Calculator::Close() on nodes that do not expect
    // new packets
    MP_RETURN_ON_FAIL(graph.WaitUntilIdle(), "graph wait until idle", StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    MP_RETURN_ON_FAIL(graph.CloseAllPacketSources(), "graph close all packet sources", StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR);
    for (auto& [outputStreamName, poller] : outputPollers) {
        size_t receivedOutputs = 0;
        SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
        if (poller.Next(&packet)) {
            SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
            OVMS_RETURN_ON_FAIL(serializePacket(outputStreamName, *response, packet));
            outputPollersWithReceivedPacket.insert(outputStreamName);
            ++receivedOutputs;
        }
        SPDLOG_TRACE("Received all: {} packets for: {}", receivedOutputs, outputStreamName);
    }
    MP_RETURN_ON_FAIL(graph.WaitUntilDone(), "grap wait until done", StatusCode::MEDIAPIPE_EXECUTION_ERROR);
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

Status MediapipeGraphExecutor::deserializeTimestampIfAvailable(const KFSRequest& request, Timestamp& timestamp) {
    auto timestampParamIt = request.parameters().find(TIMESTAMP_PARAMETER_NAME);
    if (timestampParamIt != request.parameters().end()) {
        SPDLOG_DEBUG("Found {} timestamp parameter in request for: {}", TIMESTAMP_PARAMETER_NAME, request.model_name());
        auto& parameterChoice = timestampParamIt->second;
        if (parameterChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            // Cannot create with error checking since error check = abseil death test
            timestamp = Timestamp::CreateNoErrorChecking(parameterChoice.int64_param());
        } else {
            auto status = Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, "Invalid timestamp format in request parameter OVMS_MP_TIMESTAMP. Should be int64");
            SPDLOG_DEBUG(status.string());
            return status;
        }
    }
    return StatusCode::OK;
}

static inline Status checkTimestamp(const KFSRequest& request, const Timestamp& timestamp) {
    if (!timestamp.IsRangeValue()) {
        SPDLOG_DEBUG("Timestamp not in range: {}; for request to: {};", timestamp.DebugString(), request.model_name());
        return Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, timestamp.DebugString());
    }
    return StatusCode::OK;
}

Status MediapipeGraphExecutor::partialDeserialize(std::shared_ptr<const KFSRequest> request, ::mediapipe::CalculatorGraph& graph) {
    OVMS_RETURN_ON_FAIL(deserializeTimestampIfAvailable(*request, this->currentStreamTimestamp));
    OVMS_RETURN_ON_FAIL(checkTimestamp(*request, this->currentStreamTimestamp));
    for (const auto& input : request->inputs()) {
        const auto& inputName = input.name();
        if (std::find_if(this->inputNames.begin(), this->inputNames.end(), [&inputName](auto streamName) { return streamName == inputName; }) == this->inputNames.end()) {
            SPDLOG_DEBUG("Request for {}, contains not expected input name: {}", request->model_name(), inputName);
            return Status(StatusCode::INVALID_UNEXPECTED_INPUT, std::string(inputName) + " is unexpected");
        }
        OVMS_RETURN_ON_FAIL(createPacketAndPushIntoGraph<HolderWithRequestOwnership>(inputName, request, graph, this->currentStreamTimestamp, this->inputTypes, pythonBackend));
    }
    currentStreamTimestamp = currentStreamTimestamp.NextAllowedInStream();
    return StatusCode::OK;
}

Status MediapipeGraphExecutor::validateSubsequentRequest(const ::inference::ModelInferRequest& request) const {
    if (request.model_name() != this->name) {
        return StatusCode::MEDIAPIPE_INCORRECT_SERVABLE_NAME;
    }
    if (request.model_version() != this->version &&
        request.model_version() != "0" &&    // default version does not matter for user
        !request.model_version().empty()) {  // empty the same as default
        return StatusCode::MEDIAPIPE_INCORRECT_SERVABLE_VERSION;
    }
    return StatusCode::OK;
}

static bool streamSynchronizedWrite(::grpc::ServerReaderWriterInterface<::inference::ModelStreamInferResponse, KFSRequest>& stream,
    std::mutex& mtx, ::inference::ModelStreamInferResponse& resp) {
    const std::lock_guard<std::mutex> lock(mtx);
    return stream.Write(resp);
}

Status MediapipeGraphExecutor::inferStream(const KFSRequest& firstRequest, ::grpc::ServerReaderWriterInterface<::inference::ModelStreamInferResponse, KFSRequest>& stream) {
    SPDLOG_DEBUG("Start streaming KServe request mediapipe graph: {} execution", this->name);
    std::mutex streamWriterMutex;
    try {
        // Init
        ::mediapipe::CalculatorGraph graph;
        MP_RETURN_ON_FAIL(graph.Initialize(this->config), "graph initialization", StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);

        // Installing observers
        for (const auto& outputName : this->outputNames) {
            MP_RETURN_ON_FAIL(graph.ObserveOutputStream(outputName, [&stream, &streamWriterMutex, &outputName, this](const ::mediapipe::Packet& packet) -> absl::Status {
                try {
                    ::inference::ModelStreamInferResponse resp;
                    OVMS_RETURN_MP_ERROR_ON_FAIL(serializePacket(outputName, *resp.mutable_infer_response(), packet), "error in serialization");
                    *resp.mutable_infer_response()->mutable_model_name() = this->name;
                    *resp.mutable_infer_response()->mutable_model_version() = this->version;
                    resp.mutable_infer_response()->mutable_parameters()->operator[](MediapipeGraphExecutor::TIMESTAMP_PARAMETER_NAME).set_int64_param(packet.Timestamp().Value());
                    if (!streamSynchronizedWrite(stream, streamWriterMutex, resp)) {
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
        std::map<std::string, mediapipe::Packet> inputSidePackets{createInputSidePackets(&firstRequest)};
#if (PYTHON_DISABLE == 0)
        if (inputSidePackets.count(PYTHON_SESSION_SIDE_PACKET_TAG)) {
            const std::string absMessage = "Incoming input side packet: " + PYTHON_SESSION_SIDE_PACKET_TAG + " is special reserved name and cannot be used";
            SPDLOG_DEBUG("Failed to insert predefined input side packet: {} with error: {}", PYTHON_SESSION_SIDE_PACKET_TAG, absMessage);
            return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
        }
        inputSidePackets[PYTHON_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<PythonNodeResourcesMap>(this->pythonNodeResourcesMap).At(mediapipe::Timestamp(STARTING_TIMESTAMP));
#endif
        MP_RETURN_ON_FAIL(graph.StartRun(inputSidePackets), "graph start", StatusCode::MEDIAPIPE_GRAPH_START_ERROR);

        // Deserialize first request
        OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(this->partialDeserialize(
                                                  std::shared_ptr<const KFSRequest>(&firstRequest,
                                                      // Custom deleter to avoid deallocation by custom holder
                                                      // Conversion to shared_ptr is required for unified deserialization method
                                                      // for first and subsequent requests
                                                      [](const KFSRequest*) {}),
                                                  graph),
            "partial deserialization of first request");

        // Read loop
        // Here we create ModelInferRequest with shared ownership,
        // and move it down to custom packet holder to ensure
        // lifetime is extended to lifetime of deserialized Packets.
        auto req = std::make_shared<::inference::ModelInferRequest>();
        while (stream.Read(req.get())) {
            auto pstatus = this->validateSubsequentRequest(*req);
            if (pstatus.ok()) {
                OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(this->partialDeserialize(req, graph), "partial deserialization of subsequent requests");
            } else {
                OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(pstatus, "validate subsequent requests");
            }
            if (graph.HasError()) {
                SPDLOG_DEBUG("Graph {}: encountered an error, stopping the execution", this->name);
                break;
            }
            req = std::make_shared<::inference::ModelInferRequest>();
        }

        SPDLOG_DEBUG("Graph {}: Closing packet sources...", this->name);
        // Close input streams
        MP_RETURN_ON_FAIL(graph.CloseAllPacketSources(), "closing all packet sources", StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR);

        SPDLOG_DEBUG("Graph {}: Closed all packet sources. Waiting untill done...", this->name);
        MP_RETURN_ON_FAIL(graph.WaitUntilDone(), "waiting until done", StatusCode::MEDIAPIPE_EXECUTION_ERROR);
        SPDLOG_DEBUG("Graph {}: Done execution", this->name);
        return StatusCode::OK;
    } catch (...) {
        return Status(StatusCode::UNKNOWN_ERROR, "Exception while processing MediaPipe graph");  // To be displayed in method level above
    }
}

Status MediapipeGraphExecutor::serializePacket(const std::string& name, ::inference::ModelInferResponse& response, const ::mediapipe::Packet& packet) const {
    Status status;
    SPDLOG_DEBUG("Received packet from output stream: {}", name);
    if (this->outputTypes.at(name) == mediapipe_packet_type_enum::KFS_RESPONSE) {
        SPDLOG_DEBUG("Response processing packet type KFSPass name: {}", name);
        status = receiveAndSerializePacket<KFSResponse*>(packet, response, name);
    } else if (this->outputTypes.at(name) == mediapipe_packet_type_enum::TFTENSOR) {
        SPDLOG_DEBUG("Response processing packet type TF Tensor name: {}", name);
        status = receiveAndSerializePacket<tensorflow::Tensor>(packet, response, name);
    } else if (this->outputTypes.at(name) == mediapipe_packet_type_enum::TFLITETENSOR) {
        SPDLOG_DEBUG("Response processing packet type TFLite Tensor name: {}", name);
        std::string details{"Response processing packet type TFLite Tensor is not supported"};
        status = Status(StatusCode::NOT_IMPLEMENTED, std::move(details));
    } else if (this->outputTypes.at(name) == mediapipe_packet_type_enum::MPTENSOR) {
        SPDLOG_DEBUG("Response processing packet type MP Tensor name: {}", name);
        status = receiveAndSerializePacket<mediapipe::Tensor>(packet, response, name);
    } else if (this->outputTypes.at(name) == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
        SPDLOG_DEBUG("Response processing Mediapipe Image Frame: {}", name);
        status = receiveAndSerializePacket<mediapipe::ImageFrame>(packet, response, name);
#if (PYTHON_DISABLE == 0)
    } else if (this->outputTypes.at(name) == mediapipe_packet_type_enum::OVMS_PY_TENSOR) {
        SPDLOG_DEBUG("Response processing Ovms Python Tensor name: {}", name);
        status = receiveAndSerializePacket<PyObjectWrapper<py::object>>(packet, response, name);
#endif
    } else if ((this->outputTypes.at(name) == mediapipe_packet_type_enum::OVTENSOR) ||
               (this->outputTypes.at(name) == mediapipe_packet_type_enum::UNKNOWN)) {
        SPDLOG_DEBUG("Response processing packet type:  OVTensor name: {}", name);
        status = receiveAndSerializePacket<ov::Tensor>(packet, response, name);
    } else {
        status = Status(StatusCode::UNKNOWN_ERROR, "Unreachable code");
    }
    return status;
}
}  // namespace ovms
