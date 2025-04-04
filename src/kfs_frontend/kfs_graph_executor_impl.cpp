//*****************************************************************************
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
#include "kfs_graph_executor_impl.hpp"

#include <chrono>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../kfs_frontend/kfs_utils.hpp"
#include "../logging.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../predict_request_validation_utils.hpp"
#include "../status.hpp"
#include "../tfs_frontend/tfs_utils.hpp"

#pragma warning(push)
#pragma warning(disable : 6385 6386 6326 6011 6294 6201 4309 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#pragma warning(push)
#pragma warning(disable : 6269 6294 6201)
#include "opencv2/opencv.hpp"
#pragma warning(pop)

#if (PYTHON_DISABLE == 0)
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020 6001)
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma warning(pop)

#include "../python/python_backend.hpp"
#include "../python/pythonnoderesources.hpp"
#include "src/python/ovms_py_tensor.hpp"
namespace py = pybind11;
#endif

namespace ovms {

// Utilities

using namespace request_validation_utils;

const std::string TIMESTAMP_PARAMETER_NAME = "OVMS_MP_TIMESTAMP";

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

template <typename T>
static Status receiveAndSerializePacket(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName);

template <>
Status receiveAndSerializePacket<tensorflow::Tensor>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto& received = packet.Get<tensorflow::Tensor>();
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
        const auto& received = packet.Get<ov::Tensor>();
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
Status receiveAndSerializePacket<KFSResponse>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<KFSResponse>();
        response = std::move(received);
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

#define COPY_INPUT_VALUE_BY_VALUE(TYPE, PROTO_PREFIX)                                                                                       \
    TYPE* ptr = reinterpret_cast<TYPE*>(data);                                                                                              \
    const auto& input = request.inputs(inputIndex);                                                                                         \
    if (!input.has_contents()) {                                                                                                            \
        return Status(StatusCode::INVALID_CONTENT_SIZE, "Input does not have input tensor contents field");                                 \
    }                                                                                                                                       \
    const auto& contents = input.contents();                                                                                                \
    if (!contents.PROTO_PREFIX##_contents_size()) {                                                                                         \
        return Status(StatusCode::INVALID_CONTENT_SIZE, "Input does not have proper size of input tensor " #PROTO_PREFIX "contents field"); \
    }                                                                                                                                       \
    for (auto& number : contents.PROTO_PREFIX##_contents()) {                                                                               \
        *(ptr++) = number;                                                                                                                  \
    }                                                                                                                                       \
    break;

static Status validateRawInputContent(const size_t expectedBytes, const std::string bufferLocation, const std::string& requestedName, const KFSRequest& request) {
    if (expectedBytes != bufferLocation.size()) {
        std::stringstream ss;
        ss << "Expected: " << expectedBytes << " bytes; Actual: " << bufferLocation.size() << " bytes; input name: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }
    return StatusCode::OK;
}

static Status validateInputContent(const KFSTensorInputProto& proto, const size_t expectedBytes, const std::string& requestedName, const KFSRequest& request) {
    auto precision = KFSPrecisionToOvmsPrecision(proto.datatype());
    size_t elementsCount = getElementsCount(proto, precision);
    if (expectedBytes != KFSDataTypeSize(proto.datatype()) * elementsCount) {
        std::stringstream ss;
        ss << "Expected: " << expectedBytes << " values; Actual: " << KFSDataTypeSize(proto.datatype()) * elementsCount << " values; input name: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid value size of tensor proto - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_VALUE_COUNT, details);
    }
    return StatusCode::OK;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<mediapipe::Tensor>& outTensor, PythonBackend* pythonBackend) {
    auto requestInputItr = request.inputs().begin();
    OVMS_RETURN_ON_FAIL(getRequestInput(requestInputItr, requestedName, request));
    auto inputIndex = requestInputItr - request.inputs().begin();
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
        ov::element::Type precision = ovmsPrecisionToIE2Precision(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
        size_t expectedBytes = 1;
        bool expectedBufferSizeValid = computeExpectedBufferSizeReturnFalseIfOverflow<int>(rawShape, precision.size(), expectedBytes);
        if (!expectedBufferSizeValid) {
            const std::string details = "Provided shape and datatype declare too large buffer.";
            SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
        if (request.raw_input_contents().size()) {
            auto& bufferLocation = request.raw_input_contents().at(inputIndex);
            OVMS_RETURN_ON_FAIL(validateRawInputContent(expectedBytes, bufferLocation, requestedName, request));
            std::memcpy(data, bufferLocation.data(), bufferLocation.size());
        } else {  // need to copy each value separately
            OVMS_RETURN_ON_FAIL(validateInputContent(*requestInputItr, expectedBytes, requestedName, request));
            switch (datatype) {
            case mediapipe::Tensor::ElementType::kFloat32: {
                COPY_INPUT_VALUE_BY_VALUE(float, fp32);
            }
            case mediapipe::Tensor::ElementType::kInt32: {
                COPY_INPUT_VALUE_BY_VALUE(int32_t, int);
            }
            case mediapipe::Tensor::ElementType::kInt8: {
                COPY_INPUT_VALUE_BY_VALUE(int8_t, int);
            }
            case mediapipe::Tensor::ElementType::kUInt8: {
                COPY_INPUT_VALUE_BY_VALUE(uint8_t, uint);
            }
            case mediapipe::Tensor::ElementType::kBool: {
                COPY_INPUT_VALUE_BY_VALUE(bool, bool);
            }
            case mediapipe::Tensor::ElementType::kFloat16:
            default:
                return ovms::Status(ovms::StatusCode::NOT_IMPLEMENTED, "There is no support for types different than fp32, i32, i8, u8, bool");
            }
        }
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
        size_t expectedBytes = 1;
        bool expectedBufferSizeValid = computeExpectedBufferSizeReturnFalseIfOverflow<int64_t>(rawShape, KFSDataTypeSize(requestInputItr->datatype()), expectedBytes);
        if (!expectedBufferSizeValid) {
            const std::string details = "Provided shape and datatype declare too large buffer.";
            SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
        outTensor = std::make_unique<tensorflow::Tensor>(datatype, tensorShape);
        if (request.raw_input_contents().size()) {
            auto& bufferLocation = request.raw_input_contents().at(inputIndex);
            if (outTensor->TotalBytes() != bufferLocation.size()) {
                std::stringstream ss;
                ss << "Mediapipe deserialization content size mismatch; allocated TF Tensor: " << outTensor->TotalBytes() << " bytes vs KServe buffer: " << bufferLocation.size() << " bytes";
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_CONTENT_SIZE, details);
            }
            void* tfTensordata = outTensor->data();
            std::memcpy(tfTensordata, bufferLocation.data(), bufferLocation.size());
        } else {
            OVMS_RETURN_ON_FAIL(validateInputContent(*requestInputItr, expectedBytes, requestedName, request));
            void* data = outTensor->data();
            switch (datatype) {
            case TFSDataType::DT_FLOAT: {
                COPY_INPUT_VALUE_BY_VALUE(float, fp32);
            }
            case TFSDataType::DT_DOUBLE: {
                COPY_INPUT_VALUE_BY_VALUE(double, fp64);
            }
            case TFSDataType::DT_INT64: {
                COPY_INPUT_VALUE_BY_VALUE(int64_t, int64);
            }
            case TFSDataType::DT_INT32: {
                COPY_INPUT_VALUE_BY_VALUE(int32_t, int);
            }
            case TFSDataType::DT_INT16: {
                COPY_INPUT_VALUE_BY_VALUE(int16_t, int);
            }
            case TFSDataType::DT_INT8: {
                COPY_INPUT_VALUE_BY_VALUE(int8_t, int);
            }
            case TFSDataType::DT_UINT64: {
                COPY_INPUT_VALUE_BY_VALUE(uint64_t, uint64);
            }
            case TFSDataType::DT_UINT32: {
                COPY_INPUT_VALUE_BY_VALUE(uint32_t, uint);
            }
            case TFSDataType::DT_UINT16: {
                COPY_INPUT_VALUE_BY_VALUE(uint16_t, uint);
            }
            case TFSDataType::DT_UINT8: {
                COPY_INPUT_VALUE_BY_VALUE(uint8_t, uint);
            }
            case TFSDataType::DT_BOOL: {
                COPY_INPUT_VALUE_BY_VALUE(bool, bool);
            }
            case TFSDataType::DT_HALF:
            default:
                return ovms::Status(ovms::StatusCode::NOT_IMPLEMENTED, "There is no support for types different than fp32, int64, int32, uint32, uint64, int8, uint8, bool");
            }
        }
    }
    HANDLE_DESERIALIZATION_EXCEPTION("Tensorflow tensor")
    return StatusCode::OK;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, std::unique_ptr<ov::Tensor>& outTensor, PythonBackend* pythonBackend) {
    auto requestInputItr = request.inputs().begin();
    OVMS_RETURN_ON_FAIL(getRequestInput(requestInputItr, requestedName, request));
    auto inputIndex = requestInputItr - request.inputs().begin();
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
        size_t expectedBytes = 1;
        bool expectedBufferSizeValid = computeExpectedBufferSizeReturnFalseIfOverflow<size_t>(shape, precision.size(), expectedBytes);
        if (!expectedBufferSizeValid) {
            const std::string details = "Provided shape and datatype declare too large buffer.";
            SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
        if (request.raw_input_contents().size()) {
            auto& bufferLocation = request.raw_input_contents().at(inputIndex);
            OVMS_RETURN_ON_FAIL(validateRawInputContent(expectedBytes, bufferLocation, requestedName, request));
            if (expectedBytes == 0) {
                outTensor = std::make_unique<ov::Tensor>(precision, shape);  // OpenVINO does not accept nullptr as data ptr
            } else {
                outTensor = std::make_unique<ov::Tensor>(precision, shape, const_cast<void*>((const void*)bufferLocation.data()));
            }
        } else {
            OVMS_RETURN_ON_FAIL(validateInputContent(*requestInputItr, expectedBytes, requestedName, request));
            if (expectedBytes == 0) {
                return StatusCode::OK;
            }
            outTensor = std::make_unique<ov::Tensor>(precision, shape);
            void* data = outTensor->data();
            switch (precision) {
            case ov::element::Type_t::f32: {
                COPY_INPUT_VALUE_BY_VALUE(float, fp32);
            }
            case ov::element::Type_t::i64: {
                COPY_INPUT_VALUE_BY_VALUE(int64_t, int64);
            }
            case ov::element::Type_t::i32: {
                COPY_INPUT_VALUE_BY_VALUE(int32_t, int);
            }
            case ov::element::Type_t::i16: {
                COPY_INPUT_VALUE_BY_VALUE(int16_t, int);
            }
            case ov::element::Type_t::i8: {
                COPY_INPUT_VALUE_BY_VALUE(int8_t, int);
            }
            case ov::element::Type_t::u64: {
                COPY_INPUT_VALUE_BY_VALUE(uint64_t, uint64);
            }
            case ov::element::Type_t::u32: {
                COPY_INPUT_VALUE_BY_VALUE(uint32_t, uint);
            }
            case ov::element::Type_t::u16: {
                COPY_INPUT_VALUE_BY_VALUE(uint16_t, uint);
            }
            case ov::element::Type_t::u8: {
                COPY_INPUT_VALUE_BY_VALUE(uint8_t, uint);
            }
            case ov::element::Type_t::boolean: {
                COPY_INPUT_VALUE_BY_VALUE(bool, bool);
            }
            case ov::element::Type_t::f64: {
                COPY_INPUT_VALUE_BY_VALUE(double, fp64);
            }
            // the rest not supported by KFS
            case ov::element::Type_t::u1:
            case ov::element::Type_t::u4:
            case ov::element::Type_t::i4:
            case ov::element::Type_t::f16:
            case ov::element::Type_t::bf16:
            case ov::element::Type_t::dynamic:
            default:
                return ovms::Status(ovms::StatusCode::NOT_IMPLEMENTED, "There is no support for types different than fp32, i64, i32, i16, i8, u64, u32, u16, u8, bool");
            }
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
    if (request.raw_input_contents_size() <= inputIndex) {
        SPDLOG_DEBUG("Data should be located in raw_input_contents if graph input tag is IMAGE");
        return StatusCode::MEDIAPIPE_EXECUTION_ERROR;
    }
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

        ov::element::Type precision = ovmsPrecisionToIE2Precision(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
        auto formatIt = datatypeToBufferFormat.find(requestInputItr->datatype());
        if (request.raw_input_contents().size()) {
            auto& bufferLocation = request.raw_input_contents().at(inputIndex);
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
        } else {
            if ((precision != ov::element::Type_t::string) && formatIt == datatypeToBufferFormat.end()) {
                const std::string details = "Provided datatype is invalid, custom datatypes are allowed only when raw_input_contents is used.";
                SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_PRECISION, details);
            }
            size_t expectedBytes;
            if (precision == ov::element::Type_t::string) {
                expectedBytes = 0;
                for (const auto& contents : request.inputs(inputIndex).contents().bytes_contents()) {
                    expectedBytes += contents.size() + sizeof(uint32_t);
                }
            } else {
                expectedBytes = 1;
                bool expectedBufferSizeValid = computeExpectedBufferSizeReturnFalseIfOverflow<py::ssize_t>(shape, precision.size(), expectedBytes);
                if (!expectedBufferSizeValid) {
                    const std::string details = "Provided shape and datatype declare too large buffer.";
                    SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
                    return Status(StatusCode::INVALID_CONTENT_SIZE, details);
                }
                OVMS_RETURN_ON_FAIL(validateInputContent(*requestInputItr, expectedBytes, requestedName, request));
            }
            auto ok = pythonBackend->createEmptyOvmsPyTensor(
                requestedName,
                shape,
                requestInputItr->datatype(),
                expectedBytes,
                outTensor);
            if (!ok) {
                SPDLOG_DEBUG("Error creating empty Python tensor");
                return StatusCode::UNKNOWN_ERROR;
            }
            void* data;
            if (!pythonBackend->getOvmsPyTensorData(outTensor, &data)) {
                return Status(StatusCode::INTERNAL_ERROR);
            }
            switch (precision) {
            case ov::element::Type_t::f32: {
                COPY_INPUT_VALUE_BY_VALUE(float, fp32);
            }
            case ov::element::Type_t::f64: {
                COPY_INPUT_VALUE_BY_VALUE(double, fp64);
            }
            case ov::element::Type_t::i64: {
                COPY_INPUT_VALUE_BY_VALUE(int64_t, int64);
            }
            case ov::element::Type_t::i32: {
                COPY_INPUT_VALUE_BY_VALUE(int32_t, int);
            }
            case ov::element::Type_t::i16: {
                COPY_INPUT_VALUE_BY_VALUE(int16_t, int);
            }
            case ov::element::Type_t::i8: {
                COPY_INPUT_VALUE_BY_VALUE(int8_t, int);
            }
            case ov::element::Type_t::u64: {
                COPY_INPUT_VALUE_BY_VALUE(uint64_t, uint64);
            }
            case ov::element::Type_t::u32: {
                COPY_INPUT_VALUE_BY_VALUE(uint32_t, uint);
            }
            case ov::element::Type_t::u16: {
                COPY_INPUT_VALUE_BY_VALUE(uint16_t, uint);
            }
            case ov::element::Type_t::u8: {
                COPY_INPUT_VALUE_BY_VALUE(uint8_t, uint);
            }
            case ov::element::Type_t::boolean: {
                COPY_INPUT_VALUE_BY_VALUE(bool, bool);
            }
            case ov::element::Type_t::string: {
                uint32_t offset = 0;
                for (const auto& contents : request.inputs(inputIndex).contents().bytes_contents()) {
                    const uint32_t size = contents.size();
                    std::memcpy(reinterpret_cast<char*>(data) + offset, &size, sizeof(uint32_t));
                    offset += sizeof(uint32_t);
                    std::memcpy(reinterpret_cast<char*>(data) + offset, contents.data(), size);
                    offset += size;
                }
                return StatusCode::OK;
            }

            // the rest not supported by KFS
            case ov::element::Type_t::u1:
            case ov::element::Type_t::u4:
            case ov::element::Type_t::i4:
            case ov::element::Type_t::f16:
            case ov::element::Type_t::bf16:
            case ov::element::Type_t::dynamic:
            default:
                return ovms::Status(ovms::StatusCode::NOT_IMPLEMENTED, "There is no support for types different than fp32, i64, i32, i16, i8, u64, u32, u16, u8, bool");
            }

            if (!ok) {
                SPDLOG_DEBUG("Error creating Python tensor from data");
                return StatusCode::UNKNOWN_ERROR;
            }
        }
    }
    HANDLE_DESERIALIZATION_EXCEPTION("Ovms Python tensor")
    return StatusCode::OK;
}
#endif

template <typename T, template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& name, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const ::mediapipe::Timestamp& timestamp, PythonBackend* pythonBackend) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
    OVMS_RETURN_ON_FAIL(validateRequestCoherencyKFS(*request, request->model_name(), MediapipeGraphDefinition::VERSION));
    if (!request->raw_input_contents().empty() && (request->raw_input_contents().size() != request->inputs().size())) {
        std::stringstream ss;
        ss << "Size of raw_input_contents: " << request->raw_input_contents().size() << " is different than number of inputs: " << request->inputs().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request->model_name(), request->model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    std::unique_ptr<T> inputTensor;
    OVMS_RETURN_ON_FAIL(deserializeTensor(name, *request, inputTensor, pythonBackend));
    SPDLOG_ERROR("Current Timestamp before actual pushing:{}", timestamp.Value());
    MP_RETURN_ON_FAIL(graph.AddPacketToInputStream(
                          name,
                          ::mediapipe::packet_internal::Create(
                              new Holder<T>(inputTensor.release(), request))
                              .At(timestamp)),
        std::string("failed to add packet to stream: ") + name, StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
    return StatusCode::OK;
}

template <template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& name, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const ::mediapipe::Timestamp& timestamp, PythonBackend* pythonBackend) {
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

// Required for unary/streaming where it is OVMS who creates the request but it is not the packet type and we have to clean up.
// In case when passing ownership is not required (unary-unary or first request of streaming) it is enough to pass shared_ptr with no-op destructor.
// Specializations are for special case when the request itself is the packet and we need to ensure there is no double free.
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

template <template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& inputName, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const ::mediapipe::Timestamp& timestamp, const stream_types_mapping_t& inputTypes, PythonBackend* pythonBackend) {
    auto it = inputTypes.find(inputName);
    if (it == inputTypes.end()) {
        std::stringstream ss;
        ss << inputName << " is unexpected";
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Unexpected input name: {}", request->model_name(), request->model_version(), details);
        return Status(StatusCode::INVALID_UNEXPECTED_INPUT, details);
    }
    auto inputPacketType = it->second;
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

static inline Status checkTimestamp(const KFSRequest& request, const ::mediapipe::Timestamp& timestamp) {
    if (!timestamp.IsRangeValue()) {
        SPDLOG_DEBUG("Timestamp not in range: {}; for request to: {};", timestamp.DebugString(), request.model_name());
        return Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, timestamp.DebugString());
    }
    return StatusCode::OK;
}

static Status deserializeTimestampIfAvailable(
    const KFSRequest& request,
    ::mediapipe::Timestamp& timestamp) {
    auto timestampParamIt = request.parameters().find(TIMESTAMP_PARAMETER_NAME);
    if (timestampParamIt != request.parameters().end()) {
        SPDLOG_DEBUG("Found {} timestamp parameter in request for: {}", TIMESTAMP_PARAMETER_NAME, request.model_name());
        auto& parameterChoice = timestampParamIt->second;
        if (parameterChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            // Cannot create with error checking since error check = abseil death test
            timestamp = ::mediapipe::Timestamp::CreateNoErrorChecking(parameterChoice.int64_param());
            if (!timestamp.IsRangeValue()) {
                SPDLOG_DEBUG("Timestamp not in range: {}; for request to: {};", timestamp.DebugString(), request.model_name());
                return Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, timestamp.DebugString());
            }
        } else {
            auto status = Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, "Invalid timestamp format in request parameter OVMS_MP_TIMESTAMP. Should be int64");
            SPDLOG_DEBUG(status.string());
            return status;
        }
    } else {
        SPDLOG_ERROR("Current Timestamp before setting:{}", timestamp.Value());
        auto now = std::chrono::system_clock::now();
        timestamp = ::mediapipe::Timestamp(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
        SPDLOG_ERROR("Current Timestamp setting:{}", timestamp.Value());
    }
    return StatusCode::OK;
}

// Implementation

const std::string& getRequestId(
    const KFSRequest& request) {
    return request.id();
}

Status onPacketReadySerializeAndSendImpl(
    const std::string& requestId,
    const std::string& endpointName,
    const std::string& endpointVersion,
    const std::string& packetName,
    const mediapipe_packet_type_enum packetType,
    const ::mediapipe::Packet& packet,
    KFSServerReaderWriter& serverReaderWriter) {

    KFSStreamResponse resp;
    OVMS_RETURN_ON_FAIL(
        onPacketReadySerializeImpl(
            requestId,
            endpointName,
            endpointVersion,
            packetName,
            packetType,
            packet,
            *resp.mutable_infer_response()));

    if (!serverReaderWriter.Write(resp)) {
        return Status(StatusCode::UNKNOWN_ERROR, "client disconnected");
    }

    return StatusCode::OK;
}

Status onPacketReadySerializeImpl(
    const std::string& requestId,
    const std::string& endpointName,
    const std::string& endpointVersion,
    const std::string& packetName,
    const mediapipe_packet_type_enum packetType,
    const ::mediapipe::Packet& packet,
    KFSResponse& response) {
    Status status;
    SPDLOG_DEBUG("Received packet from output stream: {}", packetName);
    if (packetType == mediapipe_packet_type_enum::KFS_RESPONSE) {
        SPDLOG_DEBUG("Response processing packet type KFSPass name: {}", packetName);
        status = receiveAndSerializePacket<KFSResponse>(packet, response, packetName);
    } else if (packetType == mediapipe_packet_type_enum::TFTENSOR) {
        SPDLOG_DEBUG("Response processing packet type TF Tensor name: {}", packetName);
        status = receiveAndSerializePacket<tensorflow::Tensor>(packet, response, packetName);
    } else if (packetType == mediapipe_packet_type_enum::TFLITETENSOR) {
        SPDLOG_DEBUG("Response processing packet type TFLite Tensor name: {}", packetName);
        std::string details{"Response processing packet type TFLite Tensor is not supported"};
        status = Status(StatusCode::NOT_IMPLEMENTED, std::move(details));
    } else if (packetType == mediapipe_packet_type_enum::MPTENSOR) {
        SPDLOG_DEBUG("Response processing packet type MP Tensor name: {}", packetName);
        status = receiveAndSerializePacket<mediapipe::Tensor>(packet, response, packetName);
    } else if (packetType == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
        SPDLOG_DEBUG("Response processing Mediapipe Image Frame: {}", packetName);
        status = receiveAndSerializePacket<mediapipe::ImageFrame>(packet, response, packetName);
#if (PYTHON_DISABLE == 0)
    } else if (packetType == mediapipe_packet_type_enum::OVMS_PY_TENSOR) {
        SPDLOG_DEBUG("Response processing Ovms Python Tensor name: {}", packetName);
        status = receiveAndSerializePacket<PyObjectWrapper<py::object>>(packet, response, packetName);
#endif
    } else if ((packetType == mediapipe_packet_type_enum::OVTENSOR) ||
               (packetType == mediapipe_packet_type_enum::UNKNOWN)) {
        SPDLOG_DEBUG("Response processing packet type:  OVTensor name: {}", packetName);
        status = receiveAndSerializePacket<ov::Tensor>(packet, response, packetName);
    } else {
        SPDLOG_DEBUG("Unknown error in packet serialization for packet: {}. Unreachable code", packetName);
        status = Status(StatusCode::UNKNOWN_ERROR, "Unreachable code");
    }
    response.set_model_name(endpointName);
    response.set_model_version(endpointVersion);
    response.set_id(requestId);
    response.mutable_parameters()->operator[](TIMESTAMP_PARAMETER_NAME).set_int64_param(packet.Timestamp().Value());
    return status;
}

Status createAndPushPacketsImpl(
    std::shared_ptr<const KFSRequest> request,
    stream_types_mapping_t& inputTypes,
    PythonBackend* pythonBackend,
    ::mediapipe::CalculatorGraph& graph,
    ::mediapipe::Timestamp& currentTimestamp,
    size_t& numberOfPacketsCreated) {

    OVMS_RETURN_ON_FAIL(deserializeTimestampIfAvailable(*request, currentTimestamp));
    OVMS_RETURN_ON_FAIL(checkTimestamp(*request, currentTimestamp));
    OVMS_RETURN_ON_FAIL(validateRequestCoherencyKFS(*request, request->model_name(), MediapipeGraphDefinition::VERSION));

    numberOfPacketsCreated = 0;
    for (const auto& input : request->inputs()) {
        const auto& inputName = input.name();
        auto status = createPacketAndPushIntoGraph<HolderWithRequestOwnership>(
            inputName, request, graph, currentTimestamp, inputTypes, pythonBackend);
        if (!status.ok()) {
            return status;
        }
        numberOfPacketsCreated++;
    }

    return StatusCode::OK;
}

Status deserializeInputSidePacketsFromFirstRequestImpl(
    std::map<std::string, mediapipe::Packet>& inputSidePackets,
    const KFSRequest& request) {
    static const std::string PYTHON_SESSION_SIDE_PACKET_TAG{"py"};
    for (const auto& [name, valueChoice] : request.parameters()) {
        SPDLOG_DEBUG("Found: {}; parameter in request for: {};", name, request.model_name());
        if (name == TIMESTAMP_PARAMETER_NAME) {
            SPDLOG_DEBUG("Ignored: {}; parameter in request for: {}; Parameter is reserved for MediaPipe input packet timestamps", name, request.model_name());
            continue;
        }
        if (name == PYTHON_SESSION_SIDE_PACKET_TAG) {
            const std::string absMessage = "Incoming input side packet: " + PYTHON_SESSION_SIDE_PACKET_TAG + " is special reserved name and cannot be used";
            SPDLOG_DEBUG("Failed to insert predefined input side packet: {} with error: {}", PYTHON_SESSION_SIDE_PACKET_TAG, absMessage);
            return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
        }
        if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kStringParam) {
            inputSidePackets[name] = mediapipe::MakePacket<std::string>(valueChoice.string_param());
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            inputSidePackets[name] = mediapipe::MakePacket<int64_t>(valueChoice.int64_param());
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kBoolParam) {
            inputSidePackets[name] = mediapipe::MakePacket<bool>(valueChoice.bool_param());
        } else {
            SPDLOG_DEBUG("Handling parameters of other types than: bool, string, int64 is not supported");
            return Status(StatusCode::NOT_IMPLEMENTED, "Handling parameters of other types than: bool, string, int64 is not supported");
        }
    }
    return StatusCode::OK;
}

Status validateSubsequentRequestImpl(
    const KFSRequest& request,
    const std::string& endpointName,
    const std::string& endpointVersion,
    stream_types_mapping_t& inputTypes) {
    if (request.model_name() != endpointName) {
        return StatusCode::MEDIAPIPE_INCORRECT_SERVABLE_NAME;
    }
    if (request.model_version() != endpointVersion &&
        request.model_version() != "0" &&    // default version does not matter for user
        !request.model_version().empty()) {  // empty the same as default
        return StatusCode::MEDIAPIPE_INCORRECT_SERVABLE_VERSION;
    }
    return StatusCode::OK;
}

Status sendErrorImpl(
    const std::string& message,
    KFSServerReaderWriter& serverReaderWriter) {
    ::inference::ModelStreamInferResponse resp;
    *resp.mutable_error_message() = message;

    if (serverReaderWriter.Write(resp)) {
        return StatusCode::OK;
    }

    return Status(StatusCode::UNKNOWN_ERROR, "error during sending an error response");
}

bool waitForNewRequest(
    KFSServerReaderWriter& serverReaderWriter,
    KFSRequest& newRequest) {
    return serverReaderWriter.Read(&newRequest);
}

}  // namespace ovms
