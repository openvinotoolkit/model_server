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
#include "modelapiovmsadapter.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#include "../stringutils.hpp"  // TODO dispose
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "src/mediapipe_calculators/ovmscalculator.pb.h"
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {
#define MLOG(A) LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << A << std::endl;

using std::endl;

#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)                                                  \
    {                                                                                        \
        auto* err = C_API_CALL;                                                              \
        if (err != nullptr) {                                                                \
            uint32_t code = 0;                                                               \
            const char* msg = nullptr;                                                       \
            OVMS_StatusGetCode(err, &code);                                                  \
            OVMS_StatusGetDetails(err, &msg);                                                \
            LOG(ERROR) << "Error encountred in OVMSCalculator:" << msg << " code: " << code; \
            OVMS_StatusDelete(err);                                                          \
            RET_CHECK(err == nullptr);                                                       \
        }                                                                                    \
    }
#define CREATE_GUARD(GUARD_NAME, CAPI_TYPE, CAPI_PTR) \
    std::unique_ptr<CAPI_TYPE, decltype(&(CAPI_TYPE##Delete))> GUARD_NAME(CAPI_PTR, &(CAPI_TYPE##Delete));

using InferenceOutput = std::map<std::string, ov::Tensor>;
using InferenceInput = std::map<std::string, ov::Tensor>;
// TODO
// * why std::map
// * no ret code from infer()
// * no ret code from load()
namespace ovms {
static OVMS_DataType OVPrecision2CAPI(ov::element::Type_t datatype);
static ov::element::Type_t CAPI2OVPrecision(OVMS_DataType datatype);
static ov::Tensor* makeOvTensor(OVMS_DataType datatype, const int64_t* shape, size_t dimCount, const void* voutputData, size_t bytesize);
static ov::Tensor makeOvTensorO(OVMS_DataType datatype, const int64_t* shape, size_t dimCount, const void* voutputData, size_t bytesize);

OVMSInferenceAdapter::OVMSInferenceAdapter(const std::string& servableName, uint32_t servableVersion) :
    servableName(servableName),
    servableVersion(servableVersion) {
    OVMS_ServerNew(&cserver);  // TODO retcheck's;
}
OVMSInferenceAdapter::~OVMSInferenceAdapter() {
}
InferenceOutput OVMSInferenceAdapter::infer(const InferenceInput& input) {
    /////////////////////
    // PREPARE REQUEST
    /////////////////////
    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceRequestNew(&request, cserver, servableName.c_str(), servableVersion);
    CREATE_GUARD(requestGuard, OVMS_InferenceRequest, request);

    // PREPARE EACH INPUT
    // extract single tensor
    for (const auto& [name, input_tensor] : input) {
        // TODO validate existence of tag key in map
        // or handle inference when there is no need for mapping
        const char* realInputName = name.c_str();

        const float* input_tensor_access = reinterpret_cast<float*>(input_tensor.data());
        std::stringstream ss;
        ss << " Adapter received tensor: [ ";
        for (int x = 0; x < 10; ++x) {
            ss << input_tensor_access[x] << " ";
        }
        ss << " ]";
        MLOG(ss.str());
        const auto& ovinputShape = input_tensor.get_shape();
        std::vector<int64_t> inputShape{ovinputShape.begin(), ovinputShape.end()};  // TODO error handling shape conversion
        OVMS_DataType inputDataType = OVPrecision2CAPI(input_tensor.get_element_type());
        OVMS_InferenceRequestAddInput(request, realInputName, inputDataType, inputShape.data(), inputShape.size());  // TODO retcode
        const uint32_t NOT_USED_NUM = 0;
        // TODO handle hardcoded buffertype, notUsedNum additional options? side packets?
        OVMS_InferenceRequestInputSetData(request,
            realInputName,
            reinterpret_cast<void*>(input_tensor.data()),
            input_tensor.get_byte_size(),
            OVMS_BUFFERTYPE_CPU,
            NOT_USED_NUM);  // TODO retcode
    }
    //////////////////
    //  INFERENCE
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    if (nullptr != OVMS_Inference(cserver, request, &response)) {
        MLOG("Inference failed!");
    }
    CREATE_GUARD(responseGuard, OVMS_InferenceResponse, response);
    // verify GetOutputCount
    uint32_t outputCount = 42;
    OVMS_InferenceResponseGetOutputCount(response, &outputCount);
    uint32_t parameterCount = 42;
    OVMS_InferenceResponseGetParameterCount(response, &parameterCount);
    // TODO handle output filtering. Graph definition could suggest
    // that we are not interested in all outputs from OVMS Inference
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    InferenceOutput output;
    for (size_t i = 0; i < outputCount; ++i) {
        OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId);
        output[outputName] = makeOvTensorO(datatype, shape, dimCount, voutputData, bytesize);  // TODO optimize FIXME
    }
    return output;
}
void OVMSInferenceAdapter::loadModel(const std::shared_ptr<const ov::Model>& model, ov::Core& core,
    const std::string& device, const ov::AnyMap& compilationConfig) {}
ov::Shape OVMSInferenceAdapter::getInputShape(const std::string& inputName) const { return {}; }  // TODO
std::vector<std::string> OVMSInferenceAdapter::getInputNames() { return {}; }                     // TODO
std::vector<std::string> OVMSInferenceAdapter::getOutputNames() { return {}; }                    // TODO
                                                                                                  //    virtual const ov::AnyMap& getModelConfig() const = 0; // TODO
const std::string& OVMSInferenceAdapter::getModelConfig() const {
    static std::string res{"MODEL_CONFIG_JSON"};
    return res;
}  // TODO
static OVMS_DataType OVPrecision2CAPI(ov::element::Type_t datatype) {
    static std::unordered_map<ov::element::Type_t, OVMS_DataType> precisionMap{
        {ov::element::Type_t::f64, OVMS_DATATYPE_FP64},
        {ov::element::Type_t::f32, OVMS_DATATYPE_FP32},
        {ov::element::Type_t::f16, OVMS_DATATYPE_FP16},
        {ov::element::Type_t::i64, OVMS_DATATYPE_I64},
        {ov::element::Type_t::i32, OVMS_DATATYPE_I32},
        {ov::element::Type_t::i16, OVMS_DATATYPE_I16},
        {ov::element::Type_t::i8, OVMS_DATATYPE_I8},
        {ov::element::Type_t::i4, OVMS_DATATYPE_I4},
        {ov::element::Type_t::u64, OVMS_DATATYPE_U64},
        {ov::element::Type_t::u32, OVMS_DATATYPE_U32},
        {ov::element::Type_t::u16, OVMS_DATATYPE_U16},
        {ov::element::Type_t::u8, OVMS_DATATYPE_U8},
        {ov::element::Type_t::u4, OVMS_DATATYPE_U4},
        {ov::element::Type_t::u1, OVMS_DATATYPE_U1},
        {ov::element::Type_t::boolean, OVMS_DATATYPE_BOOL},
        {ov::element::Type_t::bf16, OVMS_DATATYPE_BF16},
        {ov::element::Type_t::undefined, OVMS_DATATYPE_UNDEFINED},
        {ov::element::Type_t::dynamic, OVMS_DATATYPE_DYNAMIC}
        //    {ov::element::Type_t::, OVMS_DATATYPE_MIXEDMIXED},
        //    {ov::element::Type_t::, OVMS_DATATYPE_Q78Q78},
        //    {ov::element::Type_t::, OVMS_DATATYPE_BINBIN},
        //    {ov::element::Type_t::, OVMS_DATATYPE_CUSTOMCUSTOM
    };
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return OVMS_DATATYPE_UNDEFINED;
    }
    return it->second;
}

static ov::element::Type_t CAPI2OVPrecision(OVMS_DataType datatype) {
    static std::unordered_map<OVMS_DataType, ov::element::Type_t> precisionMap{
        {OVMS_DATATYPE_FP64, ov::element::Type_t::f64},
        {OVMS_DATATYPE_FP32, ov::element::Type_t::f32},
        {OVMS_DATATYPE_FP16, ov::element::Type_t::f16},
        {OVMS_DATATYPE_I64, ov::element::Type_t::i64},
        {OVMS_DATATYPE_I32, ov::element::Type_t::i32},
        {OVMS_DATATYPE_I16, ov::element::Type_t::i16},
        {OVMS_DATATYPE_I8, ov::element::Type_t::i8},
        {OVMS_DATATYPE_I4, ov::element::Type_t::i4},
        {OVMS_DATATYPE_U64, ov::element::Type_t::u64},
        {OVMS_DATATYPE_U32, ov::element::Type_t::u32},
        {OVMS_DATATYPE_U16, ov::element::Type_t::u16},
        {OVMS_DATATYPE_U8, ov::element::Type_t::u8},
        {OVMS_DATATYPE_U4, ov::element::Type_t::u4},
        {OVMS_DATATYPE_U1, ov::element::Type_t::u1},
        {OVMS_DATATYPE_BOOL, ov::element::Type_t::boolean},
        {OVMS_DATATYPE_BF16, ov::element::Type_t::bf16},
        {OVMS_DATATYPE_UNDEFINED, ov::element::Type_t::undefined},
        {OVMS_DATATYPE_DYNAMIC, ov::element::Type_t::dynamic}
        //    {OVMS_DATATYPE_MIXED, ov::element::Type_t::MIXED},
        //    {OVMS_DATATYPE_Q78, ov::element::Type_t::Q78},
        //    {OVMS_DATATYPE_BIN, ov::element::Type_t::BIN},
        //    {OVMS_DATATYPE_CUSTOM, ov::element::Type_t::CUSTOM
    };
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return ov::element::Type_t::undefined;
    }
    return it->second;
}

static ov::Tensor* makeOvTensor(OVMS_DataType datatype, const int64_t* shape, size_t dimCount, const void* voutputData, size_t bytesize) {
    ov::Shape ovShape;
    for (size_t i = 0; i < dimCount; ++i) {
        ovShape.push_back(shape[i]);
    }
    // here we make copy of underlying OVMS repsonse tensor
    ov::Tensor* output = new ov::Tensor(CAPI2OVPrecision(datatype), ovShape);
    std::memcpy(output->data(), voutputData, bytesize);
    return output;
}

static ov::Tensor makeOvTensorO(OVMS_DataType datatype, const int64_t* shape, size_t dimCount, const void* voutputData, size_t bytesize) {
    ov::Shape ovShape;
    for (size_t i = 0; i < dimCount; ++i) {
        ovShape.push_back(shape[i]);
    }
    // here we make copy of underlying OVMS repsonse tensor
    ov::Tensor output(CAPI2OVPrecision(datatype), ovShape);
    std::memcpy(output.data(), voutputData, bytesize);
    return output;
}
}  // namespace ovms
}  // namespace mediapipe
