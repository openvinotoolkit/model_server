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
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "../../ovms.h"           // NOLINT
#include "../../stringutils.hpp"  // TODO dispose
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "src/mediapipe_calculators/ovmscalculator.pb.h"
#include "src/kfs_frontend/kfs_grpc_inference_service.hpp"

// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {
#define MLOG(A) LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << A << std::endl;

using std::endl;

namespace {
#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)                                                  \
    {                                                                                        \
        auto* err = C_API_CALL;                                                              \
        if (err != nullptr) {                                                                \
            uint32_t code = 0;                                                               \
            const char* msg = nullptr;                                                       \
            OVMS_StatusGetCode(err, &code);                                                  \
            OVMS_StatusGetDetails(err, &msg);                                                \
            LOG(ERROR) << "Error encountred in OVMSKFSPassCalculator:" << msg << " code: " << code; \
            OVMS_StatusDelete(err);                                                          \
            RET_CHECK(err == nullptr);                                                       \
        }                                                                                    \
    }
#define CREATE_GUARD(GUARD_NAME, CAPI_TYPE, CAPI_PTR) \
    std::unique_ptr<CAPI_TYPE, decltype(&(CAPI_TYPE##Delete))> GUARD_NAME(CAPI_PTR, &(CAPI_TYPE##Delete));
}  // namespace

class OVMSKFSPassCalculator : public CalculatorBase {
    OVMS_Server* cserver{nullptr};
    OVMS_ServerSettings* _serverSettings{nullptr};
    OVMS_ModelsSettings* _modelsSettings{nullptr};
    std::unordered_map<std::string, std::string> outputNameToTag;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        for (const std::string& tag : cc->Inputs().GetTags()) {
            cc->Inputs().Tag(tag).Set<KFSRequest>();
        }
        for (const std::string& tag : cc->Outputs().GetTags()) {
            cc->Outputs().Tag(tag).Set<KFSResponse>();
        }

        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
            if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
            }
        }
        if (cc->OutputSidePackets().NumEntries() != 0) {
            for (CollectionItemId id = cc->InputSidePackets().BeginId(); id < cc->InputSidePackets().EndId(); ++id) {
                cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
            }
        }
        cc->SetOffset(TimestampDiff(0));
        OVMS_ServerNew(&cserver);

        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        /////////////////////
        // PREPARE REQUEST
        /////////////////////
        OVMS_InferenceRequest* request{nullptr};
        //TODOASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, options.servable_name().c_str(), servableVersion));
        CREATE_GUARD(requestGuard, OVMS_InferenceRequest, request);

        //for (const std::string& tag : cc->Inputs().GetTags()) {
            // TODO validate existence of tag key in map
            //const char* realInputName = inputTagInputMap.at(tag).c_str();

        //}
        //////////////////
        //  INFERENCE
        //////////////////
        OVMS_InferenceResponse* response = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
        CREATE_GUARD(responseGuard, OVMS_InferenceResponse, response);
        // verify GetOutputCount
        uint32_t outputCount = 42;
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutputCount(response, &outputCount));
        RET_CHECK(outputCount == cc->Outputs().GetTags().size());
        uint32_t parameterCount = 42;
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetParameterCount(response, &parameterCount));
        // TODO handle output filtering. Graph definition could suggest
        // that we are not interested in all outputs from OVMS Inference
        const void* voutputData;
        size_t bytesize = 42;
        //uint32_t outputId = 0;
        OVMS_DataType datatype = (OVMS_DataType)199;
        const int64_t* shape{nullptr};
        size_t dimCount = 42;
        OVMS_BufferType bufferType = (OVMS_BufferType)199;
        uint32_t deviceId = 42;
        const char* outputName{nullptr};
        for (size_t i = 0; i < outputCount; ++i) {
            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutput(response, i, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
            //ov::Tensor* outOvTensor = makeOvTensor(datatype, shape, dimCount, voutputData, bytesize);
            //cc->Outputs().Tag(outputNameToTag.at(outputName)).Add(outOvTensor, cc->InputTimestamp());
        }
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(OVMSKFSPassCalculator);
}  // namespace mediapipe
