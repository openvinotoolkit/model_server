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
        cc->Inputs().Tag("REQUEST").Set<const KFSRequest*>();
        cc->Outputs().Tag("RESPONSE").Set<KFSResponse*>();

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

        const KFSRequest* request = cc->Inputs().Tag("REQUEST").Get<const KFSRequest*>();
        KFSResponse* response = new KFSResponse();

        for (int i = 0; i < request->inputs().size(); i++){
            auto* output = response->add_outputs();
            output->set_datatype(request->inputs()[i].datatype());
            output->set_name("out");
            for (int j = 0; j < request->inputs()[i].shape_size(); j++){
                output->add_shape(request->inputs()[i].shape().at(j));
            }
        }
        
        for (int i = 0; i < request->raw_input_contents().size(); i++){

            response->add_raw_output_contents()->assign(request->raw_input_contents()[i].data(), request->raw_input_contents()[i].size());
        }

        cc->Outputs().Tag("RESPONSE").AddPacket(::mediapipe::MakePacket<KFSResponse*>(response).At(::mediapipe::Timestamp(0)));

        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(OVMSKFSPassCalculator);
}  // namespace mediapipe
