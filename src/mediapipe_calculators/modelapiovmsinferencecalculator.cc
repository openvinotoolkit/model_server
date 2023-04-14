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

#include "../ovms.h"           // NOLINT
#include "../stringutils.hpp"  // TODO dispose
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "modelapiovmsadapter.hpp"
#include "modelapiovmsadapterwrapper.hpp"
#include "src/mediapipe_calculators/ovmscalculator.pb.h"
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {
#define MLOG(A) LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << A << std::endl;

using ovms::AdapterWrapper;
using ovms::InferenceInput;
using ovms::InferenceOutput;
using ovms::OVMSInferenceAdapter;
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
            LOG(ERROR) << "Error encountred in OVMSCalculator:" << msg << " code: " << code; \
            OVMS_StatusDelete(err);                                                          \
            RET_CHECK(err == nullptr);                                                       \
        }                                                                                    \
    }
#define CREATE_GUARD(GUARD_NAME, CAPI_TYPE, CAPI_PTR) \
    std::unique_ptr<CAPI_TYPE, decltype(&(CAPI_TYPE##Delete))> GUARD_NAME(CAPI_PTR, &(CAPI_TYPE##Delete));

}  // namespace

const std::string SESSION_TAG{"SESSION"};

class ModelAPISideFeedCalculator : public CalculatorBase {
    OVMSInferenceAdapter* session{nullptr};

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        MLOG("Main GetContract start");
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        for (const std::string& tag : cc->Inputs().GetTags()) {
            cc->Inputs().Tag(tag).Set<ov::Tensor>();
        }
        for (const std::string& tag : cc->Outputs().GetTags()) {
            cc->Outputs().Tag(tag).Set<ov::Tensor>();
        }
        cc->InputSidePackets().Tag(SESSION_TAG.c_str()).Set<AdapterWrapper>();
        MLOG("Main GetContract end");
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        MLOG("Main Close");
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        MLOG("Main Open start");
        session = cc->InputSidePackets()
                      .Tag(SESSION_TAG.c_str())
                      .Get<AdapterWrapper>()
                      .adapter.get();
        for (CollectionItemId id = cc->Inputs().BeginId();
             id < cc->Inputs().EndId(); ++id) {
            if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
            }
        }
        if (cc->OutputSidePackets().NumEntries() != 0) {
            for (CollectionItemId id = cc->InputSidePackets().BeginId();
                 id < cc->InputSidePackets().EndId(); ++id) {
                cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
            }
        }
        cc->SetOffset(TimestampDiff(0));

        MLOG("Main Open end");
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        MLOG("Main process start");
        if (cc->Inputs().NumEntries() == 0) {
            return tool::StatusStop();
        }
        /////////////////////
        // PREPARE INPUT MAP
        /////////////////////

        const auto& inputTagInputMap = session->inputTagToName;
        InferenceInput input;
        InferenceOutput output;
        for (const std::string& tag : cc->Inputs().GetTags()) {
            const char* realInputName = inputTagInputMap.at(tag).c_str();
            auto& packet = cc->Inputs().Tag(tag).Get<ov::Tensor>();
            input[realInputName] = packet;
            ov::Tensor input_tensor(packet);
            const float* input_tensor_access = reinterpret_cast<float*>(input_tensor.data());
            std::stringstream ss;
            ss << "ModelAPICalculator received tensor: [ ";
            for (int x = 0; x < 10; ++x) {
                ss << input_tensor_access[x] << " ";
            }
            ss << " ] timestamp: " << cc->InputTimestamp().DebugString() << endl;
            MLOG(ss.str());
        }
        //////////////////
        //  INFERENCE
        //////////////////
        output = session->infer(input);
        auto outputsCount = output.size();
        RET_CHECK(outputsCount == cc->Outputs().GetTags().size());
        // TODO check for existence of each tag

        for (const auto& [outputName, outputTagName] : session->outputNameToTag) {
            ov::Tensor* outOvTensor = new ov::Tensor(output.at(outputName));
            // TODO check buffer ownership
            cc->Outputs().Tag(outputTagName).Add(outOvTensor, cc->InputTimestamp());
        }
        MLOG("Main process end");
        return absl::OkStatus();
    }
};

class ModelAPISessionCalculator : public CalculatorBase {
    std::unique_ptr<AdapterWrapper> adapter;
    std::unordered_map<std::string, std::string> outputNameToTag;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        MLOG("Session GetContract start");
        RET_CHECK(cc->Inputs().GetTags().empty());
        RET_CHECK(cc->Outputs().GetTags().empty());
        cc->OutputSidePackets().Tag(SESSION_TAG.c_str()).Set<AdapterWrapper>();
        const auto& options = cc->Options<OVMSCalculatorOptions>();
        RET_CHECK(!options.servable_name().empty());
        MLOG("Session GetContract middle");
        // TODO validate version from string
        // TODO validate service url format
        RET_CHECK(options.config_path().empty() ||
                  options.service_url().empty());
        // TODO validate tag_to_tensor maps so that key fulfill regex
        MLOG("Session GetContract end");
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        MLOG("Session Open start");
        for (CollectionItemId id = cc->Inputs().BeginId();
             id < cc->Inputs().EndId(); ++id) {
            if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
            }
        }
        if (cc->OutputSidePackets().NumEntries() != 0) {
            for (CollectionItemId id = cc->InputSidePackets().BeginId();
                 id < cc->InputSidePackets().EndId(); ++id) {
                cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
            }
        }
        cc->SetOffset(TimestampDiff(0));

        const std::string& servableName = cc->Options<OVMSCalculatorOptions>().servable_name();
        const std::string& servableVersion = cc->Options<OVMSCalculatorOptions>().servable_version();
        auto session = std::make_unique<AdapterWrapper>(new OVMSInferenceAdapter(servableName));
        const auto& options = cc->Options<OVMSCalculatorOptions>();
        for (const auto& [key, value] : options.tag_to_output_tensor_names()) {
            session->adapter->outputNameToTag[value] = key;
        }
        for (const auto& [key, value] : options.tag_to_input_tensor_names()) {
            session->adapter->inputTagToName[key] = value;
        }
        MLOG("Session create adapter");
        cc->OutputSidePackets().Tag(SESSION_TAG.c_str()).Set(Adopt(session.release()));
        MLOG("SessionOpen end");
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        MLOG("SessionProcess");
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(ModelAPISideFeedCalculator);
}  // namespace mediapipe
