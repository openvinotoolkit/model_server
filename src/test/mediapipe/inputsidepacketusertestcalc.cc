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

#include "../../stringutils.hpp"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
namespace mediapipe {
#define MLOG(A) LOG(ERROR) << " " << A << std::endl;
using std::endl;

const std::string ISP_STRING{"INPUT_SIDE_PACKET_STRING"};
const std::string ISP_INT64{"INPUT_SIDE_PACKET_INT64"};
const std::string ISP_BOOL{"INPUT_SIDE_PACKET_BOOL"};
const std::string IN_FP32_TAG{"INPUT_FP32"};
const std::string INT32_TAG{"OUTPUT_UINT8"};
const std::string INT64_TAG{"OUTPUT_INT64"};
const std::string BOOL_TAG{"OUTPUT_BOOL"};

class InputSidePacketUserTestCalc : public CalculatorBase {
    std::string stringParam;
    bool boolParam;
    int64_t int64Param;
    std::unordered_map<std::string, std::string> outputNameToTag;  // TODO move to Open();

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        MLOG("InputSidePacketUserTestCalc GetContract start");
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(IN_FP32_TAG).Set<ov::Tensor>();
        cc->Outputs().Tag(INT32_TAG).Set<ov::Tensor>();
        cc->Outputs().Tag(INT64_TAG).Set<ov::Tensor>();
        cc->Outputs().Tag(BOOL_TAG).Set<ov::Tensor>();
        cc->InputSidePackets().Tag(ISP_STRING.c_str()).Set<std::string>();
        cc->InputSidePackets().Tag(ISP_INT64.c_str()).Set<int64_t>();
        cc->InputSidePackets().Tag(ISP_BOOL.c_str()).Set<bool>();
        MLOG("InputSidePacketUserTestCalc GetContract end");
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        MLOG("InputSidePacketUserTestCalc Close");
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        MLOG("InputSidePacketUserTestCalc Open start");
        stringParam = cc->InputSidePackets()
                          .Tag(ISP_STRING.c_str())
                          .Get<std::string>();
        boolParam = cc->InputSidePackets()
                        .Tag(ISP_BOOL.c_str())
                        .Get<bool>();
        int64Param = cc->InputSidePackets()
                         .Tag(ISP_INT64.c_str())
                         .Get<int64_t>();
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
        MLOG("InputSidePacketUserTestCalc Open end");
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        MLOG("InputSidePacketUserTestCalc process start");
        if (cc->Inputs().NumEntries() == 0) {
            return tool::StatusStop();
        }
        ov::Tensor* intTensor = new ov::Tensor(ov::element::Type_t::i64, {1});
        cc->Outputs().Tag(INT64_TAG).Add(intTensor, cc->InputTimestamp());
        *((int64_t*)intTensor->data()) = int64Param;
        std::cout << intTensor->get_byte_size() << std::endl;
        ov::Tensor* boolTensor = new ov::Tensor(ov::element::Type_t::boolean, {1});
        *((bool*)boolTensor->data()) = boolParam;
        std::cout << boolTensor->get_byte_size() << std::endl;
        cc->Outputs().Tag(BOOL_TAG).Add(boolTensor, cc->InputTimestamp());
        // there is no string to/from tensor in ovms
        ov::Tensor* stringTensor = new ov::Tensor(ov::element::Type_t::u8, {stringParam.size()});
        std::memcpy(stringTensor->data(), stringParam.data(), stringParam.size());
        cc->Outputs().Tag(INT32_TAG).Add(stringTensor, cc->InputTimestamp());
        MLOG("InputSidePacketUserTestCalc process end");
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(InputSidePacketUserTestCalc);
}  // namespace mediapipe
