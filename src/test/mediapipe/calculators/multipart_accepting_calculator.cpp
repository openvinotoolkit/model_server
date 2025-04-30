//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include "../../../http_payload.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

namespace mediapipe {

class MultipartAcceptingCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        auto payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();
        RET_CHECK(payload.multipartParser != nullptr);
        std::string email = payload.multipartParser->getFieldByName("email");
        std::string username = payload.multipartParser->getFieldByName("username");
        std::string_view fileContent = payload.multipartParser->getFileContentByFieldName("file");

        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string{email + std::string{"+"} + username + std::string{"\n"} + std::string(fileContent)}, cc->InputTimestamp());
        return absl::OkStatus();
    }
};
#pragma GCC diagnostic pop

const std::string MultipartAcceptingCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string MultipartAcceptingCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(MultipartAcceptingCalculator);
}  // namespace mediapipe
