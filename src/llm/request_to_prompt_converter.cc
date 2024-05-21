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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

#include <string>

#include "http_payload.hpp"
constexpr size_t BATCH_SIZE = 1;

namespace mediapipe {

class RequestConverterCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    mediapipe::Timestamp timestamp{0};

    static absl::Status parsePrompt(std::unique_ptr<ovms::LLMdata>& output, const rapidjson::Value::ConstMemberIterator& messagesIt) {
        if (!messagesIt->value.IsArray()) {
            return absl::Status(absl::StatusCode::kInvalidArgument, "\"messages\" have to be an array");
        }
        for (const auto& message : messagesIt->value.GetArray()) {
            if (!message.IsObject())
                return absl::Status(absl::StatusCode::kInvalidArgument, "\"messages\" array should contains only JSON objects");
            if (message.FindMember("content") == message.MemberEnd() || message.FindMember("role") == message.MemberEnd())
                return absl::Status(absl::StatusCode::kInvalidArgument, "\"message\" structure is invalid");
            if (!message["role"].IsString())
                return absl::Status(absl::StatusCode::kInvalidArgument, "\"role\" shave to be string");
            if (!message["content"].IsString())
                return absl::Status(absl::StatusCode::kInvalidArgument, "\"content\" shave to be string");
            output->prompt += message["content"].GetString() + std::string(" ");
        }
        output->prompt += "</s>";
        return absl::OkStatus();
    }

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "RequestConverterCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());

        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<ovms::LLMdata>();

        LOG(INFO) << "RequestConverterCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "RequestConverterCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "RequestConverterCalculator [Node: " << cc->NodeName() << "] Open start";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "RequestConverterCalculator [Node: " << cc->NodeName() << "] Process start";
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            return absl::OkStatus();
        }
        auto payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();
        auto messagesIt = payload.doc->FindMember("messages");
        if (messagesIt == payload.doc->MemberEnd()) {
            return absl::Status(absl::StatusCode::kInvalidArgument, "\"messages\" field is missing in JSON body");
        }
        std::unique_ptr<ovms::LLMdata> output(new ovms::LLMdata);

        auto status = parsePrompt(output, messagesIt);
        if (status != absl::OkStatus()) {
            return status;
        }
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), timestamp);

        timestamp = timestamp.NextAllowedInStream();
        return absl::OkStatus();
    }
};

const std::string RequestConverterCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string RequestConverterCalculator::OUTPUT_TAG_NAME{"LLM_DATA"};

REGISTER_CALCULATOR(RequestConverterCalculator);
}  // namespace mediapipe
