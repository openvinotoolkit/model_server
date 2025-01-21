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
#include <algorithm>
#include <string>
#include <thread>

#include "../../../http_payload.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

using namespace std::chrono_literals;

namespace mediapipe {
/*
    Concatenates headers names and values(with no space), request body
    and rapidjson request body if exists.
*/
class OpenAIChatCompletionsMockCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string LOOPBACK_TAG_NAME;

    mediapipe::Timestamp timestamp{0};
    std::string body;
    std::shared_ptr<ovms::ClientConnection> client;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->Inputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
        cc->Outputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty() && cc->Inputs().Tag(LOOPBACK_TAG_NAME).IsEmpty()) {
            return absl::OkStatus();
        }
        if (!cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            auto data = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();
            // This calculator produces string so that it contains:
            // - URI
            // - Headers (kv pairs)
            // - Request body
            // - timestamps 0-8 (appended in cycles)
            this->body = data.uri + std::string{"\n"};
            for (auto header : data.headers) {
                this->body += header.first;
                this->body += header.second;
            }
            this->body += data.body;
            this->client = data.client;
            if (data.parsedJson != NULL) {
                rapidjson::StringBuffer buffer;
                buffer.Clear();
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                data.parsedJson->Accept(writer);
                this->body += buffer.GetString();
            }

            // Mock failing scenario
            if (data.body.find("ReturnError") != std::string::npos) {
                return absl::InvalidArgumentError("Returned error");
            }
        }

        if (client->isDisconnected()) {
            return absl::OkStatus();
        }

        this->body += std::to_string(timestamp.Value());

        // Fake workload
        // std::this_thread::sleep_for(600ms);

        /*
        TODO:
        Depending on JSON field stream true/false produce 1 packet or multiple packets

        //                                                                                         LLM Engine thread
        //                                                                                              |
        //                                                                                      ->      v           ->
        //  OVMS CORE -> DeserializatorCalc (?Type -> LLM Engine specific packets (params))     ->  LLM ClientCalc  ->  SerializatorCalc
        //                                                                                      ->     cycle        ->
        //                                              prompt: std::string
        //                                              params: std::map<key, any?>?
*/

        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string{this->body}, timestamp);

        // Once we find '8' in the string, we stop producing loopback packet meaning we end the loop
        if (std::find(this->body.begin(), this->body.end(), '8') == this->body.end()) {
            cc->Outputs().Tag(LOOPBACK_TAG_NAME).Add(new bool{true}, timestamp);
        }

        timestamp = timestamp.NextAllowedInStream();

        return absl::OkStatus();
    }
};
#pragma GCC diagnostic pop

// TODO: Names to be decided
const std::string OpenAIChatCompletionsMockCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string OpenAIChatCompletionsMockCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};
const std::string OpenAIChatCompletionsMockCalculator::LOOPBACK_TAG_NAME{"LOOPBACK"};

REGISTER_CALCULATOR(OpenAIChatCompletionsMockCalculator);
}  // namespace mediapipe
