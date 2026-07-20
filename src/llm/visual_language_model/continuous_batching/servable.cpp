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

#include "servable.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "../../../config.hpp"
#include "../../../logging.hpp"
#include "../../../tokenize/tokenize_parser.hpp"
#include "../../text_utils.hpp"
#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif

namespace ovms {

absl::Status VisualLanguageModelServable::addRequestToPipeline(std::shared_ptr<ContinuousBatchingServableExecutionContext>& executionContext) {
    auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext);
    vlmExecutionContext->generationHandle = properties->pipeline->add_request(currentRequestId++,  // to be removed from API?
        vlmExecutionContext->inputRequest.promptText, vlmExecutionContext->inputRequest.inputImages,
        vlmExecutionContext->inputRequest.generationConfig);
    return absl::OkStatus();
}

absl::Status VisualLanguageModelServable::loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", payload.body);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
    // Parsed JSON is not guaranteed to be valid, we may reach this point via multipart content type request with no valid JSON parser
    if (payload.parsedJson->HasParseError()) {
        return absl::InvalidArgumentError("Non-json request received in text generation calculator");
    }
    if (payload.uri.find("/v3/v1/") != std::string::npos) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Endpoint {} is deprecated. Use /v1/ prefix instead.", payload.uri);
    }
    if (payload.uri == "/v3/chat/completions" || payload.uri == "/v3/v1/chat/completions" ||
        payload.uri == "/v1/chat/completions") {
        executionContext->endpoint = Endpoint::CHAT_COMPLETIONS;
    } else if (payload.uri == "/v3/responses" || payload.uri == "/v3/v1/responses" ||
               payload.uri == "/v1/responses") {
        executionContext->endpoint = Endpoint::RESPONSES;
    } else if (TokenizeParser::isTokenizeEndpoint(payload.uri)) {
        executionContext->endpoint = Endpoint::TOKENIZE;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. VLM Servable allowed only on /v[13]/chat/completions, /v[13]/responses, /v[13]/tokenize");
    }
    executionContext->payload = payload;
    return absl::OkStatus();
}

std::shared_ptr<GenAiServableExecutionContext> VisualLanguageModelServable::createExecutionContext() {
    return std::make_shared<VisualLanguageModelServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> VisualLanguageModelServable::getProperties() {
    return properties;
}

}  // namespace ovms
