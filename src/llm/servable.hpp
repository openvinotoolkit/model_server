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
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/text_streamer.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../http_payload.hpp"
#include "apis/openai_completions.hpp"
#include "text_processor.hpp"

namespace ovms {
// Some pipelines internals rely on request_id, so for now we provide increasing ID
static std::atomic<uint64_t> currentRequestId = 0;

/*
GenAiServable support.

This header contains the interface for GenAiServable and its properties and execution context.
All classes in this file should not be instantiated, but rather extended by derived classes.

GenAiServable is an interface for derived classes that implement logic for specific types of pipelines. 
It uses both GenAiServableProperties and GenAiServableExecutionContext (or rather their extensions) to facilitate implementation.

GenAiServableProperties is a container for data initialized when the servable is loaded and it is reused 
for every request, in contrary to GenAiServableExecutionContext that is created multiple times during servable lifespan.
GenAiServableProperties are initialized by servable initializer (see servable_initializer.hpp).

GenAiServableExecutionContext is a container that holds required data throughout request processing.
It is created by createExecutionContext method of GenAiServable in HttpLLMCalculator, which then uses it when calling GenAiServable methods.
Instance of this class is created for each request and is passed through multiple methods of GenAiServable according to HttpLLMCalculator::Process implementation.
Note that GenAiServableExecutionContext pointer is the only parameter most of the GenAiServable methods take.
*/

struct GenAiServableExecutionContext {
    // Common API related members
    ovms::HttpPayload payload;
    Endpoint endpoint;
    std::shared_ptr<OpenAIChatCompletionsHandler> apiHandler;
    // Single tensor with inputIds for the model. This is considered general for all pipelines,
    // but depending on particular pipeline implementation it might be not required or on the other hand, insufficient.
    ov::Tensor inputIds;
    // Required for generating output and handle request on the calculator side
    std::vector<ov::genai::GenerationOutput> generationOutputs;
    std::string response;
    std::shared_ptr<ov::genai::TextStreamer> textStreamer;
    bool sendLoopbackSignal = false;
    std::string lastStreamerCallbackOutput;
};

struct GenAiServableProperties {
    // General configuration
    std::string modelsPath;
    std::string device;
    ov::AnyMap pluginConfig;
    ov::AnyMap tokenizerPluginConfig;
    // Sampling limits
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit;
    // Text processing utilities
    ov::genai::Tokenizer tokenizer;
    TextProcessor textProcessor;
    std::optional<uint32_t> maxModelLength;
};

class GenAiServable {
public:
    GenAiServable() = default;
    GenAiServable(GenAiServable&&) = default;
    GenAiServable& operator=(GenAiServable&&) = default;
    GenAiServable(const GenAiServable&) = delete;
    GenAiServable& operator=(const GenAiServable&) = delete;
    virtual ~GenAiServable() = default;

    // ----- Interface for derived classes -----

    /*
    loadRequest method implementation MUST fill executionContext payload and endpoint fields.
    Base implementation does that and makes sure URI matches either chat/completions or completions endpoint.
    */
    virtual absl::Status loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload);

    // Creates execution context for the request
    virtual std::shared_ptr<GenAiServableExecutionContext> createExecutionContext() = 0;

    // Returns properties of the servable
    virtual std::shared_ptr<GenAiServableProperties> getProperties() = 0;

    /*
    parseRequest method implementation MUST fill executionContext apiHandler field and parse request.
    For streaming requests, it MUST initialize textStreamer and lastStreamerCallbackOutput fields of executionContext.
    Base implementation creates OpenAIChatCompletionsHandler and calls its parseRequest method.
    Additionally it initializes textStreamer and lastStreamerCallbackOutput for streaming requests.
    */
    virtual absl::Status parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext);

    /*
    prepareInputs method implementation MUST fill executionContext inputIds field.
    Base implementation applies chat template to the payload body and encodes it with tokenizer.
    */
    virtual absl::Status prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext);

    /*
    scheduleExecution method should implement any necessary queueing mechanism or start asynchronous execution.
    Execution context in such case may contain handles, futures or other objects that will be used to track the execution.
    If none of that is necessary, the implementation can simply return OK status.
    Implementation should fill executionContext with data required by the read methods.
    */
    virtual absl::Status scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) = 0;

    // ----------- Unary scenario ------------

    /*
    readCompleteExecutionResults method should implement reading the results of the execution in a unary request scenario.
    If interacting with the pipeline is not asynchronous and does not require any queuing (schedulePipelineExecution implementation is essenatially void),
    then this method should run entire execution.
    Implementation MUST fill executionContext generationOutputs field.
    */
    virtual absl::Status readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) = 0;

    /*
    prepareCompleteResponse method should implement preparing the response for unary request scenario from executionContext generationOutputs.
    Implementation MUST fill executionContext response field.
    Base implementation serializes the response using apiHandler.
    */
    virtual absl::Status prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext);

    // ----------- Streaming scenario ------------

    /*
    readPartialExecutionResults method should implement reading the results of the execution in a streaming request scenario.
    If interacting with the pipeline is not asynchronous and does not require any queuing (schedulePipelineExecution implementation is essenatially void),
    then this method should run entire execution.
    Implementation MUST fill executionContext generationOutputs field.
    */
    virtual absl::Status readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) = 0;

    /*
    preparePartialResponse method should implement preparing the response for streaming request scenario from executionContext generationOutputs.
    This method also handles loopback (keep processing when stream is not finished or end otherwise). Depending on generated tokens, response might be empty string.
    In such case, calculator will not send it down the graph.
    Implementation MUST fill executionContext response and sendLoopbackSignal fields.
    Base implementation uses textStreamer to create text chunk, attempts to serialize it, and sets sendLoopbackSignal according to generation status.
    */
    virtual absl::Status preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext);
};
std::string wrapTextInServerSideEventMessage(const std::string& text);
using GenAiServableMap = std::unordered_map<std::string, std::shared_ptr<GenAiServable>>;
}  // namespace ovms
