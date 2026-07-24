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

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4251 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246 6313)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rapidjson/document.h>
#include "openvino/genai/text_streamer.hpp"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../http_payload.hpp"
#include "../sse_utils.hpp"
#include "apis/openai_api_handler.hpp"
#include "io_processing/chat_template/caps.hpp"
#include "io_processing/base_generation_config_builder.hpp"
#include "io_processing/input_processor_context.hpp"
#include "io_processing/input_request.hpp"
#include "runtime_chat_template.hpp"
#if (PYTHON_DISABLE == 0)
#include "py_jinja_template_processor.hpp"
#endif

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

enum class GenerationPhase {
    INPUT_TOKEN_PROCESSING,
    OUTPUT_TOKEN_PROCESSING,
};

enum class ChatTemplateMode {
    MINJA,  // Use GenAI's apply_chat_template (minja-based)
    JINJA,  // Use Python Jinja2 module for chat template processing
};

// Thread-safe channel for parsed streaming deltas.
// The producer (OVMSTextStreamer callback, possibly on a background executor thread)
// calls push(); the consumer (preparePartialResponse, always on the calculator thread)
// calls waitForData() then drain(). For CB/stateful paths both sides run on the same
// thread, so the mutex is acquired but uncontested.
struct DeltaChannel {
    // Push a delta from any thread (streamer callback).
    // When isLast is true, also marks the channel complete atomically so consumers
    // always see the final document and the completion flag in the same observation.
    void push(rapidjson::Document delta, bool isLast = false) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_deltas.push_back(std::move(delta));
            if (isLast)
                m_complete = true;
        }
        m_cv.notify_one();
    }

    // Signal that no more deltas will be pushed (generation complete or cancelled).
    // May be called from any thread. Also acts as a safety-net for paths where
    // push(delta, isLast=true) may not fire (e.g. client disconnection mid-stream).
    void signalComplete() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_complete = true;
        }
        m_cv.notify_one();
    }

    // Block until at least one delta is available or signalComplete() has been called.
    // For CB paths this returns immediately since data is already present.
    void waitForData() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] { return !m_deltas.empty() || m_complete; });
    }

    // Move all pending deltas out atomically. Returns an empty vector if none pending.
    std::vector<rapidjson::Document> drain() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<rapidjson::Document> result;
        result.swap(m_deltas);
        return result;
    }

    // Returns true after signalComplete() has been called.
    bool complete() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_complete;
    }

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<rapidjson::Document> m_deltas;
    bool m_complete = false;
};

struct GenAiServableExecutionContext {
    // Common API related members
    HttpPayload payload;
    Endpoint endpoint;
    std::shared_ptr<OpenAIApiHandler> apiHandler;
    // Populated in parseRequest(); carries all GenAI inputs including the generation config.
    InputRequest inputRequest;
    // Required for generating output and handle request on the calculator side
    std::vector<ov::genai::GenerationOutput> generationOutputs;
    std::string response;
    std::shared_ptr<ov::genai::TextStreamer> textStreamer;
    bool sendLoopbackSignal = false;
    bool lifecyclePrimed = false;  // true once RESPONSES lifecycle events have been primed
    DeltaChannel deltaChannel;     // thread-safe delta queue used by all streaming paths
    GenerationPhase generationPhase = GenerationPhase::INPUT_TOKEN_PROCESSING;
};

struct ExtraGenerationInfo {
    std::string bosTokenFromTokenizer;
    std::string bosTokenIdFromTokenizer;
    std::string eosTokenFromTokenizer;
    std::string eosTokenIdFromTokenizer;
    std::string chatTemplateFromTokenizer;
    std::string chatTemplateDirectory;
    bool isGgufModel;
};

struct GenAiServableProperties {
    // General configuration
    std::string modelsPath;
    ov::genai::GenerationConfig baseGenerationConfig;
    std::string toolParserName;
    std::string reasoningParserName;
    std::string device;
    ov::AnyMap pluginConfig;
    ov::AnyMap tokenizerPluginConfig;
    bool enableToolGuidedGeneration = false;
#if (PYTHON_DISABLE == 0)
    ChatTemplateMode chatTemplateMode = ChatTemplateMode::JINJA;
#else
    ChatTemplateMode chatTemplateMode = ChatTemplateMode::MINJA;
#endif
    // Chat template analysis
    ChatTemplateCaps chatTemplateCaps;
    // Sampling
    DecodingMethod decodingMethod;
    std::optional<uint32_t> maxTokensLimit;
    std::optional<uint32_t> maxModelLength;
    uint32_t bestOfLimit;
    // Text processing utilities
    ov::genai::Tokenizer tokenizer;
    // Specific pipeline properties
    bool eagle3Mode = false;
    // Controls which steps InputProcessor builds for this servable type.
    // Aggregated per-deployment context for InputProcessor.
    InputProcessorContext inputProcessorContext;
    PreparedRuntimeChatTemplate preparedRuntimeChatTemplate;

#if (PYTHON_DISABLE == 0)
    PyJinjaTemplateProcessor templateProcessor;
#endif

    bool hasPreparedPyTemplateProcessor() const {
#if (PYTHON_DISABLE == 0)
        return templateProcessor.chatTemplate != nullptr;
#else
        return false;
#endif
    }

    PyJinjaTemplateProcessor* getPreparedPyTemplateProcessorOrNull() {
#if (PYTHON_DISABLE == 0)
        return hasPreparedPyTemplateProcessor() ? &templateProcessor : nullptr;
#else
        return nullptr;
#endif
    }
};

class GenAiServable {
public:
    GenAiServable() = default;
    GenAiServable(GenAiServable&&) = default;
    GenAiServable& operator=(GenAiServable&&) = default;
    GenAiServable(const GenAiServable&) = delete;
    GenAiServable& operator=(const GenAiServable&) = delete;
    virtual ~GenAiServable() = default;

    void determineDecodingMethod();

    // ----------- Tokenize scenario ------------
    /*
    processTokenizeRequest method implements tokenization of the input text provided in executionContext payload.
    Implementation fills executionContext response field with serialized tokenization result wrapped in json.
    */
    absl::Status processTokenizeRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext);

    // ----- Interface for derived classes -----

    /*
    loadRequest method implementation MUST fill executionContext payload and endpoint fields.
    Base implementation does that and makes sure URI matches either chat/completions or completions endpoint.
    After endpoint routing, calls validateEndpoint() which derived classes can override to reject
    unsupported endpoints (e.g. VLM/Omni reject /completions).
    */
    virtual absl::Status loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const HttpPayload& payload);

    // Override to reject endpoints not supported by this servable.
    // Called after endpoint is determined. Return non-OK to reject.
    virtual absl::Status validateEndpoint(Endpoint endpoint) const {
        (void)endpoint;
        return absl::OkStatus();
    }

    // Creates execution context for the request
    virtual std::shared_ptr<GenAiServableExecutionContext> createExecutionContext() = 0;

    // Returns properties of the servable
    virtual std::shared_ptr<GenAiServableProperties> getProperties() = 0;

    /*
    parseRequest method implementation MUST fill executionContext apiHandler field and parse request.
    For streaming requests, it MUST initialize the textStreamer field of executionContext.
    Base implementation creates OpenAIChatCompletionsHandler and calls its parseRequest method.
    Additionally it initializes textStreamer for streaming requests.
    */
    virtual absl::Status parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext);

    /*
    prepareInputs method implementation MUST fill executionContext inputRequest.inputIds field.
    Base implementation applies chat template to the payload body and encodes it with tokenizer.
    */
    virtual absl::Status prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext);

    /*
    validateInputCompatibility checks whether the request input is compatible with this servable type.
    Called from prepareInputs before the InputProcessor chain runs.
    Base implementation rejects image_url content for non-VLM (text-only) servables.
    Derived classes may override to add or relax constraints.
    */
    virtual absl::Status validateInputCompatibility(std::shared_ptr<GenAiServableExecutionContext>& executionContext);

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
using GenAiServableMap = std::unordered_map<std::string, std::shared_ptr<GenAiServable>>;
void logRequestDetails(const HttpPayload& payload);
}  // namespace ovms
