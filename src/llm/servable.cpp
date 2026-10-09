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
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246 6313)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../config.hpp"
#include "../http_payload.hpp"
#include "../logging.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../profiler.hpp"
#include "apis/openai_completions.hpp"
#include "apis/openai_responses.hpp"
#include "io_processing/generation_config_builder.hpp"
#include "io_processing/input_processor.hpp"
#include "ovms_text_streamer.hpp"
#include "servable.hpp"
#include "text_utils.hpp"
#include "../tokenize/tokenize_parser.hpp"

namespace ovms {
namespace {
void finishTextStreamer(const std::shared_ptr<ov::genai::TextStreamer>& streamer,
    ov::genai::GenerationFinishReason reason) {
    if (auto ovmsStreamer = std::dynamic_pointer_cast<OVMSTextStreamer>(streamer))
        ovmsStreamer->end(reason);
    else
        streamer->end();
}
}  // namespace
namespace {

constexpr const char* SESSION_HEADER = "x-ovms-session-id";

std::string asciiLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::optional<std::string> getSessionIdHeader(const std::unordered_map<std::string, std::string>& headers) {
    for (const auto& [name, value] : headers) {
        if (asciiLower(name) == SESSION_HEADER)
            return value;
    }
    return std::nullopt;
}

std::string serializeJson(const rapidjson::Document& doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return std::string(buffer.GetString(), buffer.GetSize());
}

std::string utcNowIso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return os.str();
}

uint64_t parseEnvUnsigned(const char* name, uint64_t fallback, uint64_t minimum) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
        return fallback;
    try {
        const uint64_t value = std::stoull(raw);
        if (value >= minimum)
            return value;
    } catch (...) {
    }
    SPDLOG_LOGGER_WARN(llm_calculator_logger, "Ignoring invalid {}='{}'; using {}", name, raw, fallback);
    return fallback;
}

std::shared_ptr<SessionStateStore> processSessionStateStore() {
    static std::shared_ptr<SessionStateStore> store = SessionStateStore::fromEnvironment();
    return store;
}

}  // namespace

class SessionStateStore::Impl {
public:
    struct Manifest {
        uint32_t seed = 0;
        uint64_t nextTurn = 1;
        std::string lastAccess;
        std::string model;
    };

    struct CacheEntry {
        Manifest manifest;
        std::list<std::string>::iterator lruIt;
    };

    std::filesystem::path root;
    size_t cacheEntries;
    uint64_t maxBytes;
    size_t maxRequestBytes;
    uint64_t bytesUsed = 0;
    std::mutex mutex;
    std::list<std::string> lru;
    std::unordered_map<std::string, CacheEntry> cache;

    Impl(std::string rootDirectory, size_t cacheEntries, uint64_t maxBytes, size_t maxRequestBytes) :
        root(std::move(rootDirectory)),
        cacheEntries(std::max<size_t>(1, cacheEntries)),
        maxBytes(std::max<uint64_t>(1024 * 1024, maxBytes)),
        maxRequestBytes(std::max<size_t>(1024, maxRequestBytes)) {
        if (root.empty())
            return;
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
        if (ec) {
            SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Cannot create OVMS session store '{}': {}", root.string(), ec.message());
            root.clear();
            return;
        }
        for (std::filesystem::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->is_regular_file(ec))
                bytesUsed += static_cast<uint64_t>(it->file_size(ec));
        }
        if (ec)
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Session store size scan incomplete for '{}': {}", root.string(), ec.message());
    }

    bool enabled() const {
        return !root.empty();
    }

    static bool validSessionId(const std::string& id) {
        if (id.empty() || id.size() > 128)
            return false;
        return std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '.' || c == '_' || c == '-';
        });
    }

    std::filesystem::path sessionDir(const std::string& id) const {
        return root / id;
    }

    std::filesystem::path manifestPath(const std::string& id) const {
        return sessionDir(id) / "manifest.json";
    }

    static std::string turnName(uint64_t index) {
        std::ostringstream os;
        os << std::setw(12) << std::setfill('0') << index;
        return os.str();
    }

    static std::string readText(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    uint64_t currentFileSize(const std::filesystem::path& path) const {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec))
            return 0;
        const auto size = std::filesystem::file_size(path, ec);
        return ec ? 0 : static_cast<uint64_t>(size);
    }

    absl::Status checkQuota(const std::vector<std::pair<std::filesystem::path, std::string>>& writes) const {
        int64_t delta = 0;
        for (const auto& [path, content] : writes) {
            const uint64_t oldSize = currentFileSize(path);
            const uint64_t newSize = static_cast<uint64_t>(content.size());
            if (newSize >= oldSize)
                delta += static_cast<int64_t>(newSize - oldSize);
            else
                delta -= static_cast<int64_t>(oldSize - newSize);
        }
        if (delta > 0 && (bytesUsed > maxBytes || static_cast<uint64_t>(delta) > maxBytes - bytesUsed))
            return absl::ResourceExhaustedError("OVMS session store byte quota exceeded");
        return absl::OkStatus();
    }

    absl::Status atomicWrite(const std::filesystem::path& path, const std::string& content) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec)
            return absl::InternalError("cannot create session journal directory: " + ec.message());

        const uint64_t oldSize = currentFileSize(path);
        std::filesystem::path tmp = path;
        tmp += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out.good())
                return absl::InternalError("cannot open temporary session journal file");
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
            out.flush();
            if (!out.good()) {
                out.close();
                std::filesystem::remove(tmp, ec);
                return absl::InternalError("cannot write temporary session journal file");
            }
        }

#ifdef _WIN32
        if (!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            const DWORD err = GetLastError();
            std::filesystem::remove(tmp, ec);
            return absl::InternalError("cannot atomically replace session journal file, Win32 error " + std::to_string(err));
        }
#else
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            return absl::InternalError("cannot atomically replace session journal file: " + ec.message());
        }
#endif

        const uint64_t newSize = static_cast<uint64_t>(content.size());
        if (newSize >= oldSize)
            bytesUsed += newSize - oldSize;
        else
            bytesUsed -= std::min<uint64_t>(bytesUsed, oldSize - newSize);
        return absl::OkStatus();
    }

    static std::string manifestJson(const std::string& id, const Manifest& manifest) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.Key("schema_version"); writer.Uint(1);
        writer.Key("session_id"); writer.String(id.c_str(), static_cast<rapidjson::SizeType>(id.size()));
        writer.Key("seed"); writer.Uint(manifest.seed);
        writer.Key("next_turn"); writer.Uint64(manifest.nextTurn);
        writer.Key("last_access"); writer.String(manifest.lastAccess.c_str(), static_cast<rapidjson::SizeType>(manifest.lastAccess.size()));
        if (!manifest.model.empty()) {
            writer.Key("model"); writer.String(manifest.model.c_str(), static_cast<rapidjson::SizeType>(manifest.model.size()));
        }
        writer.EndObject();
        return std::string(buffer.GetString(), buffer.GetSize());
    }

    absl::StatusOr<Manifest> loadManifestFromDisk(const std::string& id) {
        const auto path = manifestPath(id);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec))
            return Manifest{};
        const std::string text = readText(path);
        rapidjson::Document doc;
        doc.Parse(text.c_str(), text.size());
        if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("seed") || !doc["seed"].IsUint() ||
            !doc.HasMember("next_turn") || !doc["next_turn"].IsUint64())
            return absl::DataLossError("invalid OVMS session manifest for '" + id + "'");
        Manifest manifest;
        manifest.seed = doc["seed"].GetUint();
        manifest.nextTurn = doc["next_turn"].GetUint64();
        if (doc.HasMember("last_access") && doc["last_access"].IsString())
            manifest.lastAccess.assign(doc["last_access"].GetString(), doc["last_access"].GetStringLength());
        if (doc.HasMember("model") && doc["model"].IsString())
            manifest.model.assign(doc["model"].GetString(), doc["model"].GetStringLength());
        return manifest;
    }

    void putCache(const std::string& id, const Manifest& manifest) {
        auto it = cache.find(id);
        if (it != cache.end()) {
            lru.erase(it->second.lruIt);
            cache.erase(it);
        }
        lru.push_front(id);
        cache.emplace(id, CacheEntry{manifest, lru.begin()});
        while (cache.size() > cacheEntries) {
            const std::string evict = lru.back();
            lru.pop_back();
            cache.erase(evict);
        }
    }

    absl::StatusOr<Manifest> loadManifest(const std::string& id) {
        auto it = cache.find(id);
        if (it != cache.end()) {
            Manifest manifest = it->second.manifest;
            lru.erase(it->second.lruIt);
            lru.push_front(id);
            it->second.lruIt = lru.begin();
            return manifest;
        }
        auto loaded = loadManifestFromDisk(id);
        if (!loaded.ok())
            return loaded.status();
        putCache(id, *loaded);
        return *loaded;
    }

    static uint32_t generateSeed() {
        static thread_local std::mt19937 rng{std::random_device{}()};
        uint32_t seed = 0;
        while (seed == 0)
            seed = rng();
        return seed;
    }

    static std::string generationConfigJson(const ov::genai::GenerationConfig& config, const std::string& toolChoice) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.Key("rng_seed"); writer.Uint64(config.rng_seed);
        writer.Key("temperature"); writer.Double(config.temperature);
        writer.Key("top_p"); writer.Double(config.top_p);
        writer.Key("top_k"); writer.Uint64(config.top_k);
        writer.Key("min_p"); writer.Double(config.min_p);
        writer.Key("do_sample"); writer.Bool(config.do_sample);
        writer.Key("num_beams"); writer.Uint64(config.num_beams);
        writer.Key("max_new_tokens"); writer.Uint64(config.max_new_tokens);
        writer.Key("max_length"); writer.Uint64(config.max_length);
        writer.Key("ignore_eos"); writer.Bool(config.ignore_eos);
        writer.Key("repetition_penalty"); writer.Double(config.repetition_penalty);
        writer.Key("tool_choice"); writer.String(toolChoice.c_str(), static_cast<rapidjson::SizeType>(toolChoice.size()));
        writer.Key("structured_output_active"); writer.Bool(config.structured_output_config.has_value());
        writer.EndObject();
        return std::string(buffer.GetString(), buffer.GetSize());
    }
};

SessionStateStore::SessionStateStore(std::string rootDirectory, size_t cacheEntries, uint64_t maxBytes, size_t maxRequestBytes) :
    impl(std::make_unique<Impl>(std::move(rootDirectory), cacheEntries, maxBytes, maxRequestBytes)) {}

SessionStateStore::~SessionStateStore() = default;
SessionStateStore::SessionStateStore(SessionStateStore&&) noexcept = default;
SessionStateStore& SessionStateStore::operator=(SessionStateStore&&) noexcept = default;

std::shared_ptr<SessionStateStore> SessionStateStore::fromEnvironment() {
    const char* root = std::getenv("OVMS_SESSION_STORE_DIR");
    const std::string rootDirectory = (root == nullptr) ? std::string{} : std::string(root);
    const size_t cacheEntries = static_cast<size_t>(parseEnvUnsigned("OVMS_SESSION_CACHE_ENTRIES", DEFAULT_CACHE_ENTRIES, 1));
    const uint64_t maxBytes = parseEnvUnsigned("OVMS_SESSION_MAX_BYTES", DEFAULT_MAX_BYTES, 1024 * 1024);
    const size_t maxRequestBytes = static_cast<size_t>(parseEnvUnsigned("OVMS_SESSION_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES, 1024));
    if (rootDirectory.empty()) {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "session-state: store disabled; OVMS_SESSION_STORE_DIR is unset");
    } else {
        SPDLOG_LOGGER_INFO(llm_calculator_logger,
            "session-state: constructing store from OVMS_SESSION_STORE_DIR={}",
            rootDirectory);
    }
    return std::make_shared<SessionStateStore>(rootDirectory, cacheEntries, maxBytes, maxRequestBytes);
}

bool SessionStateStore::enabled() const {
    return impl && impl->enabled();
}

absl::StatusOr<SessionTurnContext> SessionStateStore::beginTurn(
    const std::string& sessionId,
    const std::string& rawBody,
    rapidjson::Document& effectiveDocument) {
    if (!enabled())
        return SessionTurnContext{};
    if (!Impl::validSessionId(sessionId))
        return absl::InvalidArgumentError("invalid OVMS session id");
    if (rawBody.size() > impl->maxRequestBytes)
        return absl::ResourceExhaustedError("OVMS session request exceeds configured journal request limit");
    if (!effectiveDocument.IsObject())
        return absl::InvalidArgumentError("session persistence requires a JSON object request");

    std::lock_guard<std::mutex> lock(impl->mutex);
    auto manifestResult = impl->loadManifest(sessionId);
    if (!manifestResult.ok())
        return manifestResult.status();
    auto manifest = *manifestResult;

    std::optional<uint32_t> requestedSeed;
    if (effectiveDocument.HasMember("seed")) {
        if (!effectiveDocument["seed"].IsUint())
            return absl::InvalidArgumentError("session seed must be an unsigned 32-bit integer");
        requestedSeed = effectiveDocument["seed"].GetUint();
    }

    const bool existing = std::filesystem::exists(impl->manifestPath(sessionId));
    if (!existing) {
        manifest.seed = requestedSeed.value_or(Impl::generateSeed());
        manifest.nextTurn = 1;
    } else if (requestedSeed.has_value() && requestedSeed.value() != manifest.seed) {
        return absl::InvalidArgumentError(
            "session seed conflict: persisted=" + std::to_string(manifest.seed) +
            ", requested=" + std::to_string(requestedSeed.value()));
    }

    auto& allocator = effectiveDocument.GetAllocator();
    if (effectiveDocument.HasMember("seed"))
        effectiveDocument["seed"].SetUint(manifest.seed);
    else {
        rapidjson::Value seedName("seed", allocator);
        effectiveDocument.AddMember(seedName, manifest.seed, allocator);
    }

    if (effectiveDocument.HasMember("model") && effectiveDocument["model"].IsString())
        manifest.model.assign(effectiveDocument["model"].GetString(), effectiveDocument["model"].GetStringLength());
    manifest.lastAccess = utcNowIso8601();

    const uint64_t turnIndex = manifest.nextTurn++;
    const auto turnDir = impl->sessionDir(sessionId) / "turns" / Impl::turnName(turnIndex);
    const auto rawPath = turnDir / "raw-request.json";
    const auto effectivePath = turnDir / "effective-request.json";
    const auto manifestPath = impl->manifestPath(sessionId);
    const std::string effectiveBody = serializeJson(effectiveDocument);
    const std::string manifestBody = Impl::manifestJson(sessionId, manifest);

    const std::vector<std::pair<std::filesystem::path, std::string>> writes{
        {rawPath, rawBody},
        {effectivePath, effectiveBody},
        {manifestPath, manifestBody},
    };
    auto quotaStatus = impl->checkQuota(writes);
    if (!quotaStatus.ok())
        return quotaStatus;
    for (const auto& [path, content] : writes) {
        auto status = impl->atomicWrite(path, content);
        if (!status.ok())
            return status;
    }
    impl->putCache(sessionId, manifest);

    SessionTurnContext turn;
    turn.active = true;
    turn.sessionId = sessionId;
    turn.turnIndex = turnIndex;
    turn.seed = manifest.seed;
    return turn;
}

absl::Status SessionStateStore::recordGenerationConfig(
    const SessionTurnContext& turn,
    const ov::genai::GenerationConfig& config,
    const std::string& toolChoice) {
    if (!turn.active || !enabled())
        return absl::OkStatus();
    if (!Impl::validSessionId(turn.sessionId))
        return absl::InvalidArgumentError("invalid OVMS session id");

    std::lock_guard<std::mutex> lock(impl->mutex);
    const auto path = impl->sessionDir(turn.sessionId) / "turns" / Impl::turnName(turn.turnIndex) / "generation-config.json";
    const std::string content = Impl::generationConfigJson(config, toolChoice);
    auto quotaStatus = impl->checkQuota({{path, content}});
    if (!quotaStatus.ok())
        return quotaStatus;
    return impl->atomicWrite(path, content);
}

double calculatePrefillSpeed(size_t inputTokenCount, double ttftMs) {
    return ttftMs > 0.0 ? (1000.0 * inputTokenCount) / ttftMs : 0.0;
}

void GenAiServable::determineDecodingMethod() {
    getProperties()->decodingMethod = DecodingMethod::STANDARD;
    auto& pluginConfig = getProperties()->pluginConfig;
    if (pluginConfig.find("draft_model") != pluginConfig.end()) {
        using DS = GenAiServableProperties::DraftModelStrategy;
        switch (getProperties()->draftModelStrategy) {
        case DS::EAGLE3:
            getProperties()->decodingMethod = DecodingMethod::EAGLE3;
            break;
        case DS::DFLASH:
            getProperties()->decodingMethod = DecodingMethod::DFLASH;
            break;
        case DS::MTP:
            getProperties()->decodingMethod = DecodingMethod::MTP;
            break;
        default:
            getProperties()->decodingMethod = DecodingMethod::FAST_DRAFT;
            break;
        }
    }
    auto it = pluginConfig.find("prompt_lookup");
    if (it != pluginConfig.end() && it->second.as<bool>() == true) {
        getProperties()->decodingMethod = DecodingMethod::PROMPT_LOOKUP;
    }
}

absl::Status GenAiServable::loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
    if (spdlog::default_logger_raw()->level() <= spdlog::level::debug) {
        logRequestDetails(payload);
    }
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
    } else if (payload.uri == "/v3/completions" || payload.uri == "/v3/v1/completions" ||
               payload.uri == "/v1/completions") {
        executionContext->endpoint = Endpoint::COMPLETIONS;
    } else if (payload.uri == "/v3/responses" || payload.uri == "/v3/v1/responses" ||
               payload.uri == "/v1/responses") {
        executionContext->endpoint = Endpoint::RESPONSES;
    } else if (TokenizeParser::isTokenizeEndpoint(payload.uri)) {
        executionContext->endpoint = Endpoint::TOKENIZE;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v[13]/chat/completions, /v[13]/completions, /v[13]/responses, /v[13]/tokenize");
    }
    auto endpointStatus = validateEndpoint(executionContext->endpoint);
    if (!endpointStatus.ok()) {
        return endpointStatus;
    }
    executionContext->payload = payload;
    // All legacy parsers override parseRequest; session setup must precede dispatch.
    if (executionContext->endpoint == Endpoint::TOKENIZE)
        return absl::OkStatus();
    auto sessionStore = processSessionStateStore();
    executionContext->sessionTurn = {};
    auto sessionId = getSessionIdHeader(executionContext->payload.headers);
    const char* storeDir = std::getenv("OVMS_SESSION_STORE_DIR");
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
        "session-state: uri={} header_count={} session_present={} store_enabled={} store_path={}",
        payload.uri,
        payload.headers.size(),
        sessionId.has_value(),
        sessionStore->enabled(),
        storeDir ? storeDir : "<unset>");
    if (sessionId.has_value()) {
        if (sessionStore->enabled()) {
            auto turn = sessionStore->beginTurn(sessionId.value(), executionContext->payload.body, *executionContext->payload.parsedJson);
            if (!turn.ok()) {
                SPDLOG_LOGGER_ERROR(llm_calculator_logger,
                    "session-state: beginTurn failed status={}",
                    turn.status().ToString());
                return turn.status();
            }
            executionContext->sessionTurn = std::move(*turn);
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
                "session-state: beginTurn turn={} seed={} store_path={}",
                executionContext->sessionTurn.turnIndex,
                executionContext->sessionTurn.seed,
                storeDir ? storeDir : "<unset>");
        } else {
            static std::once_flag warningOnce;
            std::call_once(warningOnce, []() {
                SPDLOG_LOGGER_WARN(llm_calculator_logger,
                    "X-OVMS-Session-ID received but OVMS_SESSION_STORE_DIR is not configured; session persistence is disabled");
            });
        }
    }

    return absl::OkStatus();
}

absl::Status GenAiServable::processTokenizeRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    ovms::TokenizeRequest tokenizeRequest;
    auto status = ovms::TokenizeParser::parseTokenizeRequest(*executionContext->payload.parsedJson, tokenizeRequest);
    if (status != absl::OkStatus()) {
        return status;
    }

    ov::genai::TokenizedInputs tokens;

    if (auto strings = std::get_if<std::vector<std::string>>(&tokenizeRequest.input)) {
        tokens = getProperties()->tokenizer.encode(*strings, tokenizeRequest.parameters);
        RET_CHECK(tokens.input_ids.get_shape().size() == 2);
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLM tokenize input is of not supported type");
        return absl::InvalidArgumentError("Input should be string or array of strings");
    }

    StringBuffer responseBuffer;
    auto responseStatus = ovms::TokenizeParser::parseTokenizeResponse(responseBuffer, tokens, tokenizeRequest.parameters);

    if (!responseStatus.ok()) {
        return responseStatus;
    }

    executionContext->response = responseBuffer.GetString();

    return absl::OkStatus();
}

absl::Status GenAiServable::parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    try {
        if (executionContext->endpoint == Endpoint::RESPONSES) {
            executionContext->apiHandler = std::make_shared<OpenAIResponsesHandler>(*executionContext->payload.parsedJson,
                executionContext->endpoint,
                std::chrono::system_clock::now(),
                getProperties()->tokenizer,
                getProperties()->toolParserName,
                getProperties()->reasoningParserName);
        } else {
            executionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*executionContext->payload.parsedJson,
                executionContext->endpoint,
                std::chrono::system_clock::now(),
                getProperties()->tokenizer,
                getProperties()->toolParserName,
                getProperties()->reasoningParserName);
        }
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to create API handler: {}", e.what());
        return absl::InvalidArgumentError(std::string("Failed to create API handler: ") + e.what());
    }
    auto& config = ovms::Config::instance();

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->maxModelLength, config.getServerSettings().allowedLocalMediaPath, config.getServerSettings().allowedMediaDomains);
    if (!status.ok()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse request: {}", status.message());
        return status;
    }

    {
        auto ovmsCallback = [& ctx = *executionContext](Delta delta, bool isLast) -> ov::genai::StreamingStatus {
            ctx.deltaChannel.push(std::move(delta), isLast);
            return ov::genai::StreamingStatus::RUNNING;
        };
        ov::AnyMap streamerConfig;
        if (!executionContext->apiHandler->getRequest().skipSpecialTokens) {
            streamerConfig.insert(ov::genai::skip_special_tokens(false));
        }
        executionContext->textStreamer = std::make_shared<OVMSTextStreamer>(
            getProperties()->tokenizer,
            executionContext->apiHandler->getOutputParser(),
            executionContext->apiHandler->areToolsAvailable(),
            std::move(ovmsCallback),
            streamerConfig);
    }
    GenerationConfigBuilder configBuilder(getProperties()->baseGenerationConfig,
        getProperties()->toolParserName,
        getProperties()->enableToolGuidedGeneration,
        getProperties()->decodingMethod);
    auto inputRequestResult = executionContext->apiHandler->extractInputRequest(configBuilder);
    if (!inputRequestResult.ok()) {
        return inputRequestResult.status();
    }
    executionContext->inputRequest = std::move(*inputRequestResult);

    return absl::OkStatus();
}

absl::Status GenAiServable::validateInputCompatibility(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    // LM servables reject requests containing image content. Images are preserved
    // as JsonContainer arrays in chatHistory. Reject if any message's content array
    // contains an image_url entry.
    if (!getProperties()->inputProcessorContext.config.isVLM &&
        std::holds_alternative<ov::genai::ChatHistory>(executionContext->inputRequest.input)) {
        const auto& ch = std::get<ov::genai::ChatHistory>(executionContext->inputRequest.input);
        for (size_t i = 0; i < ch.size(); i++) {
            const auto content = ch[i]["content"];
            if (content.is_array()) {
                for (size_t j = 0; j < content.size(); j++) {
                    if (content[j]["type"].as_string().value_or("") == "image_url") {
                        return absl::Status(absl::StatusCode::kInvalidArgument, "This servable supports only text input, but image_url has been provided");
                    }
                }
            }
        }
    }
    return absl::OkStatus();
}

absl::Status GenAiServable::prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    if (executionContext->apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }

    if (executionContext->sessionTurn.active) {
        auto journalStatus = processSessionStateStore()->recordGenerationConfig(
            executionContext->sessionTurn,
            executionContext->inputRequest.generationConfig,
            executionContext->apiHandler->getToolChoice());
        if (!journalStatus.ok())
            return journalStatus;
    }

    auto status = validateInputCompatibility(executionContext);
    if (!status.ok()) {
        return status;
    }

    InputRequest& req = executionContext->inputRequest;
    InputProcessor processor(getProperties()->inputProcessorContext, req);
    status = processor.process(req);
    if (!status.ok()) {
        return status;
    }

    if (executionContext->apiHandler->getOutputParser() != nullptr) {
        executionContext->apiHandler->getOutputParser()->detectAndSetImplicitReasoningStart(req.promptText);
    }
    if (Config::instance().getServerSettings().verboseResponse) {
        executionContext->apiHandler->enableVerboseResponse(req.promptText);
    }
    if (getProperties()->maxModelLength.has_value()) {
        if (req.inputIds.get_size() > getProperties()->maxModelLength.value()) {
            std::stringstream ss;
            ss << "Number of prompt tokens: " << req.inputIds.get_size()
               << " exceeds model max length: " << getProperties()->maxModelLength.value();
            SPDLOG_LOGGER_WARN(llm_calculator_logger, ss.str());
            return absl::Status(absl::StatusCode::kInvalidArgument, ss.str());
        }
        if (executionContext->apiHandler->getMaxTokens().has_value() &&
            req.inputIds.get_size() + static_cast<size_t>(executionContext->apiHandler->getMaxTokens().value()) >
                getProperties()->maxModelLength.value()) {
            std::stringstream ss;
            ss << "Number of prompt tokens: " << req.inputIds.get_size()
               << " + max tokens value: " << executionContext->apiHandler->getMaxTokens().value()
               << " exceeds model max length: " << getProperties()->maxModelLength.value();
            SPDLOG_LOGGER_WARN(llm_calculator_logger, ss.str());
            return absl::Status(absl::StatusCode::kInvalidArgument, ss.str());
        }
    }
    executionContext->apiHandler->setPromptTokensUsage(req.inputIds.get_size());
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "{}", getPromptTokensString(req.inputIds));
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Pipeline input text: {}", req.promptText);

    return absl::OkStatus();
}

absl::Status GenAiServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    const bool hasLogprobs = executionContext->apiHandler->getRequest().logprobschat ||
                             executionContext->apiHandler->getRequest().logprobs;
    const size_t numOutputs = executionContext->generationOutputs.size();

    // Build streamer config once; shared across all per-sequence streamers.
    ov::AnyMap streamerConfig;
    if (!executionContext->apiHandler->getRequest().skipSpecialTokens) {
        streamerConfig.insert(ov::genai::skip_special_tokens(false));
    }

    std::vector<std::vector<Delta>> allDeltas;
    std::vector<ov::genai::GenerationFinishReason> finishReasons;
    std::vector<UnaryChoiceLogprobs> logprobData;
    allDeltas.reserve(numOutputs);
    finishReasons.reserve(numOutputs);

    for (size_t i = 0; i < numOutputs; ++i) {
        const auto& output = executionContext->generationOutputs[i];

        if (executionContext->apiHandler->isVerboseResponse()) {
            executionContext->apiHandler->appendVerboseRawTokens(output.generated_ids);
        }
        executionContext->apiHandler->incrementProcessedTokens(output.generated_ids.size());

        std::vector<Delta> localDeltas;
        if (numOutputs == 1) {
            // Single sequence: reuse the OVMSTextStreamer and deltaChannel built in parseRequest.
            executionContext->textStreamer->write(output.generated_ids);
            finishTextStreamer(executionContext->textStreamer, output.finish_reason);
            localDeltas = executionContext->deltaChannel.drain();
        } else {
            // Multiple sequences: each beam requires its own independent stateful streamer
            // (hold-back buffer, parser state are per-sequence).
            auto cb = [&localDeltas](Delta delta, bool) -> ov::genai::StreamingStatus {
                localDeltas.push_back(std::move(delta));
                return ov::genai::StreamingStatus::RUNNING;
            };
            auto outputParser = executionContext->apiHandler->getOutputParser();
            if (outputParser) {
                outputParser->resetStreamingState();
            }
            auto tempStreamer = std::make_shared<OVMSTextStreamer>(
                getProperties()->tokenizer,
                outputParser,
                executionContext->apiHandler->areToolsAvailable(),
                std::move(cb),
                streamerConfig);
            tempStreamer->write(output.generated_ids);
            tempStreamer->end(output.finish_reason);
        }

        allDeltas.push_back(std::move(localDeltas));
        finishReasons.push_back(output.finish_reason);
        if (hasLogprobs) {
            logprobData.push_back({output.generated_ids, output.generated_log_probs});
        }
    }

    if (hasLogprobs) {
        executionContext->response = executionContext->apiHandler->serializeUnaryResponse(
            allDeltas, finishReasons, logprobData);
    } else {
        executionContext->response = executionContext->apiHandler->serializeUnaryResponse(
            allDeltas, finishReasons);
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

absl::Status GenAiServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    if (executionContext->generationOutputs.size() != 1) {
        return absl::InternalError("For streaming we expect exactly one generation output");
    }
    auto& generationOutput = executionContext->generationOutputs[0];
    executionContext->apiHandler->incrementProcessedTokens(generationOutput.generated_ids.size());
    if (executionContext->apiHandler->isVerboseResponse()) {
        executionContext->apiHandler->appendVerboseRawTokens(generationOutput.generated_ids);
    }

    bool isFirstToken = GenerationPhase::INPUT_TOKEN_PROCESSING == executionContext->generationPhase;
    if (isFirstToken) {
        executionContext->generationPhase = GenerationPhase::OUTPUT_TOKEN_PROCESSING;
    }

    ov::genai::GenerationFinishReason finishReason = generationOutput.finish_reason;
    const bool isFinishing = (finishReason != ov::genai::GenerationFinishReason::NONE);

    // OVMSTextStreamer::write() fires the callback for each flush event, pushing
    // Documents into executionContext->deltaChannel.
    executionContext->textStreamer->write(generationOutput.generated_ids);

    if (isFinishing) {
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        // end() flushes held-back tokens and passes the actual terminal reason. Any resulting
        // Document is pushed into deltaChannel by the callback.
        finishTextStreamer(executionContext->textStreamer, finishReason);
    }

    // Drain all deltas accumulated during this write()/end() cycle.
    std::vector<Delta> deltas = executionContext->deltaChannel.drain();
    const size_t count = deltas.size();

    if (!isFinishing) {
        // For RESPONSES endpoint, always call serializeStreamingChunk so lifecycle
        // events (output_item.added, content_part.added) are emitted on the first
        // call, even before the tokenizer produces text.
        if (count > 0 || executionContext->apiHandler->getEndpoint() == Endpoint::RESPONSES) {
            // Emit each delta. All are mid-stream so finishReason is NONE.
            for (size_t i = 0; i < count; ++i) {
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(deltas[i]),
                    ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", serialized);
                }
            }
            if (count == 0) {
                // No delta generated yet — emit lifecycle events (response.created, response.in_progress)
                // for the RESPONSES endpoint before any content arrives.
                if (!executionContext->lifecyclePrimed) {
                    std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                        FinishDelta{}, ov::genai::GenerationFinishReason::NONE);
                    if (!serialized.empty()) {
                        executionContext->response += wrapTextInServerSideEventMessage(serialized);
                        executionContext->lifecyclePrimed = true;
                    }
                }
            }
        } else if (isFirstToken) {
            std::string serializedChunk = executionContext->apiHandler->serializeStreamingHandshakeChunk();
            if (!serializedChunk.empty()) {
                executionContext->response = wrapTextInServerSideEventMessage(serializedChunk);
            }
        }
        executionContext->sendLoopbackSignal = true;
    } else {
        // Finishing: emit all pending deltas; the last one gets the real finishReason.
        if (count > 0) {
            for (size_t i = 0; i < count; ++i) {
                const bool isLast = (i == count - 1);
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(deltas[i]),
                    isLast ? finishReason : ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                }
            }
        } else {
            // No delta produced (generation ended on a swallowed token).
            // Still emit a chunk carrying the finish_reason with an empty Document.
            std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                FinishDelta{}, finishReason);
            if (!serialized.empty()) {
                executionContext->response += wrapTextInServerSideEventMessage(serialized);
            }
        }
        if (executionContext->apiHandler->getStreamOptions().includeUsage) {
            std::string usageChunk = executionContext->apiHandler->serializeStreamingUsageChunk();
            if (!usageChunk.empty()) {
                executionContext->response += wrapTextInServerSideEventMessage(usageChunk);
            }
        }
        executionContext->response += wrapTextInServerSideEventMessage("[DONE]");
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", executionContext->response);
        executionContext->sendLoopbackSignal = false;
    }

    return absl::OkStatus();
}

absl::Status prepareLegacyPartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyCtx = std::static_pointer_cast<LegacyServableExecutionContextBase>(executionContext);
    if (legacyCtx->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::vector<Delta> deltas = executionContext->deltaChannel.drain();
    const bool isFinishing = executionContext->deltaChannel.complete();

    // Helper: accumulate verbose raw text from a delta's content field.
    // Both LLM-Legacy (switched from token-based) and VLM-Legacy use per-delta
    // text extraction, which is correct because OVMSTextStreamer is configured with
    // skip_special_tokens(false) in verbose mode, so delta content already includes
    // special tokens.
    auto appendVerboseContent = [&](const Delta& delta) {
        if (executionContext->apiHandler->isVerboseResponse()) {
            if (const auto* cd = std::get_if<ContentDelta>(&delta))
                executionContext->apiHandler->appendVerboseRawText(cd->text);
        }
    };

    if (!isFinishing) {
        // For RESPONSES endpoint, always call serializeStreamingChunk so that
        // output item initialization events are emitted even before the tokenizer produces text.
        if (deltas.size() > 0 || executionContext->apiHandler->getEndpoint() == Endpoint::RESPONSES) {
            for (auto& delta : deltas) {
                appendVerboseContent(delta);
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(delta), ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", serialized);
                }
            }
            if (deltas.empty()) {
                // No delta generated yet — emit lifecycle events for RESPONSES endpoint.
                if (!executionContext->lifecyclePrimed) {
                    std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                        FinishDelta{}, ov::genai::GenerationFinishReason::NONE);
                    if (!serialized.empty()) {
                        executionContext->response = wrapTextInServerSideEventMessage(serialized);
                        executionContext->lifecyclePrimed = true;
                    }
                }
            }
        }
        executionContext->sendLoopbackSignal = true;
    } else {
        // Wait for the readySignal
        // (set right after pipe->generate() returns and results are assigned)
        // to guarantee results is populated before we read finish_reasons and perf_metrics.
        legacyCtx->finished.wait();
        if (!legacyCtx->success) {
            return absl::InvalidArgumentError("Request processing failed, check its correctness.");
        }
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        // end() was already called by pipe->generate() internally; all deltas are
        // already in deltaChannel before signalComplete() fired. Drain any remaining.
        for (auto& d : executionContext->deltaChannel.drain()) {
            deltas.push_back(std::move(d));
        }
        // Legacy generation path always runs with deltas=1, so we read the single finish reason at index 0.
        const ov::genai::GenerationFinishReason finishReason = legacyCtx->legacyFinishReason();
        legacyCtx->setLegacyUsage(*executionContext->apiHandler);
        if (!deltas.empty()) {
            for (size_t i = 0; i < deltas.size(); ++i) {
                const bool isLast = (i == deltas.size() - 1);
                appendVerboseContent(deltas[i]);
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(deltas[i]),
                    isLast ? finishReason : ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                }
            }
        } else {
            // Parser produced no delta (generation ended on a swallowed token).
            std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                FinishDelta{}, finishReason);
            if (!serialized.empty()) {
                executionContext->response += wrapTextInServerSideEventMessage(serialized);
            }
        }
        if (executionContext->apiHandler->getStreamOptions().includeUsage)
            executionContext->response += wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingUsageChunk());
        executionContext->response += wrapTextInServerSideEventMessage("[DONE]");
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", executionContext->response);
        executionContext->sendLoopbackSignal = false;
    }
    return absl::OkStatus();
}

absl::Status LegacyServableBase::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    return prepareLegacyPartialResponse(executionContext);
}

void logRequestDetails(const ovms::HttpPayload& payload) {
    auto parsedJson = payload.parsedJson;
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    parsedJson->Accept(writer);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", buffer.GetString());
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
    for (const auto& [name, value] : payload.headers) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request header: {}={}", name, value);
    }
}

}  // namespace ovms
