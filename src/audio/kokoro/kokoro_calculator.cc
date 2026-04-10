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
#include <mutex>
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/audio/audio_utils.hpp"
#include "src/http_payload.hpp"
#include "src/logging.hpp"
#include "src/port/dr_audio.hpp"

#include "../../model_metric_reporter.hpp"
#include "../../executingstreamidguard.hpp"

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#include <espeak-ng/speak_lib.h>

#include "kokoro_servable.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace ovms;

namespace {

#ifndef espeakPHONEMES_IPA
#define espeakPHONEMES_IPA 0x02
#endif
#ifndef espeakPHONEMES_NO_STRESS
#define espeakPHONEMES_NO_STRESS 0x08
#endif

std::string retone(const std::string& p) {
    std::string result = p;
    
    auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    
    // Tone mark replacements
    replaceAll(result, "˧˩˧", "↓");  // third tone
    replaceAll(result, "˧˥", "↗");   // second tone
    replaceAll(result, "˥˩", "↘");   // fourth tone
    replaceAll(result, "˥", "→");    // first tone
    
    // Unicode character replacements (UTF-8 encoded)
    replaceAll(result, "\xCA\x97\xCC\x89", "ɨ");  // chr(635)+chr(809)
    replaceAll(result, "\xCA\x91\xCC\x89", "ɨ");  // chr(633)+chr(809)
    
    // Verify chr(809) removed
    if (result.find("\xCC\x89") != std::string::npos) {
        SPDLOG_WARN("Combining diacritic (chr 809) still present: {}", result);
    }
    
    return result;
}

std::string getEspeakVoice(const std::string& isoLanguageCode) {
    // ISO 639-1 codes with optional region codes
    if (isoLanguageCode == "en-us") {
        return "en-us";  // American English (default for 'en')
    } else if (isoLanguageCode == "en-gb") {
        return "en";     // British English
    } else if (isoLanguageCode == "en") {
        return "en-us";  // Default to American English when only 'en' specified
    } else if (isoLanguageCode == "es") {
        return "es";
    } else if (isoLanguageCode == "fr") {
        return "fr";
    } else if (isoLanguageCode == "hi") {
        return "hi";
    } else if (isoLanguageCode == "it") {
        return "it";
    } else if (isoLanguageCode == "ja") {
        return "ja";
    } else if (isoLanguageCode == "pt-br") {
        return "pt";     // Brazilian Portuguese
    } else if (isoLanguageCode == "zh" || isoLanguageCode == "zh-cn") {
        return "cmn-latn-pinyin";    // Mandarin Chinese
    }
    return "";  // Unsupported
}

bool isSupportedLanguage(const std::string& isoLanguageCode) {
    // Only accept ISO 639-1 codes and regional variants
    return !getEspeakVoice(isoLanguageCode).empty();
}

void espeakPhonemizeAll(const std::string& textUtf8, std::string& outIpa, const std::string& language = "en", bool noStress = true) {
    outIpa.clear();
    auto& espeak = ovms::EspeakInstance::instance();
    if (!espeak.isReady()) {
        SPDLOG_ERROR("eSpeak not initialized");
        return;
    }

    std::lock_guard<std::mutex> guard(espeak.mutex());

    // Get the eSpeak voice name from the ISO language code
    // Kokoro supports 9 languages: American English, British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin Chinese
    std::string voiceName = getEspeakVoice(language);
    if (voiceName.empty()) {
        // This should not happen if validation was done, but fallback just in case
        SPDLOG_ERROR("Invalid language code '{}' passed to espeakPhonemizeAll", language);
        voiceName = "en-us";
    }
    if (espeak_SetVoiceByName(voiceName.c_str()) != EE_OK) {
        SPDLOG_ERROR("Failed to set eSpeak voice '{}'", voiceName);
        if (voiceName != "en-us" && espeak_SetVoiceByName("en-us") == EE_OK) {
            voiceName = "en-us";
        } else {
            return;
        }
    }

    const int mode = espeakPHONEMES_IPA | (noStress ? espeakPHONEMES_NO_STRESS : 0);
    const void* pos = static_cast<const void*>(textUtf8.c_str());
    const char* endPtr = static_cast<const char*>(pos) + textUtf8.size();
    std::string rawIpa;

    while (pos && static_cast<const char*>(pos) < endPtr) {
        const char* ipaChunk = espeak_TextToPhonemes(&pos, espeakCHARS_UTF8, mode);
        if (ipaChunk && *ipaChunk) {
            if (!rawIpa.empty()) {
                rawIpa.push_back(' ');
            }
            rawIpa.append(ipaChunk);
        }
    }

    // Strip combining diacriticals (U+0300..U+036F) and collapse spaces
    std::string cleaned;
    cleaned.reserve(rawIpa.size());
    for (size_t i = 0; i < rawIpa.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(rawIpa[i]);
        if (i + 1 < rawIpa.size()) {
            unsigned char next = static_cast<unsigned char>(rawIpa[i + 1]);
            if ((c == 0xCC && next >= 0x80) || (c == 0xCD && next <= 0xAF)) {
                i++;
                continue;
            }
        }
        cleaned.push_back(c);
    }

    outIpa.reserve(cleaned.size());
    bool lastSpace = false;
    for (char c : cleaned) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!lastSpace) {
                outIpa.push_back(' ');
                lastSpace = true;
            }
        } else {
            outIpa.push_back(c);
            lastSpace = false;
        }
    }

    if (!outIpa.empty() && std::isspace(static_cast<unsigned char>(outIpa.back()))) {
        outIpa.pop_back();
    }

    SPDLOG_DEBUG("IPA phonemes: '{}' (length: {})", outIpa, outIpa.size());
}


size_t utf8CharLen(unsigned char lead) {
    if (lead < 0x80)
        return 1;
    if ((lead >> 5) == 0x6)
        return 2;
    if ((lead >> 4) == 0xE)
        return 3;
    if ((lead >> 3) == 0x1E)
        return 4;
    return 1;
}

void tokenize(const std::string& textUtf8,
    std::vector<int64_t>& tokenIds,
    const ovms::VocabIndex& ix,
    const std::string& language = "en") {
    tokenIds.clear();
    // Reserve estimated capacity to avoid reallocations
    tokenIds.reserve(textUtf8.size() / 2);
    
    size_t pos = 0;
    const size_t n = textUtf8.size();
    size_t unknownCount = 0;

    while (pos < n) {
        size_t maxTry = std::min(ix.max_token_bytes, n - pos);
        int foundId = -1;
        size_t foundLen = 0;

        for (size_t len = maxTry; len > 0; --len) {
            auto it = ix.by_token.find(std::string(textUtf8.data() + pos, len));
            if (it != ix.by_token.end()) {
                foundId = it->second;
                foundLen = len;
                break;
            }
        }

        if (foundId >= 0) {
            tokenIds.push_back(foundId);
            pos += foundLen;
        } else {
            const unsigned char lead = static_cast<unsigned char>(textUtf8[pos]);
            const size_t adv = utf8CharLen(lead);
            std::string unknownBytes(textUtf8.data() + pos, std::min(adv, n - pos));
            unknownCount++;
            SPDLOG_DEBUG("Tokenizer [lang={}]: unknown phoneme at pos {}: '{}' (skipping)",
                language, pos, unknownBytes);
            pos += std::min(adv, n - pos);
        }
    }
    if (unknownCount > 0) {
        SPDLOG_WARN("Tokenize [lang={}]: {} unknown phonemes found. Produced {} token ids. "
                    "Consider updating vocabulary for better {} speech quality.",
                    language, unknownCount, tokenIds.size(), language);
    } else {
        SPDLOG_DEBUG("Tokenize [lang={}]: produced {} ids without unknown phonemes", language, tokenIds.size());
    }
}
}  // namespace

namespace mediapipe {

const std::string KOKORO_SESSION_SIDE_PACKET_TAG = "KOKORO_NODE_RESOURCES";

class KokoroCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    std::string defaultLanguage;  // Language configured in graph pbtxt

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->InputSidePackets().Tag(KOKORO_SESSION_SIDE_PACKET_TAG).Set<KokoroServableMap>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(kokoro_calculator_logger, "KokoroCalculator [Node: {}] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(kokoro_calculator_logger, "KokoroCalculator [Node: {}] Open", cc->NodeName());
        
        // Read language from graph configuration
        const auto& options = cc->Options<KokoroCalculatorOptions>();
        this->defaultLanguage = options.has_language() ? options.language() : "en";
        
        // Normalize language code to lowercase
        std::transform(this->defaultLanguage.begin(), this->defaultLanguage.end(), this->defaultLanguage.begin(), ::tolower);
        
        // Validate language is supported
        if (!isSupportedLanguage(this->defaultLanguage)) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Invalid language in graph config: '", this->defaultLanguage, "'. ",
                "Supported ISO 639-1 language codes: en, es, fr, hi, it, ja, pt-br, zh. ",
                "Regional variants: en-us, en-gb, pt-br, zh-cn"
            ));
        }
        
        SPDLOG_LOGGER_DEBUG(kokoro_calculator_logger, 
            "KokoroCalculator [Node: {}] configured for language: {}", 
            cc->NodeName(), this->defaultLanguage);
        
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(kokoro_calculator_logger, "KokoroCalculator [Node: {}] Process start", cc->NodeName());
        try {
            KokoroServableMap servablesMap = cc->InputSidePackets()
                                                 .Tag(KOKORO_SESSION_SIDE_PACKET_TAG)
                                                 .Get<KokoroServableMap>();
            auto servableIt = servablesMap.find(cc->NodeName());
            RET_CHECK(servableIt != servablesMap.end())
                << "Could not find initialized Kokoro node named: " << cc->NodeName();
            auto servable = servableIt->second;

            const auto& payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();
            auto it = payload.parsedJson->FindMember("input");
            RET_CHECK(it != payload.parsedJson->MemberEnd()) << "Missing 'input' in request";
            RET_CHECK(it->value.IsString()) << "'input' must be a string";
            const std::string text = it->value.GetString();

            // Read optional "voice" parameter (OpenAI TTS API)
            std::string voiceName;
            auto voiceIt = payload.parsedJson->FindMember("voice");
            if (voiceIt != payload.parsedJson->MemberEnd() && voiceIt->value.IsString()) {
                voiceName = voiceIt->value.GetString();
            }

            // Language is configured in the graph pbtxt, not from request
            // Use the defaultLanguage set during Open()
            const std::string language = this->defaultLanguage;
            SPDLOG_DEBUG("Using configured language: {}", language);

            // Text -> IPA phonemization
            std::string phonemes;
            
            // Use eSpeak for all languages
            espeakPhonemizeAll(text, phonemes, language, /*noStress=*/false);
            if(language == "zh" || language == "zh-cn"){
                phonemes = retone(phonemes);
            }
            
            SPDLOG_DEBUG("Input text: '{}' (language: {}), IPA phonemes ({} chars): '{}'", text, language, phonemes.size(), phonemes);

            // Preserve trailing punctuation from original text (eSpeak strips it)
            // if (!text.empty()) {
            //     char last = text.back();
            //     if (last == '.' || last == '!' || last == '?' || last == ';' || last == ':' || last == ',') {
            //         phonemes.push_back(last);
            //     }
            // }
            SPDLOG_DEBUG("After E2M mapping ({} chars): '{}'", phonemes.size(), phonemes);
            // IPA -> Kokoro token IDs
            const auto& vocabIx = servable->getVocabIndex();
            std::vector<int64_t> tokenIds;
            tokenize(phonemes, tokenIds, vocabIx, language);

            // Wrap with PAD token (id=0) at both ends — matches official
            // forward_with_tokens: input_ids = [[0, *tokens, 0]]
            tokenIds.insert(tokenIds.begin(), 0);
            tokenIds.push_back(0);

            // Voice embedding — select slice from voice pack based on content token count
            size_t numContentTokens = tokenIds.size() >= 2 ? tokenIds.size() - 2 : 0;  // exclude BOS pad + EOS
            const float* voiceSlice = servable->getVoiceSlice(voiceName, numContentTokens);
            RET_CHECK(voiceSlice != nullptr) << "No voice pack loaded (place .bin files in <model_dir>/voices/)";

            auto inputIdsTensor = ov::Tensor{ov::element::i64, ov::Shape{1, tokenIds.size()}};
            auto refS = ov::Tensor{ov::element::f32, ov::Shape{1, KokoroServable::STYLE_DIM}};
            auto speed = ov::Tensor{ov::element::f32, ov::Shape{1}};

            *reinterpret_cast<float*>(speed.data()) = 1.0f;
            std::copy(tokenIds.data(), tokenIds.data() + tokenIds.size(),
                reinterpret_cast<int64_t*>(inputIdsTensor.data()));
            std::copy(voiceSlice, voiceSlice + KokoroServable::STYLE_DIM,
                reinterpret_cast<float*>(refS.data()));

            // Inference
            ModelMetricReporter unused(nullptr, nullptr, "unused", 1);
            auto executingStreamIdGuard =
                std::make_unique<ExecutingStreamIdGuard>(servable->getInferRequestsQueue(), unused);
            ov::InferRequest& inferRequest = executingStreamIdGuard->getInferRequest();

            inferRequest.set_tensor("input_ids", inputIdsTensor);
            inferRequest.set_tensor("103", refS);
            inferRequest.set_tensor("speed", speed);
            inferRequest.start_async();
            inferRequest.wait();

            // Collect audio output
            auto out = inferRequest.get_tensor(inferRequest.get_compiled_model().outputs()[0]);
            RET_CHECK(out.get_shape().size() == 1);
            RET_CHECK(out.get_element_type() == ov::element::f32);
            const size_t samples = out.get_shape()[0];
            const float* data = out.data<float>();

            SPDLOG_DEBUG("Model output: {} audio samples ({:.2f}s at 24kHz)",
                samples, static_cast<float>(samples) / 24000.0f);

            void* wavDataPtr = nullptr;
            size_t wavSize = 0;
            prepareAudioOutputKokoro(&wavDataPtr, wavSize, samples, data);

            auto output = std::make_unique<std::string>(reinterpret_cast<char*>(wavDataPtr), wavSize);
            drwav_free(wavDataPtr, NULL);

            cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        } catch (const std::exception& e) {
            SPDLOG_ERROR("KokoroCalculator [Node: {}] Process failed: {}", cc->NodeName(), e.what());
            return absl::InvalidArgumentError(e.what());
        } catch (...) {
            SPDLOG_ERROR("KokoroCalculator [Node: {}] Process failed: unknown error", cc->NodeName());
            return absl::InvalidArgumentError("Kokoro processing failed");
        }
        SPDLOG_LOGGER_DEBUG(kokoro_calculator_logger, "KokoroCalculator [Node: {}] Process end", cc->NodeName());
        return absl::OkStatus();
    }
};

const std::string KokoroCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string KokoroCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(KokoroCalculator);

}  // namespace mediapipe
