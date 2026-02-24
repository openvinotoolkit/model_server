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
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
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
#include "absl/strings/escaping.h"
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

void espeakPhonemizeAll(const std::string& textUtf8, std::string& outIpa, bool noStress = true) {
    outIpa.clear();
    auto& espeak = ovms::EspeakInstance::instance();
    if (!espeak.isReady()) {
        SPDLOG_ERROR("eSpeak not initialized");
        return;
    }

    std::lock_guard<std::mutex> guard(espeak.mutex());

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

// Post-process eSpeak IPA into Kokoro/misaki phoneme alphabet.
// Mirrors misaki.espeak.EspeakFallback.E2M for American English.
// void espeakIpaToKokoro(std::string& ps) {
//     // Helper: replace all occurrences of `from` with `to` in `s`.
//     auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
//         if (from.empty()) return;
//         size_t pos = 0;
//         while ((pos = s.find(from, pos)) != std::string::npos) {
//             s.replace(pos, from.size(), to);
//             pos += to.size();
//         }
//     };

//     // --- Multi-char replacements (longest first) ---
//     // Syllabic n with glottal stop
//     replaceAll(ps, "\xca\x94\xcb\x8c\x6e\xcc\xa9", "\xca\x94\x6e");  // ʔˌn̩ → ʔn
//     replaceAll(ps, "\xca\x94\x6e\xcc\xa9", "\xca\x94\x6e");              // ʔn̩ → ʔn
//     // Syllabic mark before consonant → ᵊ + consonant
//     // ə̩l → ᵊl  (syllabic l)
//     replaceAll(ps, "\xc9\x99\xcc\xa9\x6c", "\xe1\xb5\x8a\x6c");          // əl̩ → ᵊl  (approximation)

//     // Diphthongs
//     replaceAll(ps, "a\xc9\xaa", "I");       // aɪ → I
//     replaceAll(ps, "a\xca\x8a", "W");       // aʊ → W
//     replaceAll(ps, "e\xc9\xaa", "A");       // eɪ → A
//     replaceAll(ps, "\xc9\x94\xc9\xaa", "Y"); // ɔɪ → Y
//     replaceAll(ps, "o\xca\x8a", "O");       // oʊ → O  (American)
//     replaceAll(ps, "\xc9\x99\xca\x8a", "O"); // əʊ → O  (British)

//     // Affricates
//     replaceAll(ps, "d\xca\x92", "\xca\xa4");  // dʒ → ʤ
//     replaceAll(ps, "t\xca\x83", "\xca\xa7");  // tʃ → ʧ

//     // Palatalization
//     replaceAll(ps, "\xca\xb2\x6f", "jo");     // ʲo → jo
//     replaceAll(ps, "\xca\xb2\xc9\x99", "j\xc9\x99"); // ʲə → jə
//     replaceAll(ps, "\xca\xb2", "");           // ʲ → (delete)

//     // R-colored vowels and vowel length
//     replaceAll(ps, "\xc9\x9c\xcb\x90\xc9\xb9", "\xc9\x9c\xc9\xb9"); // ɜːɹ → ɜɹ
//     replaceAll(ps, "\xc9\x9c\xcb\x90", "\xc9\x9c\xc9\xb9");           // ɜː → ɜɹ
//     replaceAll(ps, "\xc9\xaa\xc9\x99", "i\xc9\x99");                   // ɪə → iə

//     // --- Single-char replacements ---
//     replaceAll(ps, "\xc9\x9a", "\xc9\x99\xc9\xb9"); // ɚ → əɹ
//     replaceAll(ps, "\xc9\x90", "\xc9\x99");           // ɐ → ə
//     replaceAll(ps, "\xc9\xac", "l");                   // ɬ → l
//     replaceAll(ps, "\xc3\xa7", "k");                   // ç → k
//     replaceAll(ps, "x", "k");                           // x → k
//     replaceAll(ps, "r", "\xc9\xb9");                   // r → ɹ
//     replaceAll(ps, "\xcb\x90", "");                     // ː → (strip length marks)
//     replaceAll(ps, "\xcc\x83", "");                     // ̃ → (strip nasal tilde)

//     // British vowel mappings (in case eSpeak uses 'en' voice)
//     replaceAll(ps, "\xc9\x92", "\xc9\x94");           // ɒ → ɔ

//     // Remaining standalone vowels (must be AFTER diphthong replacements)
//     replaceAll(ps, "o", "\xc9\x94");                   // o → ɔ  (for espeak < 1.52)
//     replaceAll(ps, "e", "A");                           // e → A

//     // Flap and glottal stop (misaki version != 2.0)
//     replaceAll(ps, "\xc9\xbe", "T");                   // ɾ → T
//     replaceAll(ps, "\xca\x94", "t");                   // ʔ → t
// }

size_t utf8CharLen(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;
}

void tokenize(const std::string& textUtf8,
              std::vector<int64_t>& tokenIds,
              const ovms::VocabIndex& ix) {
    tokenIds.clear();
    size_t pos = 0;
    const size_t n = textUtf8.size();

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
            SPDLOG_WARN("Tokenizer: unknown bytes at pos {}: '{}'",
                        pos, std::string(textUtf8.data() + pos, std::min(adv, n - pos)));
            pos += std::min(adv, n - pos);
        }
    }
    SPDLOG_DEBUG("Tokenize: produced {} ids", tokenIds.size());
}
}  // namespace

namespace mediapipe {

const std::string KOKORO_SESSION_SIDE_PACKET_TAG = "KOKORO_NODE_RESOURCES";

class KokoroCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

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
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(kokoro_calculator_logger, "KokoroCalculator [Node: {}] Process start", cc->NodeName());

        KokoroServableMap servablesMap = cc->InputSidePackets()
            .Tag(KOKORO_SESSION_SIDE_PACKET_TAG).Get<KokoroServableMap>();
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

        // Text -> IPA phonemization
        std::string phonemes;
        espeakPhonemizeAll(text, phonemes, /*noStress=*/false);
        SPDLOG_DEBUG("Input text: '{}', IPA phonemes ({} chars): '{}'", text, phonemes.size(), phonemes);

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
        std::vector<std::vector<int64_t>> inputTokens(1);
        tokenize(phonemes, inputTokens[0], vocabIx);

        // Wrap with PAD token (id=0) at both ends — matches official
        // forward_with_tokens: input_ids = [[0, *tokens, 0]]
        inputTokens[0].insert(inputTokens[0].begin(), 0);
        inputTokens[0].push_back(0);

        // Voice embedding — select slice from voice pack based on content token count
        auto& ids = inputTokens[0];
        size_t numContentTokens = ids.size() >= 2 ? ids.size() - 2 : 0;  // exclude BOS pad + EOS
        const float* voiceSlice = servable->getVoiceSlice(voiceName, numContentTokens);
        RET_CHECK(voiceSlice != nullptr) << "No voice pack loaded (place .bin files in <model_dir>/voices/)";

        auto inputIdsTensor = ov::Tensor{ov::element::i64, ov::Shape{1, ids.size()}};
        auto refS = ov::Tensor{ov::element::f32, ov::Shape{1, KokoroServable::STYLE_DIM}};
        auto speed = ov::Tensor{ov::element::f32, ov::Shape{1}};

        *reinterpret_cast<float*>(speed.data()) = 1.0f;
        std::copy(ids.data(), ids.data() + ids.size(),
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
        SPDLOG_LOGGER_DEBUG(kokoro_calculator_logger, "KokoroCalculator [Node: {}] Process end", cc->NodeName());
        return absl::OkStatus();
    }
};

const std::string KokoroCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string KokoroCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(KokoroCalculator);

}  // namespace mediapipe
