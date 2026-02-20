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

        // Text -> IPA phonemization
        std::string phonemes;
        espeakPhonemizeAll(text, phonemes, /*noStress=*/true);
        SPDLOG_DEBUG("Input text: '{}', IPA phonemes ({} chars): '{}'", text, phonemes.size(), phonemes);

        // IPA -> Kokoro token IDs
        const auto& vocabIx = servable->getVocabIndex();
        std::vector<std::vector<int64_t>> inputTokens(1);
        tokenize(phonemes, inputTokens[0], vocabIx);

        // Prepend PAD token (id=0) - Kokoro model requires BOS/PAD at start
        inputTokens[0].insert(inputTokens[0].begin(), 0);

        // Append EOS (period token = 4) if not already present
        if (inputTokens[0].empty() || inputTokens[0].back() != 4) {
            inputTokens[0].push_back(4);
        }

        // Voice embedding
        std::vector<float> voice = {
            -0.2296, 0.1835, -0.0069, -0.1240, -0.2505, 0.0112, -0.0759, -0.1650,
            -0.2665, -0.1965, 0.0242, -0.1667, 0.3524, 0.2140, 0.3069, -0.3377,
            -0.0878, -0.0477, 0.0813, -0.2135, -0.2340, -0.1971, 0.0200, 0.0145,
            0.0016, 0.2596, -0.2665, 0.1434, 0.0503, 0.0867, 0.1905, -0.1281,
            0.0658, -0.0639, -0.0920, 0.2444, -0.1506, -0.2197, 0.1385, 0.2133,
            -0.0755, -0.0188, -0.0142, 0.2301, -0.0776, -0.0748, 0.0172, 0.0430,
            -0.1009, 0.1519, 0.1137, 0.0641, 0.2264, 0.1911, -0.0205, 0.2578,
            0.2210, -0.0784, -0.0235, -0.0547, 0.2191, -0.1623, -0.2416, 0.0076,
            0.0574, 0.2186, 0.0080, 0.0473, 0.0972, 0.0286, 0.1324, 0.0686,
            0.2652, -0.2237, -0.0980, -0.1693, -0.1866, 0.2273, 0.2008, -0.0683,
            0.0957, 0.0623, -0.1891, 0.1620, 0.1811, -0.0516, -0.0800, -0.1416,
            -0.2374, -0.1892, 0.1726, -0.0690, -0.0300, 0.0467, -0.2811, -0.1603,
            0.0342, -0.1054, -0.0604, -0.0475, -0.0908, -0.1286, 0.1105, -0.1186,
            0.0582, 0.1887, 0.0345, 0.2081, 0.1404, -0.2532, 0.0026, 0.0402,
            0.0812, -0.0512, 0.0128, 0.0084, -0.0970, -0.0362, 0.0036, -0.0720,
            -0.0850, 0.0221, -0.1037, 0.0569, 0.0187, -0.0649, -0.0288, -0.1795,
            0.0045, 0.2535, 0.6751, 0.1578, -0.0966, 0.1516, 0.2109, 0.2033,
            -0.2155, -0.1783, 0.0836, -0.1050, 0.0676, -0.0237, 0.0387, -0.2564,
            0.1891, 0.1305, -0.3239, -0.1312, 0.2723, 0.0745, 0.1335, 0.0302,
            0.0172, 0.2207, 0.0215, -0.0379, -0.1954, 0.4944, 0.2905, -0.0306,
            0.2858, 0.2341, 0.0545, 0.4626, 0.2947, 0.3802, 0.2820, 0.1557,
            0.1743, -0.1410, 0.0986, 0.4751, -0.2146, 0.3530, -0.2357, -0.5626,
            -0.0617, 0.2190, 0.0992, -0.2365, 0.3726, 0.2092, 0.1660, 0.1928,
            0.5731, -0.1734, -0.0816, -0.3191, -0.1871, -0.2217, -0.0112, 0.1261,
            0.1601, 0.3835, 0.0451, -0.1927, -0.1116, 0.2204, -0.0379, -0.0094,
            -0.0455, -0.4831, -0.3345, -0.2119, 0.4803, 0.1214, 0.1723, 0.2605,
            0.0051, -0.2587, 0.0511, -0.1318, 0.0227, -0.0645, 0.2573, -0.0205,
            0.0665, -0.3562, -0.6070, 0.4191, 0.0351, 0.2033, -0.5508, -0.1415,
            -0.1249, -0.0986, -0.1120, -0.1187, 0.0600, 0.1974, 0.5017, -0.0247,
            -0.2986, 0.3983, -0.1159, -0.4275, -0.0164, -0.3783, 0.0717, 0.1478,
            -0.1144, 0.2292, 0.2741, 0.4309, -0.1611, 0.0755, -0.0981, 0.4584,
            -0.2061, -0.0787, -0.1779, 0.2275, -0.1742, -0.2230, -0.1739, 0.0646
        };

        auto& ids = inputTokens[0];

        auto inputIdsTensor = ov::Tensor{ov::element::i64, ov::Shape{1, ids.size()}};
        auto refS = ov::Tensor{ov::element::f32, ov::Shape{1, voice.size()}};
        auto speed = ov::Tensor{ov::element::f32, ov::Shape{1}};

        *reinterpret_cast<float*>(speed.data()) = 0.8f;
        std::copy(ids.data(), ids.data() + ids.size(),
                  reinterpret_cast<int64_t*>(inputIdsTensor.data()));
        std::copy(voice.data(), voice.data() + voice.size(),
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
        prepareAudioOutputKokoro(&wavDataPtr, wavSize, 32, samples, data);

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
