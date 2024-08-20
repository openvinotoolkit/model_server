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
#include <atomic>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

#include <fmt/ranges.h>
#include <openvino/openvino.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../profiler.hpp"
#include "../logging.hpp"

using namespace rapidjson;
using namespace ovms;

namespace mediapipe {

using InputDataType = std::string;
using OutputDataType = std::string;


class EmbeddingsCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

    mediapipe::Timestamp timestamp{0};
    std::chrono::time_point<std::chrono::system_clock> created;


public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<InputDataType>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<OutputDataType>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculator  [Node: {}] Open start", cc->NodeName());

        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "EmbeddingsCalculator [Node: {}] Open end", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        return absl::OkStatus();
    }
};
const std::string EmbeddingsCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string EmbeddingsCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(EmbeddingsCalculator);

}  // namespace mediapipe
