//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <cstdint>
#include <string>

namespace ovms {

inline const std::string PYTHON_SESSION_SIDE_PACKET_TAG = "py";
inline const std::string LLM_SESSION_SIDE_PACKET_TAG = "llm";
inline const std::string IMAGE_GEN_SESSION_SIDE_PACKET_TAG = "pipes";
inline const std::string EMBEDDINGS_SESSION_SIDE_PACKET_TAG = "embeddings_servable";
inline const std::string RERANK_SESSION_SIDE_PACKET_TAG = "rerank_servable";
inline const std::string STT_SESSION_SIDE_PACKET_TAG = "s2t_servable";
inline const std::string TTS_SESSION_SIDE_PACKET_TAG = "t2s_servable";
inline const std::string PYTHON_SIDE_PACKET_NAME = "py";
inline const std::string LLM_SESSION_PACKET_NAME = "llm";
inline constexpr int64_t STARTING_TIMESTAMP_VALUE = 0;

}  // namespace ovms
