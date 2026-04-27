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
#include "side_packet_builder.hpp"

#include "graph_executor_constants.hpp"
#include "graph_side_packets.hpp"

namespace ovms {

void buildInputSidePackets(std::map<std::string, mediapipe::Packet>& inputSidePackets,
    const GraphSidePackets& sidePackets) {
    const auto ts = ::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE);
#if (PYTHON_DISABLE == 0)
    inputSidePackets[PYTHON_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<PythonNodeResourcesMap>(sidePackets.pythonNodeResourcesMap).At(ts);
#endif
    inputSidePackets[LLM_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<GenAiServableMap>(sidePackets.genAiServableMap).At(ts);
    inputSidePackets[LLM_EXECUTION_CONTEXT_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<GenAiExecutionContextMap>(sidePackets.genAiExecutionContextMap).At(ts);
    inputSidePackets[IMAGE_GEN_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<ImageGenerationPipelinesMap>(sidePackets.imageGenPipelinesMap).At(ts);
    inputSidePackets[EMBEDDINGS_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<EmbeddingsServableMap>(sidePackets.embeddingsServableMap).At(ts);
    inputSidePackets[RERANK_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<RerankServableMap>(sidePackets.rerankServableMap).At(ts);
    inputSidePackets[STT_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<SttServableMap>(sidePackets.sttServableMap).At(ts);
    inputSidePackets[TTS_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<TtsServableMap>(sidePackets.ttsServableMap).At(ts);
}

}  // namespace ovms
