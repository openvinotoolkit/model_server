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
#include "src/mediapipe_internal/node_initializer.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/live_graph_config.hpp"
#include "mediapipe/framework/formats/detection.pb.h"

namespace ovms {
namespace {
constexpr char ByteTrackCalculatorName[] = "ByteTrackCalculator";

class VideoTrackingNodeInitializer final : public NodeInitializer {
public:
    bool matches(const std::string&) const override {
        return false;
    }

    Status initialize(const ::mediapipe::CalculatorGraphConfig_Node&,
        const std::string&, const std::string&, GraphSidePackets&, GraphMetadata&,
        PythonBackend*, const MetricProvider*) override {
        return StatusCode::OK;
    }

    Status prepareLiveGraph(const ::mediapipe::CalculatorGraphConfig& config,
        GraphSidePackets& sidePackets) override {
        bool hasByteTrack = false;
        for (const auto& node : config.node()) {
            hasByteTrack = hasByteTrack || node.calculator() == ByteTrackCalculatorName;
        }
        if (!hasByteTrack)
            return StatusCode::OK;

        auto profile = makeLiveGraphConfig(config, LiveGraphVideoInputQueueSize,
            {{"video", LiveGraphStreamCodec::imageFrame()}},
            {{"tracked_detections", LiveGraphStreamCodec::typed<std::vector<mediapipe::Detection>>()}});

        auto* nodes = profile->graph.mutable_node();
        for (int index = nodes->size() - 1; index >= 0; --index) {
            const auto& node = nodes->Get(index);
            if (node.calculator() == "DetectionColorByIdCalculator" ||
                node.calculator() == "AnnotationOverlayCalculator") {
                nodes->DeleteSubrange(index, 1);
                continue;
            }
            auto* mutableNode = nodes->Mutable(index);
            for (int streamIndex = 0; streamIndex < mutableNode->input_stream_size(); ++streamIndex) {
                const std::string& stream = mutableNode->input_stream(streamIndex);
                if (stream == "IMAGE:input_video")
                    mutableNode->set_input_stream(streamIndex, "IMAGE:video");
            }
        }

        return registerLiveGraphProfile(VideoTrackingLiveGraphProfile, std::move(profile), sidePackets);
    }
};

const bool videoTrackingNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<VideoTrackingNodeInitializer>());
    return true;
}();
}  // namespace
}  // namespace ovms
