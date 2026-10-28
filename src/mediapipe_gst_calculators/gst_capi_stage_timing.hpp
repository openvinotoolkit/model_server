// Copyright (c) 2026 Intel Corporation
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

#ifndef SRC_MEDIAPIPE_GST_CALCULATORS_GST_CAPI_STAGE_TIMING_HPP_
#define SRC_MEDIAPIPE_GST_CALCULATORS_GST_CAPI_STAGE_TIMING_HPP_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace mediapipe {
class GstCapiStageTiming {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    enum class Stage : size_t {
        Process,
        ReadPull,
        SurfaceExtract,
        SurfaceSync,
        RemoveData,
        SetData,
        Inference,
        ResponseAccess,
        ResponseDelete,
        Count
    };

    GstCapiStageTiming() {
        const char* enabled = std::getenv("OVMS_MPI_E4_DIAGNOSTICS");
        enabled_ = enabled && std::string(enabled) == "stages";
        const char* surfaceSync = std::getenv("OVMS_MPI_E4_VA_SYNC");
        surfaceSyncEnabled_ = enabled_ && surfaceSync && std::string(surfaceSync) == "on";
        const char* warmup = std::getenv("OVMS_MPI_E4_WARMUP_FRAMES");
        if (warmup && *warmup)
            warmupFrames_ = std::atoi(warmup);
    }

    bool beginFrame() {
        if (!enabled_)
            return false;
        return framesSeen_++ >= warmupFrames_;
    }

    bool surfaceSyncEnabled() const {
        return surfaceSyncEnabled_;
    }

    TimePoint mark() const {
        return enabled_ ? Clock::now() : TimePoint{};
    }

    void add(Stage stage, TimePoint start, TimePoint stop, bool measured) {
        if (enabled_ && measured)
            durations_[static_cast<size_t>(stage)] += stop - start;
    }

    void finishFrame(TimePoint start, bool measured) {
        if (!enabled_ || !measured)
            return;
        add(Stage::Process, start, Clock::now(), true);
        ++measuredFrames_;
    }

    void report(const char* calculator) const {
        if (!enabled_ || measuredFrames_ == 0)
            return;
        const auto millisecondsPerFrame = [&](Stage stage) {
            return std::chrono::duration<double, std::milli>(durations_[static_cast<size_t>(stage)]).count() /
                   measuredFrames_;
        };
        const double processMs = millisecondsPerFrame(Stage::Process);
        std::cout << std::fixed << std::setprecision(6)
                  << "E4_PROCESS case=" << calculator << " measured=" << measuredFrames_
                  << " process_wall_ms_per_frame=" << processMs
                  << " process_fps=" << 1000.0 / processMs
                  << " read_pull_ms_per_frame=" << millisecondsPerFrame(Stage::ReadPull)
                  << " surface_extract_ms_per_frame=" << millisecondsPerFrame(Stage::SurfaceExtract)
                  << " surface_sync_ms_per_frame=" << millisecondsPerFrame(Stage::SurfaceSync)
                  << " remove_data_ms_per_frame=" << millisecondsPerFrame(Stage::RemoveData)
                  << " set_data_ms_per_frame=" << millisecondsPerFrame(Stage::SetData)
                  << " ovms_inference_ms_per_frame=" << millisecondsPerFrame(Stage::Inference)
                  << " response_access_ms_per_frame=" << millisecondsPerFrame(Stage::ResponseAccess)
                  << " response_delete_ms_per_frame=" << millisecondsPerFrame(Stage::ResponseDelete)
                  << std::endl;
    }

private:
    bool enabled_ = false;
    bool surfaceSyncEnabled_ = false;
    int warmupFrames_ = 20;
    size_t framesSeen_ = 0;
    size_t measuredFrames_ = 0;
    std::array<std::chrono::nanoseconds, static_cast<size_t>(Stage::Count)> durations_{};
};
}  // namespace mediapipe

#endif  // SRC_MEDIAPIPE_GST_CALCULATORS_GST_CAPI_STAGE_TIMING_HPP_