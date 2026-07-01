//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../dags/pipelinedefinitionstatus.hpp"
#include "src/metrics/metric.hpp"
#include "../model_metric_reporter.hpp"
#include "../single_version_servable_definition.hpp"
#include "../tensorinfo_fwd.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "mediapipegraphconfig.hpp"
#include "graph_side_packets.hpp"
#include "packettypes.hpp"
#include "graphqueue.hpp"

namespace ovms {
class MetricConfig;
class MetricRegistry;
class ServableNameChecker;
class MediapipeGraphExecutor;
class Status;
class PythonBackend;

class MediapipeGraphDefinition : public SingleVersionServableDefinition {
public:
    virtual ~MediapipeGraphDefinition();
    MediapipeGraphDefinition(const std::string name,
        const MediapipeGraphConfig& config = MGC,
        MetricRegistry* registry = nullptr,
        const MetricConfig* metricConfig = nullptr,
        PythonBackend* pythonBackend = nullptr);

    const PipelineDefinitionStatus& getStatus() const override {
        return this->status;
    }
    const std::vector<std::string>& getLoraAliases() const { return loraAliases; }
    bool shouldHideBaseModelInRouting() const { return hideBaseModelInRouting; }

    const PipelineDefinitionStateCode getStateCode() const { return status.getStateCode(); }
    bool isAvailable() const override { return status.isAvailable(); }
    const tensor_map_t getInputsInfo() const override;
    const tensor_map_t getOutputsInfo() const override;
    const MediapipeGraphConfig& getMediapipeGraphConfig() const { return this->mgconfig; }
    MediapipeServableMetricReporter& getMetricReporter() const override { return *this->reporter; }
    Status create(std::unique_ptr<MediapipeGraphExecutor>& pipeline);

    Status reload(const ServableNameChecker& checker, const MediapipeGraphConfig& config);
    Status validate(const ServableNameChecker& checker);
    void retire();
    Status initializeNodes();
    bool isReloadRequired(const MediapipeGraphConfig& config) const;

    // Idle unload feature
    Status unload();
    // wakeUpIfUnloaded: thread-safe wrapper — holds lifecycleMtx, double-checks
    // the UNLOADED state, and calls wakeUp() exactly once; other concurrent callers
    // wait on the mutex then return immediately since the state is no longer UNLOADED.
    Status wakeUpIfUnloaded(const ServableNameChecker& checker);
    bool isIdleUnloadEnabled() const;
    bool shouldUnloadDueToIdle() const;

    // Test-only: backdate the last-activity timestamp by the given number of seconds
    // so idle-timeout behavior can be exercised deterministically without sleeping.
    void backdateLastActivityForTest(int64_t seconds) {
        int64_t nowNs = std::chrono::steady_clock::now().time_since_epoch().count();
        lastActivityTimeNs->store(nowNs - seconds * 1'000'000'000LL, std::memory_order_relaxed);
    }

    // Returns the shared active-inference counter so create() can hand it to the executor.
    // Not exposed in tests directly — use shouldUnloadDueToIdle() to observe the effect.
    const std::shared_ptr<std::atomic<int64_t>>& getActiveInferenceCount() const {
        return activeInferenceCount;
    }

    static const std::string SCHEDULER_CLASS_NAME;

protected:
    std::shared_ptr<GraphSidePackets> sidePacketMaps;

    struct ValidationResultNotifier {
        ValidationResultNotifier(PipelineDefinitionStatus& status, std::condition_variable& loadedNotify) :
            status(status),
            loadedNotify(loadedNotify) {
        }
        ~ValidationResultNotifier() {
            if (passed) {
                status.handle(ValidationPassedEvent());
                loadedNotify.notify_all();
            } else {
                status.handle(ValidationFailedEvent());
            }
        }
        bool passed = false;

    private:
        PipelineDefinitionStatus& status;
        std::condition_variable& loadedNotify;
    };

    virtual Status validateForConfigFileExistence();
    Status resolveGraphQueueSize();
    Status validateForConfigLoadableness();

    Status setStreamTypes();
    Status dryInitializeTest();
    Status initializeQueueIfRequired();

    std::string chosenConfig;
    static MediapipeGraphConfig MGC;

    bool passKfsRequestFlag;
    std::unordered_map<std::string, mediapipe_packet_type_enum> inputTypes;
    std::unordered_map<std::string, mediapipe_packet_type_enum> outputTypes;
    PipelineDefinitionStatus status;

    MediapipeGraphConfig mgconfig;
    ::mediapipe::CalculatorGraphConfig config;

    Status createInputsInfo();
    Status createOutputsInfo();
    Status createInputSidePacketsInfo();

    mutable std::shared_mutex metadataMtx;

private:
    StatusCode notLoadedYetCode() const override;
    StatusCode notLoadedAnymoreCode() const override;

    tensor_map_t inputsInfo;
    tensor_map_t outputsInfo;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::string> inputSidePacketNames;

    std::vector<std::string> loraAliases;
    // When using LoRA adapters with STATIC/FUSE Adapter Config mode, lora weights are always active
    // and since that's true we want to expose only lora aliases/composite aliases in routing, and hide base model.
    bool hideBaseModelInRouting = false;

    PythonBackend* pythonBackend;

    std::unique_ptr<MediapipeServableMetricReporter> reporter;
    std::shared_ptr<GraphQueue> queue;

    // Idle unload: timestamp (nanoseconds from steady_clock epoch) of the last
    // inference activity. Updated in create() on every inference acquisition and
    // when an in-flight inference finishes (via ActiveInferenceGuard destructor).
    // Held as shared_ptr so executors can safely write to it even after the
    // definition is retired/destroyed — the atomic outlives the definition.
    std::shared_ptr<std::atomic<int64_t>> lastActivityTimeNs;

    // Count of inferences currently executing on this graph. Incremented when a
    // MediapipeGraphExecutor is created (in create()) via the executor's RAII
    // ActiveInferenceGuard, and decremented when that executor is destroyed (after
    // the caller finishes infer()/inferStream()). The count is therefore held for
    // the executor's lifetime, which spans the inference. A non-zero value prevents
    // shouldUnloadDueToIdle()/unload() from tearing down the definition.
    // Shared_ptr so MediapipeGraphExecutor can hold a copy safely beyond the
    // create() call — the executor owns the counter reference for its lifetime.
    std::shared_ptr<std::atomic<int64_t>> activeInferenceCount;

    // Cached copy of mgconfig.getIdleUnloadTimeoutSeconds() so the watcher thread can
    // read it lock-free. mgconfig itself is only safe to read under lifecycleMtx
    // (reload() reassigns it). Updated in the constructor and in reload() (under the
    // lock) whenever mgconfig is assigned.
    std::atomic<int64_t> idleUnloadTimeoutSecondsCache{0};

    // Serializes ALL per-definition lifecycle mutations (reload/retire/unload/wakeUp)
    // so they are mutually exclusive regardless of which thread runs them or which
    // outer lock (ModelManager::configMtx) is held by the caller. This is required
    // because unload() runs on the watcher thread (no configMtx) while reload()/retire()
    // run on the config thread (under configMtx) and they mutate the same per-definition
    // state (this->status variant, this->sidePacketMaps).
    // Recursive because wakeUpIfUnloaded() holds it and calls reload(), which also takes it.
    // Lock ordering is one-directional: configMtx -> lifecycleMtx. Nothing here ever
    // acquires configMtx, so no deadlock is possible.
    mutable std::recursive_mutex lifecycleMtx;
};
}  // namespace ovms
