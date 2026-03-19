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
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../dags/pipelinedefinitionstatus.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../model_metric_reporter.hpp"
#include "../single_version_servable_definition.hpp"
#include "../tensorinfo.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "mediapipegraphconfig.hpp"
#include "packettypes.hpp"

namespace ovms {
class MetricConfig;
class MetricRegistry;
class ServableNameChecker;
class MediapipeGraphExecutor;
class Status;
class PythonBackend;
class PythonNodeResources;
struct EmbeddingsServable;
class GenAiServable;
struct ImageGenerationPipelines;
struct RerankServable;
struct SttServable;
class TtsServable;
using PythonNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>;
using GenAiServableMap = std::unordered_map<std::string, std::shared_ptr<GenAiServable>>;
using RerankServableMap = std::unordered_map<std::string, std::shared_ptr<RerankServable>>;
using SttServableMap = std::unordered_map<std::string, std::shared_ptr<SttServable>>;
using TtsServableMap = std::unordered_map<std::string, std::shared_ptr<TtsServable>>;
using EmbeddingsServableMap = std::unordered_map<std::string, std::shared_ptr<EmbeddingsServable>>;
using ImageGenerationPipelinesMap = std::unordered_map<std::string, std::shared_ptr<ImageGenerationPipelines>>;

struct GraphSidePackets {
    PythonNodeResourcesMap pythonNodeResourcesMap;
    GenAiServableMap genAiServableMap;
    ImageGenerationPipelinesMap imageGenPipelinesMap;
    EmbeddingsServableMap embeddingsServableMap;
    RerankServableMap rerankServableMap;
    SttServableMap sttServableMap;
    TtsServableMap ttsServableMap;
    void clear() {
        pythonNodeResourcesMap.clear();
        genAiServableMap.clear();
        imageGenPipelinesMap.clear();
        embeddingsServableMap.clear();
        rerankServableMap.clear();
        sttServableMap.clear();
        ttsServableMap.clear();
    }
    bool empty() {
        return (pythonNodeResourcesMap.empty() &&
                genAiServableMap.empty() &&
                imageGenPipelinesMap.empty() &&
                embeddingsServableMap.empty() &&
                rerankServableMap.empty() &&
                sttServableMap.empty() &&
                ttsServableMap.empty());
    }
};

class MediapipeGraphDefinition : public SingleVersionServableDefinition {

public:
    virtual ~MediapipeGraphDefinition();
    MediapipeGraphDefinition(const std::string name,
        const MediapipeGraphConfig& config = MGC,
        MetricRegistry* registry = nullptr,
        const MetricConfig* metricConfig = nullptr,
        PythonBackend* pythonBackend = nullptr);

    const std::string& getName() const override { return SingleVersionServableDefinition::getName(); }
    const PipelineDefinitionStatus& getStatus() const override {
        return this->status;
    }

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

    static const std::string SCHEDULER_CLASS_NAME;
    static const std::string PYTHON_NODE_CALCULATOR_NAME;
    static const std::string LLM_NODE_CALCULATOR_NAME;
    static const std::string IMAGE_GEN_CALCULATOR_NAME;
    static const std::string EMBEDDINGS_NODE_CALCULATOR_NAME;
    static const std::string RERANK_NODE_CALCULATOR_NAME;
    static const std::string STT_NODE_CALCULATOR_NAME;
    static const std::string TTS_NODE_CALCULATOR_NAME;

protected:
    GraphSidePackets sidePacketMaps;

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
    Status validateForConfigLoadableness();

    Status setStreamTypes();
    Status dryInitializeTest();
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

    PythonBackend* pythonBackend;

    std::unique_ptr<MediapipeServableMetricReporter> reporter;
};
}  // namespace ovms
