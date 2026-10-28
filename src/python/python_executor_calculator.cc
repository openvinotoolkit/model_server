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
#include <memory>
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4005 6001 6385 6386 6326 6011 6246 4456)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "py_object_handle.hpp"
#include "python_calculators_plugin_api.hpp"
#include "python_calculators_plugin_loader.hpp"

namespace mediapipe {

static constexpr const char* PYTHON_SESSION_SIDE_PACKET_TAG = "PYTHON_NODE_RESOURCES";
static constexpr const char* LOOPBACK_TAG = "LOOPBACK";

// Python execution is delegated to libpython_calculators, so this calculator does not link libpython.
class PythonExecutorCalculator : public CalculatorBase {
    struct GeneratorDeleter {
        const ovms::PythonCalculatorsPluginApi* api = nullptr;
        void operator()(void* generator) const {
            api->releaseGenerator(generator);
        }
    };

    const ovms::PythonCalculatorsPluginApi* api = nullptr;
    std::shared_ptr<ovms::PythonNodeResources> nodeResources;
    std::unique_ptr<void, GeneratorDeleter> generator{nullptr, GeneratorDeleter{}};
    bool hasLoopback{false};
    // The calculator manages timestamp for outputs to work independently of inputs
    // this way we can support timestamp continuity for more than one request in streaming scenario.
    mediapipe::Timestamp outputTimestamp;

    static void setPacketTypes(PacketTypeSet& packetTypes) {
        for (const std::string& tag : packetTypes.GetTags()) {
            if (tag == LOOPBACK_TAG) {
                packetTypes.Tag(tag).Set<bool>();
            } else {
                packetTypes.Tag(tag).Set<ovms::PyObjectHandle>();
            }
        }
    }

    static absl::Status executionFailed(CalculatorContext* cc, const std::string& message) {
        LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << message;
        return absl::Status(absl::StatusCode::kInternal, "Error occurred during graph execution");
    }

    // Takes ownership of all outputs; outputs without a matching output stream are released.
    void pushOutputs(CalculatorContext* cc, std::vector<ovms::PythonNodeOutput>& outputs, bool pushLoopback) {
        for (auto& output : outputs) {
            auto handle = std::make_unique<ovms::PyObjectHandle>(output.object, api->releaseObject);
            if (cc->Outputs().HasTag(output.tag)) {
                cc->Outputs().Tag(output.tag).Add(handle.release(), outputTimestamp);
            }
        }
        outputs.clear();
        if (pushLoopback) {
            outputTimestamp++;
            cc->Outputs().Tag(LOOPBACK_TAG).Add(std::make_unique<bool>(true).release(), outputTimestamp);
        }
    }

    bool receivedNewData(CalculatorContext* cc) {
        for (const std::string& tag : cc->Inputs().GetTags()) {
            if (tag != LOOPBACK_TAG && !cc->Inputs().Tag(tag).IsEmpty()) {
                return true;
            }
        }
        return false;
    }

    absl::Status generate(CalculatorContext* cc) {
        std::vector<ovms::PythonNodeOutput> outputs;
        std::string message;
        bool produced = false;
        const auto result = api->generatorNext(*nodeResources, generator.get(), outputs, produced, message);
        if (produced) {
            pushOutputs(cc, outputs, true);
        }
        if (result != ovms::PythonPluginResult::OK) {
            return executionFailed(cc, message);
        }
        if (!produced) {
            LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] finished generating. Resetting the generator.";
            generator.reset();
        }
        return absl::OkStatus();
    }

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());

        if (cc->Inputs().HasTag(LOOPBACK_TAG) != cc->Outputs().HasTag(LOOPBACK_TAG))
            return absl::Status(absl::StatusCode::kInvalidArgument, "If LOOPBACK is used, it must be defined on both input and output of the node");

        setPacketTypes(cc->Inputs());
        setPacketTypes(cc->Outputs());
        cc->InputSidePackets().Tag(PYTHON_SESSION_SIDE_PACKET_TAG).Set<ovms::PythonNodeResourcesMap>();
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Open start";
        hasLoopback = cc->Inputs().HasTag(LOOPBACK_TAG);
        api = ovms::getPythonCalculatorsPluginApi();
        if (api == nullptr) {
            return absl::Status(absl::StatusCode::kFailedPrecondition, "Python calculators plugin is not loaded");
        }
        generator = std::unique_ptr<void, GeneratorDeleter>(nullptr, GeneratorDeleter{api});

        const auto& nodeResourcesMap = cc->InputSidePackets().Tag(PYTHON_SESSION_SIDE_PACKET_TAG).Get<ovms::PythonNodeResourcesMap>();
        auto it = nodeResourcesMap.find(cc->NodeName());
        if (it == nodeResourcesMap.end()) {
            LOG(INFO) << "Could not find initialized Python node named: " << cc->NodeName();
            RET_CHECK(false);
        }
        nodeResources = it->second;
        if (nodeResources == nullptr || !api->hasPythonBackend(*nodeResources)) {
            return absl::Status(absl::StatusCode::kFailedPrecondition, "Python backend is not available for PythonExecutorCalculator");
        }
        outputTimestamp = mediapipe::Timestamp(mediapipe::Timestamp::Unset());
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Process start";
        if (generator != nullptr) {
            if (receivedNewData(cc)) {
                LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Node is already processing data. Create new stream for another request.";
                return absl::Status(absl::StatusCode::kResourceExhausted, "Node is already processing data. Create new stream for another request.");
            }
            if (auto status = generate(cc); !status.ok()) {
                return status;
            }
        } else {
            // If execute yields, first request sets initial timestamp to input timestamp, then each cycle increments it.
            // If execute returns, input timestamp is also output timestamp.
            outputTimestamp = cc->InputTimestamp();

            std::vector<const void*> inputs;
            for (const std::string& tag : cc->Inputs().GetTags()) {
                if (tag == LOOPBACK_TAG) {
                    continue;
                }
                if (cc->Inputs().Tag(tag).IsEmpty()) {
                    LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Received empty packet on input: " << tag
                              << ". Execution will continue without that input.";
                    continue;
                }
                inputs.push_back(cc->Inputs().Tag(tag).Get<ovms::PyObjectHandle>().get());
            }

            std::vector<ovms::PythonNodeOutput> outputs;
            void* newGenerator = nullptr;
            std::string message;
            const auto result = api->execute(*nodeResources, inputs, outputs, newGenerator, message);
            if (result != ovms::PythonPluginResult::OK) {
                return executionFailed(cc, message);
            }
            if (newGenerator != nullptr) {
                generator = std::unique_ptr<void, GeneratorDeleter>(newGenerator, GeneratorDeleter{api});
                if (!hasLoopback) {
                    generator.reset();
                    return executionFailed(cc, "Bad python node configuration. Execute yielded, but LOOPBACK is not defined in the node");
                }
                if (auto status = generate(cc); !status.ok()) {
                    return status;
                }
            } else {
                pushOutputs(cc, outputs, false);
            }
        }
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PythonExecutorCalculator);
}  // namespace mediapipe
