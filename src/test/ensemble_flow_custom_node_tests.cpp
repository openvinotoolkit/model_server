//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../custom_node.hpp"
#include "../custom_node_library_manager.hpp"
#include "../dl_node.hpp"
#include "../entry_node.hpp"
#include "../exit_node.hpp"
#include "../node_library.hpp"
#include "../node_library_utils.hpp"
#include "../pipelinedefinition.hpp"
#include "../stringutils.hpp"
#include "test_utils.hpp"

using namespace ovms;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

class EnsembleFlowCustomNodePipelineExecutionTest : public TestWithTempDir {
protected:
    void SetUp() override {
        TestWithTempDir::SetUp();
        CustomNodeLibraryManager manager;
        ASSERT_EQ(manager.loadLibrary(
                      this->libraryName,
                      this->libraryPath),
            StatusCode::OK);
        ASSERT_EQ(manager.getLibrary(
                      this->libraryName,
                      this->library),
            StatusCode::OK);
        dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineOutputName,
            InferenceEngine::Precision::FP32,
            DUMMY_MODEL_SHAPE,
            InferenceEngine::Layout::NC);
        dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineInputName,
            InferenceEngine::Precision::FP32,
            DUMMY_MODEL_SHAPE,
            InferenceEngine::Layout::NC);
    }

    template <typename T>
    void prepareRequest(const std::vector<T>& data) {
        this->prepareRequest(this->request, data);
    }

    template <typename T>
    void prepareRequest(PredictRequest& request, const std::vector<T>& data, const std::string& inputName = pipelineInputName, const shape_t shape = {}) {
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[inputName];
        proto.set_dtype(tensorflow::DataTypeToEnum<T>::value);
        proto.mutable_tensor_content()->assign((char*)data.data(), data.size() * sizeof(T));
        if (shape.size()) {
            for (auto& dim : shape) {
                proto.mutable_tensor_shape()->add_dim()->set_size(dim);
            }
        } else {
            proto.mutable_tensor_shape()->add_dim()->set_size(1);
            proto.mutable_tensor_shape()->add_dim()->set_size(data.size());
        }
    }

    template <typename T>
    std::unique_ptr<Pipeline> prepareSingleNodePipelineWithLibraryMock() {
        const std::vector<float> inputValues{3.5, 2.1, -0.2};
        auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineInputName,
            InferenceEngine::Precision::FP32,
            shape_t{1, 3},
            InferenceEngine::Layout::NC);
        const tensor_map_t inputsInfo{{pipelineInputName, inputTensorInfo}};
        this->prepareRequest(inputValues);
        auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
        const tensor_map_t outputsInfo{{pipelineOutputName, dagDummyModelOutputTensorInfo}};
        auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);
        auto custom_node = std::make_unique<CustomNode>(
            customNodeName,
            createLibraryMock<T>(),
            parameters_t{});

        auto pipeline = std::make_unique<Pipeline>(*input_node, *output_node);
        pipeline->connect(*input_node, *custom_node, {{pipelineInputName, customNodeInputName}});
        pipeline->connect(*custom_node, *output_node, {{customNodeOutputName, pipelineOutputName}});

        pipeline->push(std::move(input_node));
        pipeline->push(std::move(custom_node));
        pipeline->push(std::move(output_node));
        return pipeline;
    }

    template <typename T>
    void checkResponse(std::vector<T> data, std::function<T(T)> op) {
        this->checkResponse(this->pipelineOutputName, data, op);
    }

    template <typename T>
    void checkResponse(const std::string& outputName, std::vector<T> data, std::function<T(T)> op) {
        this->checkResponse(outputName, this->response, data, op);
    }

    template <typename T>
    void checkResponse(const std::string& outputName, const PredictResponse& response, const std::vector<T>& data, const shape_t& shape) {
        ASSERT_TRUE(response.outputs().contains(outputName)) << outputName;
        const auto& proto = response.outputs().at(outputName);

        ASSERT_EQ(proto.tensor_content().size(), data.size() * sizeof(T));
        ASSERT_EQ(proto.tensor_shape().dim_size(), shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            ASSERT_EQ(proto.tensor_shape().dim(i).size(), shape[i]);
        }

        auto* ptr = reinterpret_cast<const T*>(proto.tensor_content().c_str());
        const std::vector<T> actual(ptr, ptr + data.size());
        for (size_t i = 0; i < actual.size(); i++) {
            EXPECT_NEAR(actual[i], data[i], 0.001) << " i is: " << i;
        }
    }

    template <typename T>
    void checkResponse(const std::string& outputName, const PredictResponse& response, std::vector<T> data, std::function<T(T)> op) {
        std::transform(data.begin(), data.end(), data.begin(), op);
        ASSERT_TRUE(response.outputs().contains(outputName));
        const auto& proto = response.outputs().at(outputName);

        ASSERT_EQ(proto.tensor_content().size(), data.size() * sizeof(T));
        ASSERT_EQ(proto.tensor_shape().dim_size(), 2);
        ASSERT_EQ(proto.tensor_shape().dim(0).size(), 1);
        ASSERT_EQ(proto.tensor_shape().dim(1).size(), data.size());

        auto* ptr = reinterpret_cast<const T*>(proto.tensor_content().c_str());

        const std::vector<T> actual(ptr, ptr + data.size());

        for (size_t i = 0; i < actual.size(); i++) {
            EXPECT_NEAR(actual[i], data[i], 0.001);
        }
    }

    PredictRequest request;
    PredictResponse response;

    NodeLibrary library;

    const std::string customNodeName = "add_sub_node";
    const std::string libraryName = "add_sub_lib";
    const std::string libraryPath = "/ovms/bazel-bin/src/lib_node_add_sub.so";
    const std::string customNodeInputName = "input_numbers";
    const std::string customNodeOutputName = "output_numbers";
    static constexpr const char* pipelineInputName = "pipeline_input";
    const std::string pipelineOutputName = "pipeline_output";
    std::shared_ptr<ovms::TensorInfo> dagDummyModelOutputTensorInfo;
    std::shared_ptr<ovms::TensorInfo> dagDummyModelInputTensorInfo;
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, AddSubCustomNode) {
    // Most basic configuration, just process single add-sub custom node pipeline request
    // input  add-sub  output
    //  O------->O------->O
    const std::vector<float> inputValues{3.2, 5.7, -2.4};
    this->prepareRequest(inputValues);

    const float addValue = 2.5;
    const float subValue = 4.8;

    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineInputName,
        InferenceEngine::Precision::FP32,
        shape_t{1, 3},
        InferenceEngine::Layout::NC);
    const tensor_map_t inputsInfo{{pipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(pipelineOutputName,
        InferenceEngine::Precision::FP32,
        shape_t{1, 3},
        InferenceEngine::Layout::NC);
    const tensor_map_t outputsInfo{{pipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);
    auto custom_node = std::make_unique<CustomNode>(customNodeName, library,
        parameters_t{
            {"add_value", std::to_string(addValue)},
            {"sub_value", std::to_string(subValue)}});

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *custom_node, {{pipelineInputName, customNodeInputName}});
    pipeline.connect(*custom_node, *output_node, {{customNodeOutputName, pipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(custom_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), 1);

    this->checkResponse<float>(inputValues, [addValue, subValue](float value) -> float {
        return value + addValue - subValue;
    });
}

class EnsembleFlowCustomNodeAndDemultiplexerGatherPipelineExecutionTest : public EnsembleFlowCustomNodePipelineExecutionTest {
protected:
    ConstructorEnabledModelManager modelManager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    CustomNodeLibraryManager manager;
    NodeLibrary differentOpsLibrary;
    NodeLibrary chooseMaxLibrary;
    const std::string differentOpsLibraryName{"different_ops"};
    const std::string chooseMaxLibraryName{"choose_max"};
    const std::string differentOpsLibraryPath{"/ovms/bazel-bin/src/lib_node_perform_different_operations.so"};
    const std::string chooseMaxLibraryPath{"/ovms/bazel-bin/src/lib_node_choose_maximum.so"};
    const std::string pipelineInputName = "pipeline_input";
    const std::string pipelineOutputName = "pipeline_output";
    const std::string pipelineFactorsName = "pipeline_factors";
    const std::string chooseMaxInputName = "input_tensors";
    const std::string chooseMaxOutputName = "maximum_tensor";
    const std::string differentOpsInputName = "input_numbers";
    const std::string differentOpsFactorsInputName = "op_factors";
    const std::string differentOpsOutputName = "different_ops_results";
    const std::string differentOpsFactorsOutputName = "different_ops_factors";
    const std::unordered_map<std::string, std::string> differentOpsOutputAlias{{differentOpsOutputName, differentOpsOutputName}};
    const std::unordered_map<std::string, std::string> chooseMaxOutputAlias{{chooseMaxOutputName, chooseMaxOutputName}};
    const std::string dummyNodeName = "dummy";
    const std::string differentOpsNodeName{"different-ops-node"};
    const std::string chooseMaxNodeName{"choose-max-node"};
    const uint32_t demultiplyCount = 4;  // different ops library has (1,4,10) as output

    void SetUp() override {
        // increasing default nireq == 1 to speed up the tests
        // in multilayered demultiplication we still will have more than
        // 16 concurrent inferences
        config.setNireq(16);
        ASSERT_EQ(modelManager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
        ASSERT_EQ(manager.loadLibrary(
                      differentOpsLibraryName,
                      differentOpsLibraryPath),
            StatusCode::OK);
        ASSERT_EQ(manager.getLibrary(
                      differentOpsLibraryName,
                      differentOpsLibrary),
            StatusCode::OK);
        ASSERT_EQ(manager.loadLibrary(
                      chooseMaxLibraryName,
                      chooseMaxLibraryPath),
            StatusCode::OK);
        ASSERT_EQ(manager.getLibrary(
                      chooseMaxLibraryName,
                      chooseMaxLibrary),
            StatusCode::OK);
        dagDummyModelOutputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineOutputName,
            InferenceEngine::Precision::FP32,
            DUMMY_MODEL_SHAPE,
            InferenceEngine::Layout::NC);
        dagDummyModelInputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineInputName,
            InferenceEngine::Precision::FP32,
            DUMMY_MODEL_SHAPE,
            InferenceEngine::Layout::NC);
    }
};

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerGatherPipelineExecutionTest, MultipleDemultiplexerDummyGathersIntertwinedLevels) {
    // Most basic configuration, just process single add-sub custom node pipeline request
    // input  (differentOps    dummy   chooseMax ) XN    output
    //  O-----(----->O---------->O------->O------>...----->O
    const uint demultiplicationLayersCount = 10;
    // values choosen in a way that first choosen different ops result will be addition. all following ones will be multiplications
    const std::vector<float> inputValues{0.2, 0.7, -0.4, -0.1, 0.0001, -0.8, 0.7, 0.8, 0.9, 0.1};
    const std::vector<float> inputFactors{1, -1, 2, 2};
    parameters_t parameters{
        {"selection_criteria", "MAXIMUM_MAXIMUM"}};
    // create expected output -> it is dependent from input values & DAG topology
    auto expectedResult = inputValues;
    std::transform(expectedResult.begin(), expectedResult.end(), expectedResult.begin(),
        [demultiplicationLayersCount, inputFactors](float f) {
            for (size_t iterations = 0; iterations < demultiplicationLayersCount; ++iterations) {
                // input values are prepared in a way that the first layer will choose adding operation tensor
                if (iterations == 0) {
                    f += inputFactors[0];
                } else {
                    f *= inputFactors[2];  // different ops mutliply will be choosen
                }
                f += 1;  // dummy
            }
            return f;
        });
    PredictRequest predictRequest;
    this->prepareRequest(predictRequest, inputValues, pipelineInputName);
    this->prepareRequest(predictRequest, inputFactors, pipelineFactorsName);

    // create pipeline
    std::vector<std::unique_ptr<Node>> nodes(2 + 3 * demultiplicationLayersCount);  // entry + exit + (choose + differentOps + dummy) * layerCount
    const tensor_map_t inputsInfo{{pipelineInputName, dagDummyModelInputTensorInfo}, {pipelineFactorsName,
                                                                                         std::make_shared<ovms::TensorInfo>(pipelineFactorsName,
                                                                                             InferenceEngine::Precision::FP32,
                                                                                             shape_t{1, 4},
                                                                                             InferenceEngine::Layout::NC)}};
    nodes[0] = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    const tensor_map_t outputsInfo{{pipelineOutputName, dagDummyModelOutputTensorInfo}};
    nodes[1] = std::make_unique<ExitNode>(&response, outputsInfo);
    size_t i = 2;
    for (size_t demultiplicationLayer = 0; demultiplicationLayer < demultiplicationLayersCount; ++demultiplicationLayer) {
        nodes[i++] = std::make_unique<CustomNode>(differentOpsNodeName + "-" + std::to_string(demultiplicationLayer), differentOpsLibrary, parameters_t{}, differentOpsOutputAlias, demultiplyCount);
        nodes[i++] = std::make_unique<DLNode>(dummyNodeName + "-" + std::to_string(demultiplicationLayer), "dummy", std::nullopt, modelManager);
        nodes[i++] = std::make_unique<CustomNode>(chooseMaxNodeName + "-" + std::to_string(demultiplicationLayer), chooseMaxLibrary, parameters, chooseMaxOutputAlias, std::nullopt, std::set<std::string>({differentOpsNodeName + "-" + std::to_string(demultiplicationLayer)}));
    }

    Pipeline pipeline(dynamic_cast<EntryNode&>(*nodes[0]), dynamic_cast<ExitNode&>(*nodes[1]));
    i = 2;
    for (size_t demultiplicationLayer = 0; demultiplicationLayer < demultiplicationLayersCount; ++demultiplicationLayer) {
        if (i == 2) {  // first node after entry
            pipeline.connect(*nodes[0], *nodes[i], {{pipelineFactorsName, differentOpsFactorsInputName}, {pipelineInputName, differentOpsInputName}});
        } else {  // node inside pipeline
            pipeline.connect(*nodes[0], *nodes[i], {{pipelineFactorsName, differentOpsFactorsInputName}});
        }
        pipeline.connect(*nodes[i], *nodes[i + 1], {{differentOpsOutputName, DUMMY_MODEL_INPUT_NAME}});
        pipeline.connect(*nodes[i + 1], *nodes[i + 2], {{DUMMY_MODEL_OUTPUT_NAME, chooseMaxInputName}});
        if ((i + 3) != (2 + 3 * demultiplicationLayersCount)) {  // connect different ops to choose max
            pipeline.connect(*nodes[i + 2], *nodes[i + 3], {{chooseMaxOutputName, differentOpsInputName}});
        } else {  // if last connect to exit node
            pipeline.connect(*nodes[i + 2], *nodes[1], {{chooseMaxOutputName, pipelineOutputName}});
        }
        i = i + 3;
    }
    for (auto& node : nodes) {
        pipeline.push(std::move(node));
    }

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), 1);
    this->checkResponse(pipelineOutputName, response, expectedResult, {1, 10});
}

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerGatherPipelineExecutionTest, MultipleDemultiplexerLevelsThenDummyThenMultipleGathers) {
    // Most basic configuration, just process single add-sub custom node pipeline request
    // input  (differentOps dummy)xN   chooseMax xN    output
    //  O-----(----->O------->O---...----->O---->...----->O
    const uint demultiplicationLayersCount = 4;
    // values choosen in a way that first choosen different ops result will be addition. all following ones will be multiplications
    const std::vector<float> inputValues{0.2, 0.7, -0.4, -0.1, 0.0001, -0.8, 0.7, 0.8, 0.9, 0.1};
    const std::vector<float> inputFactors{1, -1, 2, 2};
    parameters_t parameters{
        {"selection_criteria", "MAXIMUM_MAXIMUM"}};
    // create expected output -> it is dependent from input values & DAG topology
    auto expectedResult = inputValues;
    std::transform(expectedResult.begin(), expectedResult.end(), expectedResult.begin(),
        [demultiplicationLayersCount, inputFactors](float f) {
            for (size_t iterations = 0; iterations < demultiplicationLayersCount; ++iterations) {
                // input values are prepared in a way that the first layer will choose adding operation tensor
                if (iterations == 0) {
                    f += inputFactors[0];
                } else {
                    f *= inputFactors[2];  // different ops mutliply will be choosen
                }
                f += 1;  // dummy
            }
            return f;
        });
    PredictRequest predictRequest;
    this->prepareRequest(predictRequest, inputValues, pipelineInputName);
    this->prepareRequest(predictRequest, inputFactors, pipelineFactorsName);

    // create pipeline
    size_t nodesCount = 2 + 3 * demultiplicationLayersCount;  // entry + exit + (choose + differentOps + dummy) * layerCount
    std::vector<std::unique_ptr<Node>> nodes(nodesCount);
    const tensor_map_t inputsInfo{{pipelineInputName, dagDummyModelInputTensorInfo}, {pipelineFactorsName,
                                                                                         std::make_shared<ovms::TensorInfo>(pipelineFactorsName,
                                                                                             InferenceEngine::Precision::FP32,
                                                                                             shape_t{1, 4},
                                                                                             InferenceEngine::Layout::NC)}};
    nodes[0] = std::make_unique<EntryNode>(&predictRequest, inputsInfo);
    const tensor_map_t outputsInfo{{pipelineOutputName, dagDummyModelOutputTensorInfo}};
    nodes[nodesCount - 1] = std::make_unique<ExitNode>(&response, outputsInfo);
    size_t i = 1;
    for (size_t demultiplicationLayer = 0; demultiplicationLayer < demultiplicationLayersCount; ++demultiplicationLayer) {
        nodes[i++] = std::make_unique<CustomNode>(differentOpsNodeName + "-" + std::to_string(demultiplicationLayer), differentOpsLibrary, parameters_t{}, differentOpsOutputAlias, demultiplyCount);
        nodes[i++] = std::make_unique<DLNode>(dummyNodeName + "-" + std::to_string(demultiplicationLayer), "dummy", std::nullopt, modelManager);
        nodes[nodesCount - 1 - (i / 2)] = std::make_unique<CustomNode>(chooseMaxNodeName + "-" + std::to_string(demultiplicationLayer), chooseMaxLibrary, parameters, chooseMaxOutputAlias, std::nullopt, std::set<std::string>({differentOpsNodeName + "-" + std::to_string(demultiplicationLayer)}));
    }

    Pipeline pipeline(dynamic_cast<EntryNode&>(*nodes[0]), dynamic_cast<ExitNode&>(*nodes[nodesCount - 1]));
    i = 1;
    for (size_t demultiplicationLayer = 0; demultiplicationLayer < demultiplicationLayersCount; ++demultiplicationLayer) {
        if (i == 1) {  // first node after entry needs to connect to entry
            pipeline.connect(*nodes[0], *nodes[i], {{pipelineFactorsName, differentOpsFactorsInputName}, {pipelineInputName, differentOpsInputName}});
        }
        pipeline.connect(*nodes[i], *nodes[i + 1], {{differentOpsOutputName, DUMMY_MODEL_INPUT_NAME}});
        // pass factors further if +2 node is differentOps
        if (demultiplicationLayer != demultiplicationLayersCount - 1) {
            pipeline.connect(*nodes[i], *nodes[i + 2], {{differentOpsFactorsOutputName, differentOpsFactorsInputName}});
        }
        // in between different ops & dummy node
        if (demultiplicationLayer != demultiplicationLayersCount - 1) {  // all but last dummy connect to differentOps node
            pipeline.connect(*nodes[i + 1], *nodes[i + 2], {{DUMMY_MODEL_OUTPUT_NAME, differentOpsInputName}});
        } else {  // last dummy connects to chooseMax node
            pipeline.connect(*nodes[i + 1], *nodes[i + 2], {{DUMMY_MODEL_OUTPUT_NAME, chooseMaxInputName}});
        }
        if (demultiplicationLayer != 0) {  // in between choose max nodes
            pipeline.connect(*nodes[nodesCount - 1 - (demultiplicationLayer + 1)],
                *nodes[nodesCount - 1 - demultiplicationLayer], {{chooseMaxOutputName, chooseMaxInputName}});
        } else {  // connect last choose max to exit node
            pipeline.connect(*nodes[nodesCount - 1 - (demultiplicationLayer + 1)],
                *nodes[nodesCount - 1 - demultiplicationLayer], {{chooseMaxOutputName, pipelineOutputName}});
        }
        i = i + 2;
    }
    for (auto& node : nodes) {
        pipeline.push(std::move(node));
    }

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), 1);
    this->checkResponse(pipelineOutputName, response, expectedResult, {1, 10});
}

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, SeriesOfCustomNodes) {
    constexpr int N = 100;
    constexpr int PARAMETERS_PAIRS_COUNT = 2;
    static_assert(PARAMETERS_PAIRS_COUNT > 0);
    static_assert(N > PARAMETERS_PAIRS_COUNT);
    static_assert((N % PARAMETERS_PAIRS_COUNT) == 0);
    // input      add-sub x N      output
    //  O------->O->O...O->O------->O

    const std::vector<float> inputValues{3.2, 5.7, -2.4};
    this->prepareRequest(inputValues);

    const std::array<float, PARAMETERS_PAIRS_COUNT> addValues{1.5, -2.4};
    const std::array<float, PARAMETERS_PAIRS_COUNT> subValues{-5.1, 1.9};

    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineInputName,
        InferenceEngine::Precision::FP32,
        shape_t{1, 3},
        InferenceEngine::Layout::NC);
    const tensor_map_t inputsInfo{{pipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(pipelineOutputName,
        InferenceEngine::Precision::FP32,
        shape_t{1, 3},
        InferenceEngine::Layout::NC);
    const tensor_map_t outputsInfo{{pipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

    std::unique_ptr<CustomNode> custom_nodes[N];
    for (int i = 0; i < N; i++) {
        custom_nodes[i] = std::make_unique<CustomNode>(customNodeName + std::to_string(i), library,
            parameters_t{
                {"add_value", std::to_string(addValues[i % PARAMETERS_PAIRS_COUNT])},
                {"sub_value", std::to_string(subValues[i % PARAMETERS_PAIRS_COUNT])}});
    }

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *(custom_nodes[0]), {{pipelineInputName, customNodeInputName}});
    pipeline.connect(*(custom_nodes[N - 1]), *output_node, {{customNodeOutputName, pipelineOutputName}});
    for (int i = 0; i < N - 1; i++) {
        pipeline.connect(*(custom_nodes[i]), *(custom_nodes[i + 1]), {{customNodeOutputName, customNodeInputName}});
    }

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(output_node));
    for (auto& custom_node : custom_nodes) {
        pipeline.push(std::move(custom_node));
    }

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), 1);

    this->checkResponse<float>(inputValues, [N, addValues, subValues](float value) -> float {
        for (int i = 0; i < PARAMETERS_PAIRS_COUNT; i++) {
            value += (N / PARAMETERS_PAIRS_COUNT) * addValues[i];
            value -= (N / PARAMETERS_PAIRS_COUNT) * subValues[i];
        }
        return value;
    });
}

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, ParallelCustomNodes) {
    constexpr int N = 200;
    constexpr int PARAMETERS_PAIRS_COUNT = 5;
    static_assert(PARAMETERS_PAIRS_COUNT > 0);
    static_assert(N > PARAMETERS_PAIRS_COUNT);
    static_assert((N % PARAMETERS_PAIRS_COUNT) == 0);
    /* input    add-sub x N      output
        O---------->O------------->O
        ...        ...            /\
        L---------->O-------------_|
    */

    const std::vector<float> inputValues{9.1, -3.7, 22.2};
    this->prepareRequest(inputValues);

    const std::array<float, PARAMETERS_PAIRS_COUNT> addValues{4.5, 0.2, -0.6, 0.4, -2.5};
    const std::array<float, PARAMETERS_PAIRS_COUNT> subValues{8.5, -3.2, 10.0, -0.5, 2.4};

    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineInputName,
        InferenceEngine::Precision::FP32,
        shape_t{1, 3},
        InferenceEngine::Layout::NC);
    const tensor_map_t inputsInfo{{pipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    tensor_map_t outputsInfo;
    for (size_t i = 0; i < N; ++i) {
        const std::string outputName = pipelineOutputName + std::to_string(i);
        outputsInfo.emplace(outputName,
            std::make_shared<ovms::TensorInfo>(outputName,
                InferenceEngine::Precision::FP32,
                shape_t{1, 3},
                InferenceEngine::Layout::NC));
    }
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);

    Pipeline pipeline(*input_node, *output_node);
    std::unique_ptr<CustomNode> custom_nodes[N];
    for (int i = 0; i < N; i++) {
        custom_nodes[i] = std::make_unique<CustomNode>(customNodeName + std::to_string(i), library,
            parameters_t{
                {"add_value", std::to_string(addValues[i % PARAMETERS_PAIRS_COUNT])},
                {"sub_value", std::to_string(subValues[i % PARAMETERS_PAIRS_COUNT])}});
        pipeline.connect(*input_node, *(custom_nodes[i]),
            {{pipelineInputName, customNodeInputName}});
        pipeline.connect(*(custom_nodes[i]), *output_node,
            {{customNodeOutputName, pipelineOutputName + std::to_string(i)}});
        pipeline.push(std::move(custom_nodes[i]));
    }
    pipeline.push(std::move(input_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), N);

    for (int i = 0; i < N; i++) {
        this->checkResponse<float>(
            pipelineOutputName + std::to_string(i),
            inputValues,
            [i, addValues, subValues](float value) -> float {
                value += addValues[i % PARAMETERS_PAIRS_COUNT];
                value -= subValues[i % PARAMETERS_PAIRS_COUNT];
                return value;
            });
    }
}

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, CustomAndDLNodes) {
    // input  add-sub1 dummy  add-sub2 output
    //  O------->O------O--------O------>O
    ConstructorEnabledModelManager modelManager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    modelManager.reloadModelWithVersions(config);

    const std::vector<float> inputValues{
        4, 1.5, -5, -2.5, 9.3, 0.3, -0.15, 7.4, 5.2, -2.4};
    this->prepareRequest(inputValues);

    const float addValues[] = {-0.85, 30.2};
    const float subValues[] = {1.35, -28.5};

    const tensor_map_t inputsInfo{{pipelineInputName, dagDummyModelInputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    const tensor_map_t outputsInfo{{pipelineOutputName, dagDummyModelOutputTensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo);
    auto model_node = std::make_unique<DLNode>(
        "dummy_node",
        "dummy",
        std::nullopt,
        modelManager);
    std::unique_ptr<CustomNode> custom_node[] = {
        std::make_unique<CustomNode>(customNodeName + "_0", library,
            parameters_t{
                {"add_value", std::to_string(addValues[0])},
                {"sub_value", std::to_string(subValues[0])}}),
        std::make_unique<CustomNode>(customNodeName + "_1", library,
            parameters_t{
                {"add_value", std::to_string(addValues[1])},
                {"sub_value", std::to_string(subValues[1])}})};

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *(custom_node[0]), {{pipelineInputName, customNodeInputName}});
    pipeline.connect(*(custom_node[0]), *model_node, {{customNodeOutputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *(custom_node[1]), {{DUMMY_MODEL_OUTPUT_NAME, customNodeInputName}});
    pipeline.connect(*(custom_node[1]), *output_node, {{customNodeOutputName, pipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(custom_node[0]));
    pipeline.push(std::move(custom_node[1]));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), 1);

    this->checkResponse<float>(inputValues, [addValues, subValues](float value) -> float {
        return value + DUMMY_ADDITION_VALUE + addValues[0] + addValues[1] - subValues[0] - subValues[1];
    });
}

struct LibraryFailInExecute {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int) {
        return 1;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeExecution) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryFailInExecute>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_EXECUTION_FAILED);
}

struct LibraryCorruptedOutputHandle {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = nullptr;
        *outputsNum = 5;
        return 0;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputsCorruptedHandle) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryCorruptedOutputHandle>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED);
}

struct LibraryCorruptedOutputsNumber {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(5 * sizeof(struct CustomNodeTensor));
        *outputsNum = 0;
        return 0;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputsCorruptedNumberOfOutputs) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryCorruptedOutputsNumber>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT);
}

struct LibraryMissingOutput {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor));
        *outputsNum = 1;
        (*handle)->name = "random_not_connected_output";
        (*handle)->precision = CustomNodeTensorPrecision::FP32;
        (*handle)->dims = (uint64_t*)malloc(sizeof(uint64_t));
        (*handle)->dims[0] = 1;
        (*handle)->dimsCount = 1;
        (*handle)->data = (uint8_t*)malloc(sizeof(float) * sizeof(uint8_t));
        (*handle)->dataBytes = sizeof(float);
        return 0;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeMissingOutput) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryMissingOutput>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_MISSING_OUTPUT);
}

struct LibraryIncorrectOutputPrecision {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor));
        *outputsNum = 1;
        (*handle)->name = "output_numbers";
        (*handle)->precision = CustomNodeTensorPrecision::UNSPECIFIED;
        (*handle)->dims = (uint64_t*)malloc(sizeof(uint64_t));
        (*handle)->dimsCount = 1;
        (*handle)->data = (uint8_t*)malloc(sizeof(uint8_t));
        (*handle)->dataBytes = 1;
        return 0;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputInvalidPrecision) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryIncorrectOutputPrecision>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_INVALID_PRECISION);
}

struct LibraryIncorrectOutputShape {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor));
        *outputsNum = 1;
        (*handle)->name = "output_numbers";
        (*handle)->precision = CustomNodeTensorPrecision::FP32;
        (*handle)->dims = nullptr;
        (*handle)->dimsCount = 0;
        (*handle)->data = (uint8_t*)malloc(sizeof(uint8_t));
        (*handle)->dataBytes = 1;
        return 0;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputInvalidShape) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryIncorrectOutputShape>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_INVALID_SHAPE);
}

struct LibraryIncorrectOutputContentSize {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor));
        *outputsNum = 1;
        (*handle)->name = "output_numbers";
        (*handle)->precision = CustomNodeTensorPrecision::FP32;
        (*handle)->dims = (uint64_t*)malloc(sizeof(uint64_t));
        (*handle)->dimsCount = 1;
        (*handle)->data = nullptr;
        (*handle)->dataBytes = 0;
        return 0;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputInvalidContentSize) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryIncorrectOutputContentSize>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_INVALID_CONTENT_SIZE);
}

class EnsembleFlowCustomNodeFactoryCreateThenExecuteTest : public EnsembleFlowCustomNodePipelineExecutionTest {};

TEST_F(EnsembleFlowCustomNodeFactoryCreateThenExecuteTest, SimplePipelineFactoryCreationWithCustomNode) {
    // Nodes
    // request   custom    response
    //  O--------->O---------->O
    //          add-sub
    ConstructorEnabledModelManager manager;
    PipelineFactory factory;

    const std::vector<float> inputValues{7.8, -2.4, 1.9, 8.7, -2.4, 3.5};
    this->prepareRequest(inputValues);

    const float addValue = 0.9;
    const float subValue = 7.3;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{customNodeOutputName, customNodeOutputName}},
            std::nullopt, {}, library, parameters_t{{"add_value", std::to_string(addValue)}, {"sub_value", std::to_string(subValue)}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    // request (pipelineInputName) O--------->O custom node (customNodeInputName)
    connections["custom_node"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, customNodeInputName}}}};

    // custom node (customNodeOutputName) O--------->O response (pipelineOutputName)
    connections[EXIT_NODE_NAME] = {
        {"custom_node", {{customNodeOutputName, pipelineOutputName}}}};

    std::unique_ptr<Pipeline> pipeline;
    ASSERT_EQ(factory.createDefinition("my_new_pipeline", info, connections, manager), StatusCode::OK);
    ASSERT_EQ(factory.create(pipeline, "my_new_pipeline", &request, &response, manager), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    this->checkResponse<float>(inputValues, [addValue, subValue](float value) -> float {
        return value + addValue - subValue;
    });
}

TEST_F(EnsembleFlowCustomNodeFactoryCreateThenExecuteTest, ParallelPipelineFactoryUsageWithCustomNode) {
    //                 Nodes
    //              custom_node_N
    //         v-------->O----------v
    //  request O--------->O---------->O response     x   PARALLEL_SIMULATED_REQUEST_COUNT
    //         ^-------->O----------^
    //                add-sub
    ConstructorEnabledModelManager manager;
    PipelineFactory factory;

    const int PARALLEL_CUSTOM_NODES = 3;
    const int PARALLEL_SIMULATED_REQUEST_COUNT = 30;

    const std::vector<float> inputValues{7.8, -2.4, 1.9, 8.7, -2.4, 3.5};
    std::array<PredictRequest, PARALLEL_SIMULATED_REQUEST_COUNT> requests{};

    for (int i = 0; i < PARALLEL_SIMULATED_REQUEST_COUNT; i++) {
        this->prepareRequest(requests[i], inputValues);
    }

    const std::array<float, PARALLEL_CUSTOM_NODES> addValues{-1.5, 1.4, -0.1};
    const std::array<float, PARALLEL_CUSTOM_NODES> subValues{4.9, -1.9, -0.9};

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    for (int i = 0; i < PARALLEL_CUSTOM_NODES; i++) {
        info.emplace_back(std::move(NodeInfo(
            NodeKind::CUSTOM,
            "custom_node_" + std::to_string(i),
            "", std::nullopt,
            {{customNodeOutputName, customNodeOutputName}},
            std::nullopt, {},
            library, parameters_t{{"add_value", std::to_string(addValues[i])}, {"sub_value", std::to_string(subValues[i])}})));
    }

    pipeline_connections_t connections;

    for (int i = 0; i < PARALLEL_CUSTOM_NODES; i++) {
        // request (pipelineInputName) O--------->O custom_node_N (customNodeInputName)
        connections["custom_node_" + std::to_string(i)] = {
            {ENTRY_NODE_NAME, {{pipelineInputName, customNodeInputName}}}};
    }

    auto& responseConnections = connections[EXIT_NODE_NAME];
    for (int i = 0; i < PARALLEL_CUSTOM_NODES; i++) {
        responseConnections["custom_node_" + std::to_string(i)] =
            {{customNodeOutputName, "output_" + std::to_string(i)}};
    }

    std::unique_ptr<Pipeline> pipeline;
    ASSERT_EQ(factory.createDefinition("my_new_pipeline", info, connections, manager), StatusCode::OK);
    ASSERT_EQ(factory.create(pipeline, "my_new_pipeline", &requests[0], &response, manager), StatusCode::OK);

    auto run = [this, &requests, &manager, &factory, &inputValues, addValues, subValues, PARALLEL_CUSTOM_NODES](int i) {
        std::unique_ptr<Pipeline> pipeline;
        PredictResponse response_local;

        ASSERT_EQ(factory.create(pipeline, "my_new_pipeline", &requests[i], &response_local, manager), StatusCode::OK);
        ASSERT_EQ(pipeline->execute(), StatusCode::OK);

        for (int n = 0; n < PARALLEL_CUSTOM_NODES; n++) {
            this->checkResponse<float>("output_" + std::to_string(n), response_local, inputValues, [addValues, subValues, n](float value) -> float {
                return value + addValues[n] - subValues[n];
            });
        }
    };

    std::vector<std::promise<void>> promises(PARALLEL_SIMULATED_REQUEST_COUNT);
    std::vector<std::thread> threads;

    for (int n = 0; n < PARALLEL_SIMULATED_REQUEST_COUNT; n++) {
        threads.emplace_back(std::thread([&promises, n, &run]() {
            promises[n].get_future().get();
            run(n);
        }));
    }

    // Sleep to allow all threads to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    for (auto& promise : promises) {
        promise.set_value();
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

static const char* pipelineCustomNodeConfig = R"(
{
    "model_config_list": [],
    "custom_node_library_config_list": [
        {
            "name": "lib_add_sub",
            "base_path": "/ovms/bazel-bin/src/lib_node_add_sub.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_sub",
                    "params": {
                        "add_value": "3.2",
                        "sub_value": "2.7"
                    },
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

class EnsembleFlowCustomNodeLoadConfigThenExecuteTest : public EnsembleFlowCustomNodePipelineExecutionTest {
protected:
    void SetUp() override {
        TestWithTempDir::SetUp();
        configJsonFilePath = directoryPath + "/ovms_config_file.json";
    }

    void loadCorrectConfiguration() {
        this->loadConfiguration(pipelineCustomNodeConfig);
    }

    void loadConfiguration(const char* configContent, Status expectedStatus = StatusCode::OK) {
        createConfigFileWithContent(configContent, configJsonFilePath);
        ASSERT_EQ(manager.loadConfig(configJsonFilePath), expectedStatus);
    }

    void checkResponseForCorrectConfiguration() {
        this->checkResponse<float>(inputValues, [](float value) -> float {
            return value + 3.2 - 2.7;
        });
    }

    std::string configJsonFilePath;
    const std::string pipelineName = "my_pipeline";
    ConstructorEnabledModelManager manager;
    const std::vector<float> inputValues{2.4, 9.3, -7.1};
};

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, AddSubCustomNode) {
    std::unique_ptr<Pipeline> pipeline;
    this->prepareRequest(inputValues);
    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
}

static const char* pipelineCustomNodeReferenceMissingLibraryConfig = R"(
{
    "model_config_list": [],
    "custom_node_library_config_list": [
        {
            "name": "lib_add_sub",
            "base_path": "/ovms/bazel-bin/src/lib_node_add_sub.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "non_existing_library",
                    "params": {
                        "add_value": "3.2",
                        "sub_value": "2.7"
                    },
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, ReferenceMissingLibraryThenCorrect) {
    std::unique_ptr<Pipeline> pipeline;
    this->prepareRequest(inputValues);

    // Loading correct configuration is required for test to pass.
    // This is due to fact that when OVMS loads pipeline definition for the first time and fails, its status is RETIRED.
    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
    response.Clear();

    this->loadConfiguration(pipelineCustomNodeReferenceMissingLibraryConfig, StatusCode::PIPELINE_DEFINITION_INVALID_NODE_LIBRARY);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET);
    response.Clear();

    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
}

static const char* pipelineCustomNodeReferenceLibraryWithExecutionErrorMissingParamsLibraryConfig = R"(
{
    "model_config_list": [],
    "custom_node_library_config_list": [
        {
            "name": "lib_add_sub_new",
            "base_path": "/ovms/bazel-bin/src/lib_node_add_sub.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_sub_new",
                    "params": {
                    },
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, ReferenceLibraryWithExecutionErrorThenCorrect) {
    std::unique_ptr<Pipeline> pipeline;
    this->prepareRequest(inputValues);

    // Loading correct configuration is required for test to pass.
    // This is due to fact that when OVMS loads pipeline definition for the first time and fails, its status is RETIRED.
    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
    response.Clear();

    this->loadConfiguration(pipelineCustomNodeReferenceLibraryWithExecutionErrorMissingParamsLibraryConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_EXECUTION_FAILED);
    response.Clear();

    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
}

static const char* pipelineCustomNodeMissingParametersConfig = R"(
{
    "model_config_list": [],
    "custom_node_library_config_list": [
        {
            "name": "lib_add_sub",
            "base_path": "/ovms/bazel-bin/src/lib_node_add_sub.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_sub",
                    "params": {
                        "random_parameter": "abcd"
                    },
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, MissingRequiredNodeParametersThenCorrect) {
    std::unique_ptr<Pipeline> pipeline;
    this->prepareRequest(inputValues);

    // Loading correct configuration is required for test to pass.
    // This is due to fact that when OVMS loads pipeline definition for the first time and fails, its status is RETIRED.
    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
    response.Clear();

    this->loadConfiguration(pipelineCustomNodeMissingParametersConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_EXECUTION_FAILED);
    response.Clear();

    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
}

static const char* pipelineCustomNodeLibraryNotEscapedPathConfig = R"(
{
    "model_config_list": [],
    "custom_node_library_config_list": [
        {
            "name": "lib_add_sub_new",
            "base_path": "/ovms/bazel-bin/src/../src/lib_node_add_sub.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_sub_new",
                    "params": {
                        "add_value": "3.2",
                        "sub_value": "2.7"
                    },
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, ReferenceLibraryWithRestrictedBasePathThenCorrect) {
    std::unique_ptr<Pipeline> pipeline;
    this->prepareRequest(inputValues);

    // Loading correct configuration is required for test to pass.
    // This is due to fact that when OVMS loads pipeline definition for the first time and fails, its status is RETIRED.
    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
    response.Clear();

    this->loadConfiguration(pipelineCustomNodeLibraryNotEscapedPathConfig, StatusCode::PIPELINE_DEFINITION_INVALID_NODE_LIBRARY);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET);
    response.Clear();

    this->loadCorrectConfiguration();
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    this->checkResponseForCorrectConfiguration();
}

static const char* pipelineCustomNodeDifferentOperationsConfig = R"(
{
    "model_config_list": [],
    "custom_node_library_config_list": [
        {
            "name": "lib_perform_different_operations",
            "base_path": "/ovms/bazel-bin/src/lib_node_perform_different_operations.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_perform_different_operations",
                    "type": "custom",
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}},
                        {"op_factors": {"node_name": "request",
                                           "data_item": "pipeline_factors"}}
                    ],
                    "outputs": [
                        {"data_item": "different_ops_results",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

class EnsembleFlowCustomNodeAndDemultiplexerLoadConfigThenExecuteTest : public EnsembleFlowCustomNodeLoadConfigThenExecuteTest {
protected:
    void SetUp() override {
        EnsembleFlowCustomNodeLoadConfigThenExecuteTest::SetUp();
        configJsonFilePath = directoryPath + "/ovms_config_file.json";
    }
    const std::string differentOpsInputName = "pipeline_input";
    const std::string differentOpsFactorsName = "pipeline_factors";
};

enum OPS {
    ADD,
    SUB,
    MULTIPLY,
    DIVIDE
};

static void prepareDifferentOpsExpectedOutput(std::vector<float>& expectedOutput, const std::vector<float>& input, const std::vector<float>& factors) {
    for (size_t j = 0; j < 4; ++j) {  // iterate over ops
        for (size_t i = 0; i < DUMMY_MODEL_OUTPUT_SIZE; ++i) {
            size_t index = DUMMY_MODEL_OUTPUT_SIZE * j + i;
            switch (j) {
            case ADD:
                expectedOutput[index] = input[i] + factors[j];
                break;
            case SUB:
                expectedOutput[index] = input[i] - factors[j];
                break;
            case MULTIPLY:
                expectedOutput[index] = input[i] * factors[j];
                break;
            case DIVIDE:
                expectedOutput[index] = input[i] / factors[j];
                break;
            }
        }
    }
}

enum class Method {
    MAXIMUM_MAXIMUM,
    MAXIMUM_MINIMUM,
    MAXIMUM_AVERAGE,
};

std::vector<float> prepareGatherHighestExpectedOutput(std::vector<float> input, Method option) {
    std::vector<float> expectedOutput(DUMMY_MODEL_OUTPUT_SIZE);
    size_t tensorsCount = input.size() / DUMMY_MODEL_OUTPUT_SIZE;
    // perform operations
    std::vector<float> minimums(tensorsCount, std::numeric_limits<int>::max());
    std::vector<float> maximums(tensorsCount, std::numeric_limits<int>::lowest());
    std::vector<float> averages(tensorsCount, 0);
    for (size_t opId = 0; opId < tensorsCount; ++opId) {  // iterate over ops
        for (size_t i = 0; i < DUMMY_MODEL_OUTPUT_SIZE; ++i) {
            size_t index = DUMMY_MODEL_OUTPUT_SIZE * opId + i;
            switch (option) {
            case Method::MAXIMUM_MAXIMUM:
                maximums[opId] = std::max(maximums[opId], input[index]);
                break;
            case Method::MAXIMUM_MINIMUM:
                minimums[opId] = std::min(maximums[opId], input[index]);
                break;
            case Method::MAXIMUM_AVERAGE:
                averages[opId] += input[index];
                break;
            default:
                throw std::logic_error("");
                break;
            }
        }
        averages[opId] /= DUMMY_MODEL_OUTPUT_SIZE;
    }
    // choose tensor
    size_t whichTensor = 42;
    const std::vector<float>* fromWhichContainerToChoose = &maximums;
    switch (option) {
    case Method::MAXIMUM_MAXIMUM:
        fromWhichContainerToChoose = &maximums;
        break;
    case Method::MAXIMUM_MINIMUM:
        fromWhichContainerToChoose = &minimums;
        break;
    case Method::MAXIMUM_AVERAGE:
        fromWhichContainerToChoose = &averages;
        break;
    default:
        throw std::logic_error("");
    }
    whichTensor = std::distance(fromWhichContainerToChoose->begin(),
        std::max_element(fromWhichContainerToChoose->begin(),
            fromWhichContainerToChoose->end()));
    // copy tensor
    std::copy(input.begin() + DUMMY_MODEL_OUTPUT_SIZE * whichTensor,
        input.begin() + DUMMY_MODEL_OUTPUT_SIZE * (whichTensor + 1),
        expectedOutput.begin());
    return expectedOutput;
}

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerLoadConfigThenExecuteTest, JustDifferentOpsCustomNode) {
    std::unique_ptr<Pipeline> pipeline;
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> factors{1, 3, 2, 2};  // add/sub/multiply/divide
    this->prepareRequest(request, input, differentOpsInputName);
    this->prepareRequest(request, factors, differentOpsFactorsName);
    this->loadConfiguration(pipelineCustomNodeDifferentOperationsConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    std::vector<float> expectedOutput(4 * DUMMY_MODEL_OUTPUT_SIZE);
    prepareDifferentOpsExpectedOutput(expectedOutput, input, factors);
    this->checkResponse("pipeline_output", response, expectedOutput, {4, 1, 10});

    auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(pipelineName);
    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());

    const auto& input_A = inputs.at(differentOpsInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 10}));
    const auto& input_B = inputs.at(differentOpsFactorsName);
    EXPECT_EQ(input_B->getEffectiveShape(), shape_t({1, 4}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({4, 1, 10}));
}

static const char* pipelineCustomNodeDifferentOperationsThenDummyConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_perform_different_operations",
            "base_path": "/ovms/bazel-bin/src/lib_node_perform_different_operations.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_perform_different_operations",
                    "type": "custom",
                    "demultiply_count": 4,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}},
                        {"op_factors": {"node_name": "request",
                                           "data_item": "pipeline_factors"}}
                    ],
                    "outputs": [
                        {"data_item": "different_ops_results",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerLoadConfigThenExecuteTest, DifferentOpsCustomNodeThenDummy) {
    std::unique_ptr<Pipeline> pipeline;
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> factors{1, 3, 2, 2};  // add/sub/multiply/divide
    this->prepareRequest(request, input, differentOpsInputName);
    this->prepareRequest(request, factors, differentOpsFactorsName);
    this->loadConfiguration(pipelineCustomNodeDifferentOperationsThenDummyConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);
    std::vector<float> expectedOutput(4 * DUMMY_MODEL_OUTPUT_SIZE);
    prepareDifferentOpsExpectedOutput(expectedOutput, input, factors);
    std::transform(expectedOutput.begin(), expectedOutput.end(), expectedOutput.begin(),
        [](float f) -> float { return f + 1; });
    this->checkResponse("pipeline_output", response, expectedOutput, {4, 1, 10});

    auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(pipelineName);
    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());

    const auto& input_A = inputs.at(pipelineInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 10}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({4, 1, 10}));
}

static const char* pipelineCustomNodeDifferentOperations2OutputsConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_perform_different_operations",
            "base_path": "/ovms/bazel-bin/src/lib_node_perform_different_operations.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_perform_different_operations",
                    "type": "custom",
                    "demultiply_count": 4,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}},
                        {"op_factors": {"node_name": "request",
                                           "data_item": "pipeline_factors"}}
                    ],
                    "outputs": [
                        {"data_item": "different_ops_results",
                         "alias": "custom_node_output"},
                        {"data_item": "different_ops_factors",
                         "alias": "custom_node_factors"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                },
                {"pipeline_factors": {"node_name": "custom_node",
                                      "data_item": "custom_node_factors"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerLoadConfigThenExecuteTest, DifferentOpsCustomNode2OutputsMetadataCheck) {
    std::unique_ptr<Pipeline> pipeline;
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> factors{1, 3, 2, 2};  // add/sub/multiply/divide
    this->prepareRequest(request, input, differentOpsInputName);
    this->prepareRequest(request, factors, differentOpsFactorsName);
    this->loadConfiguration(pipelineCustomNodeDifferentOperations2OutputsConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    std::vector<float> expectedOutput(4 * DUMMY_MODEL_OUTPUT_SIZE);
    prepareDifferentOpsExpectedOutput(expectedOutput, input, factors);
    this->checkResponse("pipeline_output", response, expectedOutput, {4, 1, 10});

    auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(pipelineName);
    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());

    const auto& input_A = inputs.at(differentOpsInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 10}));
    const auto& input_B = inputs.at(differentOpsFactorsName);
    EXPECT_EQ(input_B->getEffectiveShape(), shape_t({1, 4}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({4, 1, 10}));
    const auto& outputFactors = outputs.at("pipeline_factors");
    EXPECT_EQ(outputFactors->getEffectiveShape(), shape_t({4, 1, 4}));
}

static const char* pipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_perform_different_operations",
            "base_path": "/ovms/bazel-bin/src/lib_node_perform_different_operations.so"
        },
        {
            "name": "lib_choose_maximum",
            "base_path": "/ovms/bazel-bin/src/lib_node_choose_maximum.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_perform_different_operations",
                    "type": "custom",
                    "demultiply_count": 4,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}},
                        {"op_factors": {"node_name": "request",
                                           "data_item": "pipeline_factors"}}
                    ],
                    "outputs": [
                        {"data_item": "different_ops_results",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                },
                {
                    "name": "choose_max",
                    "library_name": "lib_choose_maximum",
                    "type": "custom",
                    "gather_from_node": "custom_node",
                    "params": {
                        "selection_criteria": "MAXIMUM_MINIMUM"
                    },
                    "inputs": [
                        {"input_tensors": {"node_name": "dummyNode",
                                           "data_item": "dummy_output"}}
                    ],
                    "outputs": [
                        {"data_item": "maximum_tensor",
                         "alias": "maximum_tensor_alias"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "choose_max",
                                     "data_item": "maximum_tensor_alias"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerLoadConfigThenExecuteTest, DifferentOpsCustomNodeThenDummyThenChooseMaximum) {
    std::unique_ptr<Pipeline> pipeline;
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> factors{1, 3, 2, 2};  // add/sub/multiply/divide
    this->prepareRequest(request, input, differentOpsInputName);
    this->prepareRequest(request, factors, differentOpsFactorsName);
    this->loadConfiguration(pipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    std::vector<float> expectedOutput(4 * DUMMY_MODEL_OUTPUT_SIZE);
    prepareDifferentOpsExpectedOutput(expectedOutput, input, factors);
    std::transform(expectedOutput.begin(), expectedOutput.end(), expectedOutput.begin(),
        [](float f) -> float { return f + 1; });
    std::vector<float> expectedResult = prepareGatherHighestExpectedOutput(expectedOutput, Method::MAXIMUM_MINIMUM);
    this->checkResponse("pipeline_output", response, expectedResult, {1, 10});
}

static const char* pipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumThenDummyConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_perform_different_operations",
            "base_path": "/ovms/bazel-bin/src/lib_node_perform_different_operations.so"
        },
        {
            "name": "lib_choose_maximum",
            "base_path": "/ovms/bazel-bin/src/lib_node_choose_maximum.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_perform_different_operations",
                    "type": "custom",
                    "demultiply_count": 4,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}},
                        {"op_factors": {"node_name": "request",
                                           "data_item": "pipeline_factors"}}
                    ],
                    "outputs": [
                        {"data_item": "different_ops_results",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                },
                {
                    "name": "choose_max",
                    "library_name": "lib_choose_maximum",
                    "type": "custom",
                    "gather_from_node": "custom_node",
                    "params": {
                        "selection_criteria": "MAXIMUM_MAXIMUM"
                    },
                    "inputs": [
                        {"input_tensors": {"node_name": "dummyNode",
                                           "data_item": "dummy_output"}}
                    ],
                    "outputs": [
                        {"data_item": "maximum_tensor",
                         "alias": "maximum_tensor_alias"}
                    ]
                },
                {
                    "name": "dummyNode2",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "choose_max",
                               "data_item": "maximum_tensor_alias"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "dummyNode2",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerLoadConfigThenExecuteTest, DifferentOpsCustomNodeThenDummyThenChooseMaximumThenDummyAgain) {
    std::unique_ptr<Pipeline> pipeline;
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> factors{1, 3, 2, 2};  // add/sub/multiply/divide
    this->prepareRequest(request, input, differentOpsInputName);
    this->prepareRequest(request, factors, differentOpsFactorsName);
    this->loadConfiguration(pipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumThenDummyConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    std::vector<float> expectedOutput(4 * DUMMY_MODEL_OUTPUT_SIZE);
    prepareDifferentOpsExpectedOutput(expectedOutput, input, factors);
    std::transform(expectedOutput.begin(), expectedOutput.end(), expectedOutput.begin(),
        [](float f) -> float { return f + 1; });
    std::vector<float> expectedResult = prepareGatherHighestExpectedOutput(expectedOutput, Method::MAXIMUM_MAXIMUM);
    std::transform(expectedResult.begin(), expectedResult.end(), expectedResult.begin(),
        [](float f) -> float { return f + 1; });
    this->checkResponse("pipeline_output", response, expectedResult, {1, 10});
}

static const char* demultiplyThenDummyThenChooseMaximumConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_choose_maximum",
            "base_path": "/ovms/bazel-bin/src/lib_node_choose_maximum.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input"],
            "demultiply_count": 0,
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                },
                {
                    "name": "choose_max",
                    "library_name": "lib_choose_maximum",
                    "type": "custom",
                    "gather_from_node": "request",
                    "params": {
                        "selection_criteria": "MAXIMUM_MAXIMUM"
                    },
                    "inputs": [
                        {"input_tensors": {"node_name": "dummyNode",
                                           "data_item": "dummy_output"}}
                    ],
                    "outputs": [
                        {"data_item": "maximum_tensor",
                         "alias": "maximum_tensor_alias"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "choose_max",
                                     "data_item": "maximum_tensor_alias"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerLoadConfigThenExecuteTest, DemultiplyThenDummyThenChooseMaximum) {
    std::unique_ptr<Pipeline> pipeline;
    std::vector<float> input(4 * DUMMY_MODEL_OUTPUT_SIZE);
    std::fill(input.begin(), input.end(), 1.0);

    uint iterations = -1;
    uint number = 0;
    std::transform(input.begin(), input.end(), input.begin(),
        [&iterations, &number](float f) -> float {
            iterations++;
            number = iterations/10;
            return f + number; });

    this->prepareRequest(request, input, differentOpsInputName, {4, 1, 10});
    this->loadConfiguration(demultiplyThenDummyThenChooseMaximumConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    auto status = pipeline->execute();
    ASSERT_EQ(status, StatusCode::OK) << status.string();

    std::vector<float> expectedOutput{5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    this->checkResponse("pipeline_output", response, expectedOutput, {1, 10});
}

struct LibraryParamControlledMetadata {
    static bool startsWith(const char* str, const char* prefix) {
        // Ensure null terminated
        const int MAX = 300;
        const char* end = str;
        for (; *end != '\0'; ++end) {
            if ((end - str) > MAX) {
                EXPECT_TRUE(false);
            }
        }
        const char* end2 = prefix;
        for (; *end2 != '\0'; ++end2) {
            if ((end2 - str) > MAX) {
                EXPECT_TRUE(false);
            }
        }
        size_t strLen = std::strlen(str);
        size_t prefixLen = std::strlen(prefix);
        return strLen < prefixLen ? false : std::memcmp(str, prefix, prefixLen) == 0;
    }
    // Extract TensorInfo out of string in format: "1,3,500,500;FP32"
    static CustomNodeTensorInfo extractMetadata(const char* key, const char* value) {
        std::string keyStr = key;
        std::string valueStr = value;
        auto tokens = tokenize(valueStr, ';');
        EXPECT_EQ(tokens.size(), 2);
        std::string shapeStr = tokens[0];
        std::string precisionStr = tokens[1];
        tokens = tokenize(shapeStr, ',');
        EXPECT_GE(tokens.size(), 1);
        shape_t shape;
        std::transform(tokens.begin(), tokens.end(), std::back_inserter(shape),
            [](const std::string& str) { return std::stoull(str); });
        CustomNodeTensorPrecision precision = toCustomNodeTensorPrecision(InferenceEngine::Precision::FromStr(precisionStr));
        CustomNodeTensorInfo info;
        info.name = key;
        info.dimsCount = shape.size();
        info.dims = (uint64_t*)malloc(info.dimsCount * sizeof(uint64_t));
        std::memcpy(info.dims, shape.data(), info.dimsCount * sizeof(uint64_t));
        info.precision = precision;
        return info;
    }
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int) {
        return 1;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount) {
        int inputs = 0;
        for (int i = 0; i < paramsCount; i++) {
            if (startsWith(params[i].key, "in_")) {
                inputs++;
            }
        }
        if (inputs == 0) {
            return 1;
        }
        *infoCount = inputs;
        *info = (struct CustomNodeTensorInfo*)malloc(inputs * sizeof(CustomNodeTensorInfo));
        int preparedInputsMetaCount = 0;
        for (int i = 0; i < paramsCount; i++) {
            if (startsWith(params[i].key, "in_")) {
                (*info)[preparedInputsMetaCount] = extractMetadata(params[i].key, params[i].value);
                preparedInputsMetaCount++;
            }
        }
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount) {
        int outputs = 0;
        for (int i = 0; i < paramsCount; i++) {
            if (startsWith(params[i].key, "out_")) {
                outputs++;
            }
        }
        if (outputs == 0) {
            return 1;
        }
        *infoCount = outputs;
        *info = (struct CustomNodeTensorInfo*)malloc(outputs * sizeof(CustomNodeTensorInfo));
        int preparedInputsMetaCount = 0;
        for (int i = 0; i < paramsCount; i++) {
            if (startsWith(params[i].key, "out_")) {
                (*info)[preparedInputsMetaCount] = extractMetadata(params[i].key, params[i].value);
                preparedInputsMetaCount++;
            }
        }
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

class EnsembleConfigurationValidationWithCustomNode : public ::testing::Test {
protected:
    void SetUp() override {
        mockedLibrary = createLibraryMock<LibraryParamControlledMetadata>();
        ASSERT_TRUE(mockedLibrary.isValid());
    }

    NodeLibrary mockedLibrary;

    const std::string customNodeInputName = "input_numbers";
    const std::string customNodeOutputName = "output_numbers";
    static constexpr const char* pipelineInputName = "pipeline_input";
    const std::string pipelineOutputName = "pipeline_output";
};

TEST_F(EnsembleConfigurationValidationWithCustomNode, SuccessfulConfiguration) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "1,30,7;I32"},
                {"out_OutputNumbers_2", "1,8;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,30,7;I32"},
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, SuccessfulConfigurationWithDynamicShapeInInput) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "1,30,7;I32"},
                {"out_OutputNumbers_2", "1,8;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,0,7;I32"},
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, SuccessfulConfigurationWithDynamicShapeInOutput) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "1,0,7;I32"},
                {"out_OutputNumbers_2", "1,8;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,30,7;I32"},
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, ShapesNotMatchBetweenDLModelAndCustomNode) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::DL, "dummy_node_1", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node_2", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10,7;FP32"},  // 1,10 is correcct
                {"in_InputNumbers_2", "1,10;FP32"},
                {"out_OutputNumbers", "1,2000;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_node_2"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["custom_node"] = {
        {"dummy_node_1", {{DUMMY_MODEL_OUTPUT_NAME, "in_InputNumbers_1"}}},
        {"dummy_node_2", {{DUMMY_MODEL_OUTPUT_NAME, "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, ShapesNotMatchBetweenCustomNodeAndDLNode) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10,7;I32"},
                {"out_OutputNumbers", "1,8;FP32"}  // 1,10 is correct
            }},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers_1"}}}};

    connections["dummy_node"] = {
        {"custom_node", {{"out", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, ShapesNotMatchBetweenCustomNodes) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers_1", "1,30,7;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,30,7;I32"},
                {"in_InputNumbers_2", "1,8,1;I32"},  // 1,8 is correct
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, PrecisionNotMatchBetweenDLModelAndCustomNode) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::DL, "dummy_node_1", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::DL, "dummy_node_2", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"in_InputNumbers_2", "1,10;I32"},  // FP32 is correct
                {"out_OutputNumbers", "1,2000;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["dummy_node_2"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["custom_node"] = {
        {"dummy_node_1", {{DUMMY_MODEL_OUTPUT_NAME, "in_InputNumbers_1"}}},
        {"dummy_node_2", {{DUMMY_MODEL_OUTPUT_NAME, "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_PRECISION);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, PrecisionNotMatchBetweenCustomNodeAndDLNode) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10,7;I32"},
                {"out_OutputNumbers", "1,10;I32"}  // FP32 is correct
            }},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers_1"}}}};

    connections["dummy_node"] = {
        {"custom_node", {{"out", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_PRECISION);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, PrecisionNotMatchBetweenCustomNodes) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers_1", "1,30,7;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,30,7;FP32"},  // I32 is correct
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_PRECISION);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, NotAllCustomNodeInputsAreConnected) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "1,30,7;I32"},
                {"out_OutputNumbers_2", "1,8;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,30,7;I32"},
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    // Missing connection {"1", "in_InputNumbers_1"}
    connections["custom_node_2"] = {
        {"custom_node_1", {{"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_NOT_ALL_INPUTS_CONNECTED);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, CustomNodeMissingOutput) {
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"out", "not_existing_output"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,30,7;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_1", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL_OUTPUT);
}

TEST_F(EnsembleConfigurationValidationWithCustomNode, InvalidSharedLibrary) {
    NodeLibrary invalidLibrary{};
    ASSERT_FALSE(invalidLibrary.isValid());
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, std::nullopt, {}, invalidLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "1,30,7;I32"},
                {"out_OutputNumbers_2", "1,8;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, invalidLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,30,7;I32"},
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_DEFINITION_INVALID_NODE_LIBRARY);
}

struct LibraryErrorsOnMetadataCall {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 1;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 1;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleConfigurationValidationWithCustomNode, SharedLibraryErrorsOnMetadataCall) {
    NodeLibrary libraryFailingOnMetadataCall = createLibraryMock<LibraryErrorsOnMetadataCall>();
    ASSERT_TRUE(libraryFailingOnMetadataCall.isValid());
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, std::nullopt, {}, libraryFailingOnMetadataCall,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "1,30,7;I32"},
                {"out_OutputNumbers_2", "1,8;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, libraryFailingOnMetadataCall,
            parameters_t{
                {"in_InputNumbers_1", "1,30,7;I32"},
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::NODE_LIBRARY_METADATA_FAILED);
}

class EnsembleConfigurationValidationWithDemultiplexer : public EnsembleConfigurationValidationWithCustomNode {};

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationSingleDemultiplexer) {
    const size_t demultiplyCount = 7;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "7,1,700;I32"},
                {"out_OutputNumbers_2", "7,1,8;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,700;I32"},
                {"in_InputNumbers_2", "1,8;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1"}},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationSingleDynamicDemultiplexerFirst) {
    std::optional<uint32_t> demultiplyCount = 0;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers_1", "0,1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers", "1,10;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1"}},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};
    connections["custom_node_2"] = {{"custom_node_1", {{"1", "in_InputNumbers_1"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationSingleDynamicDemultiplexerFixedLibraryFirstMetadataCheck) {
    std::optional<uint32_t> demultiplyCount = 0;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers_1", "12,1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers", "1,10;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1"}},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};
    connections["custom_node_2"] = {{"custom_node_1", {{"1", "in_InputNumbers_1"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);

    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());

    const auto& input_A = inputs.at(pipelineInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 10}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({12, 1, 10}));
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationFixedDemultiplexerDynamicLibraryFirstMetadataCheckShouldAlsoWarnInLog) {
    std::optional<uint32_t> demultiplyCount = 12;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers_1", "0,1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers", "1,10;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1"}},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};
    connections["custom_node_2"] = {{"custom_node_1", {{"1", "in_InputNumbers_1"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);

    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());

    const auto& input_A = inputs.at(pipelineInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 10}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({12, 1, 10}));
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationDynamicLibraryShapesMetadataCheckShouldAlsoWarnInLog) {
    std::optional<uint32_t> demultiplyCount = std::nullopt;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,0,0,0;FP32"},
                {"out_OutputNumbers_1", "0,1,0;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {}},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_1", {{"1", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);

    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());

    const auto& input_A = inputs.at(pipelineInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 0, 0, 0}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({0, 1, 0}));
}
TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationSingleDynamicDemultiplexerLast) {
    std::optional<uint32_t> demultiplyCount = 0;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers_1", "0, 1,10;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1"}},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_1", {{"1", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationSingleDynamicDemultiplexerAndDynamicGather) {
    std::optional<uint32_t> demultiplyCount = 0;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers_1", "0,1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_2", "1,10;FP32"},
                {"out_OutputNumbers_2", "1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"3", "out_OutputNumbers_3"}}, std::nullopt, {"custom_node_1"}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_3", "0,1,10;FP32"},
                {"out_OutputNumbers_3", "1,10;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers_1"}}}};
    connections["custom_node_2"] = {{"custom_node_1", {{"1", "in_InputNumbers_2"}}}};
    connections["custom_node_3"] = {{"custom_node_2", {{"2", "in_InputNumbers_3"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_3", {{"3", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationSingleFixedDemultiplexerAndDynamicGather) {
    std::optional<uint32_t> demultiplyCount = 12;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers_1", "12,1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_2", "1,10;FP32"},
                {"out_OutputNumbers_2", "1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"3", "out_OutputNumbers_3"}}, std::nullopt, {"custom_node_1"}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_3", "0,1,10;FP32"},
                {"out_OutputNumbers_3", "1,10;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers_1"}}}};
    connections["custom_node_2"] = {{"custom_node_1", {{"1", "in_InputNumbers_2"}}}};
    connections["custom_node_3"] = {{"custom_node_2", {{"2", "in_InputNumbers_3"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_3", {{"3", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationSingleDynamicDemultiplexerAndFixedGatherShouldWarnInLog) {
    std::optional<uint32_t> demultiplyCount = 0;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers_1", "0,1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"2", "out_OutputNumbers_2"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_2", "1,10;FP32"},
                {"out_OutputNumbers_2", "1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"3", "out_OutputNumbers_3"}}, std::nullopt, {"custom_node_1"}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_3", "12,1,10;FP32"},
                {"out_OutputNumbers_3", "1,10;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers_1"}}}};
    connections["custom_node_2"] = {{"custom_node_1", {{"1", "in_InputNumbers_2"}}}};
    connections["custom_node_3"] = {{"custom_node_2", {{"2", "in_InputNumbers_3"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_3", {{"3", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationSingleDynamicDemultiplexerFixedLibraryDynamicGatherMetadataCheck) {
    std::optional<uint32_t> demultiplyCount = 0;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers_1", "12,1,10;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers", "1,10;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1"}},
    };

    pipeline_connections_t connections;
    connections["custom_node_1"] = {{ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};
    connections["custom_node_2"] = {{"custom_node_1", {{"1", "in_InputNumbers_1"}}}};
    connections[EXIT_NODE_NAME] = {{"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);

    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());

    const auto& input_A = inputs.at(pipelineInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 10}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({12, 1, 10}));
}
TEST_F(EnsembleConfigurationValidationWithDemultiplexer, SuccessfulConfigurationMultipleDemultiplexers) {
    const size_t demultiplyCount1 = 11;
    const size_t demultiplyCount2 = 43;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount1, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "11,1,700;I32"},
                {"out_OutputNumbers_2", "11,1,8;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount2, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,700;I32"},
                {"in_InputNumbers_2", "1,8;FP32"},
                {"out_OutputNumbers", "43,1,2000;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,2000;FP32"},
                {"out_OutputNumbers", "1,5;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1", "custom_node_2"}},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections["custom_node_3"] = {
        {"custom_node_2", {{"out", "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_3", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, MultipleBatchInCustomNodeRestricted) {
    const size_t demultiplyCount = 9;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "3,3,10;FP32"},  // 1,3,10 is correct
                {"out_OutputNumbers_1", "9,1,700;I32"},
                {"out_OutputNumbers_2", "9,1,8;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,700;I32"},
                {"in_InputNumbers_2", "1,8;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    auto status = pipelineDefinition->validate(manager);
    ASSERT_EQ(status, StatusCode::PIPELINE_DEMULTIPLEXER_MULTIPLE_BATCH_SIZE) << status.string();
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, DemultiplexerNodeNotEnoughDimensionsToDemultiply) {
    const size_t demultiplyCount = 29;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, demultiplyCount},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers", "25,1,12;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["custom_node"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "in_InputNumbers_1"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_NOT_ENOUGH_SHAPE_DIMENSIONS_TO_DEMULTIPLY);
}

class DummyModelWithMockedMetadata : public ovms::ModelInstance {
    ovms::tensor_map_t mockedInputsInfo, mockedOutputsInfo;

public:
    DummyModelWithMockedMetadata(const ovms::tensor_map_t& inputsInfo, const ovms::tensor_map_t& outputsInfo) :
        ovms::ModelInstance("dummy", 1),
        mockedInputsInfo(inputsInfo),
        mockedOutputsInfo(outputsInfo) {}

    size_t getBatchSize() const override {
        return 1;
    }

    const ovms::ModelConfig& getModelConfig() const override {
        return DUMMY_MODEL_CONFIG;
    }

    const ovms::tensor_map_t& getInputsInfo() const override {
        return mockedInputsInfo;
    }

    const ovms::tensor_map_t& getOutputsInfo() const override {
        return mockedOutputsInfo;
    }
};

class ModelWithDummyModelWithMockedMetadata : public ovms::Model {
    std::shared_ptr<DummyModelWithMockedMetadata> modelInstance;

public:
    ModelWithDummyModelWithMockedMetadata(const std::string& name, std::shared_ptr<DummyModelWithMockedMetadata> modelInstance) :
        Model(name, false, nullptr),
        modelInstance(modelInstance) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t) override {
        return modelInstance;
    }
};

std::shared_ptr<ModelWithDummyModelWithMockedMetadata> dummyModelWithMockedMetadata;

class ModelManagerWithModelWithDummyModelWithMockedMetadata : public ovms::ModelManager {
    std::shared_ptr<DummyModelWithMockedMetadata> modelInstance;

public:
    ModelManagerWithModelWithDummyModelWithMockedMetadata(std::shared_ptr<DummyModelWithMockedMetadata> modelInstance) :
        modelInstance(modelInstance) {}
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return std::make_shared<ModelWithDummyModelWithMockedMetadata>("dummy", modelInstance);
    }
};

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, ShapesNotMatchBetweenDLModelAndCustomNode) {
    const size_t demultiplyCount = 33;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, demultiplyCount},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10;FP32"},
                {"out_OutputNumbers", "1,25,12;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["custom_node"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "in_InputNumbers_1"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node", {{"out", pipelineOutputName}}}};

    auto dummyModelInstance = std::make_shared<DummyModelWithMockedMetadata>(
        tensor_map_t{
            {DUMMY_MODEL_INPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                         DUMMY_MODEL_INPUT_NAME,
                                         InferenceEngine::Precision::FP32,
                                         shape_t{1, 10})}},
        tensor_map_t{
            {DUMMY_MODEL_OUTPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                          DUMMY_MODEL_OUTPUT_NAME,
                                          InferenceEngine::Precision::FP32,
                                          shape_t{demultiplyCount, 1, 11})}});  // demultiplyCount, 1, 10 is correct

    ModelManagerWithModelWithDummyModelWithMockedMetadata manager(dummyModelInstance);
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, ShapesNotMatchBetweenCustomNodeAndDLNode) {
    const size_t demultiplyCount = 25;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,10,7;I32"},
                {"out_OutputNumbers", "25,1,12;FP32"}  // 25,1,10 is correct
            }},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers_1"}}}};

    connections["dummy_node"] = {
        {"custom_node", {{"out", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, ShapesNotMatchBetweenCustomNodes) {
    const size_t demultiplyCount = 19;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_2", "19,1,8;I32"},
                {"out_OutputNumbers_1", "19,1,30,7;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,30,10;I32"},  // 1,30,7 is correct
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, DemultiplyCountNotMatchingOutputSecondDimensionValue) {
    const size_t demultiplyCount = 87;
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_2", "87,1,8;I32"},
                {"out_OutputNumbers_1", "86,1,30,7;I32"}  // 87,1,30,7 is correct
            }},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,30,7;I32"},
                {"in_InputNumbers_2", "1,8;I32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_DEMULTIPLY_COUNT_DOES_NOT_MATCH_BLOB_SHARD_COUNT);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, DemultiplyCountNotMatchingOutputShapeBeforeExitNode) {
    const size_t demultiplyCount = 213;
    const std::set<std::string> gatherFrom{"custom_node_1"};
    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers", "220,1,30,7;I32"}  // 213,1,30,7 is correct
            }},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, gatherFrom},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_1", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_DEMULTIPLY_COUNT_DOES_NOT_MATCH_BLOB_SHARD_COUNT);
}

class EnsembleConfigurationValidationWithGather : public EnsembleConfigurationValidationWithCustomNode {};

TEST_F(EnsembleConfigurationValidationWithGather, SuccessfulConfiguration) {
    const size_t demultiplyCount = 13;
    const std::set<std::string> gatherFrom{"custom_node_1"};

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "13,1,700;I32"},
                {"out_OutputNumbers_2", "13,1,8;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,700;I32"},
                {"in_InputNumbers_2", "1,8;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, gatherFrom, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "13,1,2000;FP32"},
                {"out_OutputNumbers", "1,5;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections["custom_node_3"] = {
        {"custom_node_2", {{"out", "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_3", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithGather, SuccessfulConfigurationWithDLNodeAsDemultliplexer) {
    const size_t demultiplyCount = 53;
    const std::set<std::string> gatherFrom{"dummy_node"};

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, demultiplyCount},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, gatherFrom, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "53,1,2000;FP32"},
                {"out_OutputNumbers", "1,5;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["custom_node_1"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"out", "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    auto dummyModelInstance = std::make_shared<DummyModelWithMockedMetadata>(
        tensor_map_t{
            {DUMMY_MODEL_INPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                         DUMMY_MODEL_INPUT_NAME,
                                         InferenceEngine::Precision::FP32,
                                         shape_t{1, demultiplyCount, 10})}},
        tensor_map_t{
            {DUMMY_MODEL_OUTPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                          DUMMY_MODEL_OUTPUT_NAME,
                                          InferenceEngine::Precision::FP32,
                                          shape_t{demultiplyCount, 1, 10})}});

    ModelManagerWithModelWithDummyModelWithMockedMetadata manager(dummyModelInstance);
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithGather, SuccessfulConfigurationWithDLNodeAsGather) {
    const size_t demultiplyCount = 102;
    const std::set<std::string> gatherFrom{"custom_node_1"};

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers", "102,1,2000;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,2000;I32"},
                {"out_OutputNumbers", "1,10;FP32"}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, std::nullopt, gatherFrom},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"out", "in_InputNumbers"}}}};

    connections["dummy_node"] = {
        {"custom_node_2", {{"out", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, pipelineOutputName}}}};

    auto dummyModelInstance = std::make_shared<DummyModelWithMockedMetadata>(
        tensor_map_t{
            {DUMMY_MODEL_INPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                         DUMMY_MODEL_INPUT_NAME,
                                         InferenceEngine::Precision::FP32,
                                         shape_t{demultiplyCount, 1, 10})}},
        tensor_map_t{
            {DUMMY_MODEL_OUTPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                          DUMMY_MODEL_OUTPUT_NAME,
                                          InferenceEngine::Precision::FP32,
                                          shape_t{1, demultiplyCount, 10})}});

    ModelManagerWithModelWithDummyModelWithMockedMetadata manager(dummyModelInstance);
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::OK);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, MultipleGathersNotAllowedInNonExitNode) {
    const size_t demultiplyCount1 = 11;
    const size_t demultiplyCount2 = 43;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount1, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "11,1,700;I32"},
                {"out_OutputNumbers_2", "11,1,8;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount2, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,700;I32"},
                {"in_InputNumbers_2", "1,8;FP32"},
                {"out_OutputNumbers", "43,1,2000;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {"custom_node_1", "custom_node_2"}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "11,43,1,2000;FP32"},
                {"out_OutputNumbers", "1,5;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections["custom_node_3"] = {
        {"custom_node_2", {{"out", "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_3", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_MANUAL_GATHERING_FROM_MULTIPLE_NODES_NOT_SUPPORTED);
}

TEST_F(EnsembleConfigurationValidationWithGather, ShapesNotMatchBetweenDLModelAndCustomNode) {
    const size_t demultiplyCount = 53;
    const std::set<std::string> gatherFrom{"dummy_node"};

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, demultiplyCount},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, gatherFrom, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "53,1,2000;FP32"},
                {"out_OutputNumbers", "1,5;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["dummy_node"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, DUMMY_MODEL_INPUT_NAME}}}};

    connections["custom_node_1"] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"out", "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_2", {{"out", pipelineOutputName}}}};

    auto dummyModelInstance = std::make_shared<DummyModelWithMockedMetadata>(
        tensor_map_t{
            {DUMMY_MODEL_INPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                         DUMMY_MODEL_INPUT_NAME,
                                         InferenceEngine::Precision::FP32,
                                         shape_t{1, demultiplyCount, 10})}},
        tensor_map_t{
            {DUMMY_MODEL_OUTPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                          DUMMY_MODEL_OUTPUT_NAME,
                                          InferenceEngine::Precision::FP32,
                                          shape_t{demultiplyCount, 1, 11})}});  // demultiplyCount, 1, 10 is correct

    ModelManagerWithModelWithDummyModelWithMockedMetadata manager(dummyModelInstance);
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithGather, ShapesNotMatchBetweenCustomNodeAndDLNode) {
    const size_t demultiplyCount = 102;
    const std::set<std::string> gatherFrom{"custom_node_1"};

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,10;FP32"},
                {"out_OutputNumbers", "102,1,2000;I32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,2000;I32"},
                {"out_OutputNumbers", "1,10;FP32"}}},
        {NodeKind::DL, "dummy_node", "dummy", std::nullopt, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_OUTPUT_NAME}}, std::nullopt, gatherFrom},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"out", "in_InputNumbers"}}}};

    connections["dummy_node"] = {
        {"custom_node_2", {{"out", DUMMY_MODEL_INPUT_NAME}}}};

    connections[EXIT_NODE_NAME] = {
        {"dummy_node", {{DUMMY_MODEL_OUTPUT_NAME, pipelineOutputName}}}};

    auto dummyModelInstance = std::make_shared<DummyModelWithMockedMetadata>(
        tensor_map_t{
            {DUMMY_MODEL_INPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                         DUMMY_MODEL_INPUT_NAME,
                                         InferenceEngine::Precision::FP32,
                                         shape_t{demultiplyCount, 1, 11})}},  // 1, demultiplyCount, 10 is correct
        tensor_map_t{
            {DUMMY_MODEL_OUTPUT_NAME, std::make_shared<ovms::TensorInfo>(
                                          DUMMY_MODEL_OUTPUT_NAME,
                                          InferenceEngine::Precision::FP32,
                                          shape_t{1, demultiplyCount, 10})}});

    ModelManagerWithModelWithDummyModelWithMockedMetadata manager(dummyModelInstance);
    ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithGather, ShapesNotMatchBetweenCustomNodes) {
    const size_t demultiplyCount = 51;
    const std::set<std::string> gatherFrom{"custom_node_1"};

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "51,1,700;I32"},
                {"out_OutputNumbers_2", "51,1,8;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,700;I32"},
                {"in_InputNumbers_2", "1,8;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, gatherFrom, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "51,1,2001;FP32"},  // 51,1,2000 is correct
                {"out_OutputNumbers", "1,5;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections["custom_node_3"] = {
        {"custom_node_2", {{"out", "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_3", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::INVALID_SHAPE);
}

TEST_F(EnsembleConfigurationValidationWithGather, DemultiplyCountNotMatchingInputSecondDimensionValue) {
    const size_t demultiplyCount = 94;
    const std::set<std::string> gatherFrom{"custom_node_1"};

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "94,1,700;I32"},
                {"out_OutputNumbers_2", "94,1,8;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,700;I32"},
                {"in_InputNumbers_2", "1,8;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, gatherFrom, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "95,1,2000;FP32"},  // 94,1,2000 is correct
                {"out_OutputNumbers", "1,5;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections["custom_node_3"] = {
        {"custom_node_2", {{"out", "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_3", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_DEMULTIPLY_COUNT_DOES_NOT_MATCH_BLOB_SHARD_COUNT);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, DemultipliersGatherNodesNotInLIFOOrder) {
    const size_t demultiplyCount1 = 11;
    const size_t demultiplyCount2 = 43;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"1", "out_OutputNumbers_1"}, {"2", "out_OutputNumbers_2"}}, demultiplyCount1, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers_1", "11,1,700;I32"},
                {"out_OutputNumbers_2", "11,1,8;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount2, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers_1", "1,700;I32"},
                {"in_InputNumbers_2", "1,8;FP32"},
                {"out_OutputNumbers", "43,1,2000;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {"custom_node_1"}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "11,1,2000;FP32"},
                {"out_OutputNumbers", "1,100;I32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_2"}},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"1", "in_InputNumbers_1"},
                              {"2", "in_InputNumbers_2"}}}};

    connections["custom_node_3"] = {
        {"custom_node_2", {{"out", "in_InputNumbers"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_3", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_WRONG_DEMULTIPLEXER_GATHER_NODES_ORDER);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, GatherNodeWithoutDemultiplexerPath) {
    const size_t demultiplyCount1 = 11;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount1, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers", "11,1,700;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers", "1,700;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers1", "1,700;FP32"},
                {"in_InputNumbers2", "1,700;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {"custom_node_1"}},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_3"] = {
        {"custom_node_1", {{"out", "in_InputNumbers1"}}},
        {"custom_node_2", {{"out", "in_InputNumbers2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_3", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_WRONG_DEMULTIPLEXER_GATHER_NODES_ORDER);
}

TEST_F(EnsembleConfigurationValidationWithDemultiplexer, DemultiplexerWithoutGatherNodePath) {
    const size_t demultiplyCount1 = 11;

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {{pipelineInputName, pipelineInputName}}},
        {NodeKind::CUSTOM, "custom_node_1", "", std::nullopt, {{"out", "out_OutputNumbers"}}, demultiplyCount1, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,3,10;FP32"},
                {"out_OutputNumbers", "11,1,700;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_2", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {"custom_node_1"}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "11,1,700;FP32"},
                {"out_OutputNumbers", "1,700;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_3", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers", "1,700;FP32"},
                {"out_OutputNumbers", "1,700;FP32"}}},
        {NodeKind::CUSTOM, "custom_node_4", "", std::nullopt, {{"out", "out_OutputNumbers"}}, std::nullopt, {}, mockedLibrary,
            parameters_t{
                {"in_InputNumbers1", "1,700;FP32"},
                {"in_InputNumbers2", "1,700;FP32"},
                {"out_OutputNumbers", "1,2000;FP32"}}},
        {NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, {}},
    };

    pipeline_connections_t connections;

    connections["custom_node_1"] = {
        {ENTRY_NODE_NAME, {{pipelineInputName, "in_InputNumbers"}}}};

    connections["custom_node_2"] = {
        {"custom_node_1", {{"out", "in_InputNumbers"}}}};

    connections["custom_node_3"] = {
        {"custom_node_1", {{"out", "in_InputNumbers"}}}};

    connections["custom_node_4"] = {
        {"custom_node_2", {{"out", "in_InputNumbers1"}}},
        {"custom_node_3", {{"out", "in_InputNumbers2"}}}};

    connections[EXIT_NODE_NAME] = {
        {"custom_node_4", {{"out", pipelineOutputName}}}};

    ConstructorEnabledModelManager manager;
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>("my_new_pipeline", info, connections);
    ASSERT_EQ(pipelineDefinition->validate(manager), StatusCode::PIPELINE_WRONG_DEMULTIPLEXER_GATHER_NODES_ORDER);
}

class EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest : public EnsembleFlowCustomNodeLoadConfigThenExecuteTest {
protected:
    void SetUp() override {
        EnsembleFlowCustomNodeLoadConfigThenExecuteTest::SetUp();
        configJsonFilePath = directoryPath + "/ovms_config_file.json";
    }
    const std::string differentOpsInputName = "pipeline_input";
};

static const char* pipelineCustomNodeDynamicDemultiplexThenDummyConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_dynamic_demultiplex",
            "base_path": "/ovms/bazel-bin/src/lib_node_dynamic_demultiplex.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_dynamic_demultiplex",
                    "type": "custom",
                    "demultiply_count": 0,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "dynamic_demultiplex_results",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest, JustDynamicDemultiplexerConfig) {
    std::unique_ptr<Pipeline> pipeline;
    uint8_t dynamicDemultiplyCount = 3;
    std::vector<float> input{static_cast<float>(dynamicDemultiplyCount), 1, 2, 3, 4, 5, 6, 7, 8, 9};
    this->prepareRequest(request, input, differentOpsInputName);
    this->loadConfiguration(pipelineCustomNodeDynamicDemultiplexThenDummyConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    std::vector<float> expectedOutput(dynamicDemultiplyCount * DUMMY_MODEL_OUTPUT_SIZE);
    for (size_t i = 0; i < dynamicDemultiplyCount; ++i) {
        std::copy(input.begin(), input.end(), expectedOutput.begin() + i * DUMMY_MODEL_OUTPUT_SIZE);
    }
    std::transform(expectedOutput.begin(), expectedOutput.end(), expectedOutput.begin(),
        [](float f) -> float { return f + 1; });
    this->checkResponse("pipeline_output", response, expectedOutput, {dynamicDemultiplyCount, 1, 10});

    auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(pipelineName);
    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());
    auto& input_A = inputs.at(pipelineInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 10}));
    auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({0, 1, 10}));

    std::shared_ptr<ModelInstance> modelInstance;
    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = manager.getModelInstance("dummy", 1, modelInstance, modelInstanceUnloadGuardPtr);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    tensor_map_t modelInputs = modelInstance->getInputsInfo();
    tensor_map_t modelOutputs = modelInstance->getOutputsInfo();
    ASSERT_NE(modelInputs.find("b"), modelInputs.end());
    ASSERT_NE(modelOutputs.find("a"), modelOutputs.end());
    auto inputDummy = modelInputs.at("b");
    EXPECT_EQ(inputDummy->getEffectiveShape(), shape_t({1, 10}));
    auto outputDummy = modelOutputs.at("a");
    EXPECT_EQ(outputDummy->getEffectiveShape(), shape_t({1, 10}));

    modelInputs.clear();
    modelOutputs.clear();

    auto inputs2 = pipelineDefinition->getInputsInfo();
    auto outputs2 = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs2.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs2.find(pipelineOutputName), outputs.end());
    auto input_A2 = inputs2.at(pipelineInputName);
    EXPECT_EQ(input_A2->getEffectiveShape(), shape_t({1, 10}));
    auto output2 = outputs2.at(pipelineOutputName);
    EXPECT_EQ(output2->getEffectiveShape(), shape_t({0, 1, 10}));

    status = manager.getModelInstance("dummy", 1, modelInstance, modelInstanceUnloadGuardPtr);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    modelInputs = modelInstance->getInputsInfo();
    modelOutputs = modelInstance->getOutputsInfo();
    ASSERT_NE(modelInputs.find("b"), modelInputs.end());
    ASSERT_NE(modelOutputs.find("a"), modelOutputs.end());
    auto inputDummy2 = modelInputs.at("b");
    EXPECT_EQ(inputDummy2->getEffectiveShape(), shape_t({1, 10}));
    auto outputDummy2 = modelOutputs.at("a");
    EXPECT_EQ(outputDummy2->getEffectiveShape(), shape_t({1, 10}));
}

static const char* pipelineCustomNodeDynamicDemultiplexThenDummyDemultiplexerConnectedToExitConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_dynamic_demultiplex",
            "base_path": "/ovms/bazel-bin/src/lib_node_dynamic_demultiplex.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_dynamic_demultiplex",
                    "type": "custom",
                    "demultiply_count": 0,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "dynamic_demultiplex_results",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                },
                {"pipeline_output2": {"node_name": "custom_node",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest, JustDynamicDemultiplexerThenDummyBothConnectedToExitConfigMetadataCheck) {
    this->loadConfiguration(pipelineCustomNodeDynamicDemultiplexThenDummyDemultiplexerConnectedToExitConfig);

    auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(pipelineName);
    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());
    const auto& input_A = inputs.at(pipelineInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({1, 10}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({0, 1, 10}));
    const auto& output2 = outputs.at(pipelineOutputName + "2");
    EXPECT_EQ(output2->getEffectiveShape(), shape_t({0, 1, 10}));
}

static const char* pipelineEntryNodeDynamicDemultiplexThenDummyConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_dynamic_demultiplex",
            "base_path": "/ovms/bazel-bin/src/lib_node_dynamic_demultiplex.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "demultiply_count": 0,
            "inputs": ["pipeline_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest, DynamicDemultiplexerEntryThenDummyConfig) {
    std::unique_ptr<Pipeline> pipeline;
    uint8_t dynamicDemultiplyCount = 3;
    std::vector<float> input(3 * DUMMY_MODEL_OUTPUT_SIZE);
    std::iota(input.begin(), input.end(), 42);
    this->prepareRequest(request, input, differentOpsInputName, {dynamicDemultiplyCount, 1, 10});
    this->loadConfiguration(pipelineEntryNodeDynamicDemultiplexThenDummyConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    std::vector<float> expectedOutput = input;
    std::transform(expectedOutput.begin(), expectedOutput.end(), expectedOutput.begin(),
        [](float f) -> float { return f + 1; });
    this->checkResponse("pipeline_output", response, expectedOutput, {dynamicDemultiplyCount, 1, 10});
}

TEST_F(EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest, DynamicDemultiplexerEntryMetadataCorrectness) {
    this->loadConfiguration(pipelineEntryNodeDynamicDemultiplexThenDummyConfig);
    auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(pipelineName);
    ASSERT_NE(pipelineDefinition, nullptr);

    auto inputs = pipelineDefinition->getInputsInfo();
    auto outputs = pipelineDefinition->getOutputsInfo();
    ASSERT_NE(inputs.find(pipelineInputName), inputs.end());
    ASSERT_NE(outputs.find(pipelineOutputName), outputs.end());

    const auto& input_A = inputs.at(pipelineInputName);
    EXPECT_EQ(input_A->getEffectiveShape(), shape_t({0, 1, 10}));
    const auto& output = outputs.at(pipelineOutputName);
    EXPECT_EQ(output->getEffectiveShape(), shape_t({0, 1, 10}));
}

TEST_F(EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest, DynamicDemultiplexerHittingLimitShouldReturnError) {
    std::unique_ptr<Pipeline> pipeline;
    const uint64_t demultiplyLimit = 10'000;  // node.cpp
    uint64_t dynamicDemultiplyCount = demultiplyLimit + 1;
    ASSERT_GT(dynamicDemultiplyCount, demultiplyLimit) << "Current demultiply count type";
    std::vector<float> input{static_cast<float>(dynamicDemultiplyCount), 1, 2, 3, 4, 5, 6, 7, 8, 9};
    this->prepareRequest(request, input, differentOpsInputName);
    this->loadConfiguration(pipelineCustomNodeDynamicDemultiplexThenDummyConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    auto status = pipeline->execute();
    ASSERT_EQ(status, StatusCode::PIPELINE_TOO_LARGE_DIMENSION_SIZE_TO_DEMULTIPLY) << status.string();
}

static const char* pipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumNotInOrderConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_perform_different_operations",
            "base_path": "/ovms/bazel-bin/src/lib_node_perform_different_operations.so"
        },
        {
            "name": "lib_choose_maximum",
            "base_path": "/ovms/bazel-bin/src/lib_node_choose_maximum.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                },
                {
                    "name": "choose_max",
                    "library_name": "lib_choose_maximum",
                    "type": "custom",
                    "gather_from_node": "custom_node",
                    "params": {
                        "selection_criteria": "MAXIMUM_MINIMUM"
                    },
                    "inputs": [
                        {"input_tensors": {"node_name": "dummyNode",
                                           "data_item": "dummy_output"}}
                    ],
                    "outputs": [
                        {"data_item": "maximum_tensor",
                         "alias": "maximum_tensor_alias"}
                    ]
                },
                {
                    "name": "custom_node",
                    "library_name": "lib_perform_different_operations",
                    "type": "custom",
                    "demultiply_count": 4,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}},
                        {"op_factors": {"node_name": "request",
                                           "data_item": "pipeline_factors"}}
                    ],
                    "outputs": [
                        {"data_item": "different_ops_results",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "choose_max",
                                     "data_item": "maximum_tensor_alias"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDemultiplexerLoadConfigThenExecuteTest, DifferentOpsCustomNodeThenDummyThenChooseMaximumNotInOrderConfig) {
    std::unique_ptr<Pipeline> pipeline;
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> factors{1, 3, 2, 2};  // add/sub/multiply/divide
    this->prepareRequest(request, input, differentOpsInputName);
    this->prepareRequest(request, factors, differentOpsFactorsName);
    this->loadConfiguration(pipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumNotInOrderConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    std::vector<float> expectedOutput(4 * DUMMY_MODEL_OUTPUT_SIZE);
    prepareDifferentOpsExpectedOutput(expectedOutput, input, factors);
    std::transform(expectedOutput.begin(), expectedOutput.end(), expectedOutput.begin(),
        [](float f) -> float { return f + 1; });
    std::vector<float> expectedResult = prepareGatherHighestExpectedOutput(expectedOutput, Method::MAXIMUM_MINIMUM);
    this->checkResponse("pipeline_output", response, expectedResult, {1, 10});
}

TEST_F(EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest, DynamicDemultiplexerNoResults) {
    std::unique_ptr<Pipeline> pipeline;
    uint8_t dynamicDemultiplyCount = 0;
    std::vector<float> input{static_cast<float>(dynamicDemultiplyCount), 1, 2, 3, 4, 5, 6, 7, 8, 9};
    this->prepareRequest(request, input, differentOpsInputName);
    this->loadConfiguration(pipelineCustomNodeDynamicDemultiplexThenDummyConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::PIPELINE_DEMULTIPLEXER_NO_RESULTS);
}

TEST_F(EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest, DISABLED_JustDynamicDemultiplexerConfigReturning0Batch) {
    std::unique_ptr<Pipeline> pipeline;
    uint8_t dynamicDemultiplyCount = 0;
    std::vector<float> input{static_cast<float>(dynamicDemultiplyCount), 1, 2, 3, 4, 5, 6, 7, 8, 9};
    this->prepareRequest(request, input, differentOpsInputName);
    this->loadConfiguration(pipelineCustomNodeDynamicDemultiplexThenDummyConfig);
    ASSERT_EQ(manager.createPipeline(pipeline, pipelineName, &request, &response), StatusCode::OK);
    ASSERT_EQ(pipeline->execute(), StatusCode::OK);

    std::vector<float> expectedOutput(dynamicDemultiplyCount * DUMMY_MODEL_OUTPUT_SIZE);
    for (size_t i = 0; i < dynamicDemultiplyCount; ++i) {
        std::copy(input.begin(), input.end(), expectedOutput.begin() + i * DUMMY_MODEL_OUTPUT_SIZE);
    }
    std::transform(expectedOutput.begin(), expectedOutput.end(), expectedOutput.begin(),
        [](float f) -> float { return f + 1; });
    this->checkResponse("pipeline_output", response, expectedOutput, {1, dynamicDemultiplyCount, 10});
}

static const char* pipelineCustomNode2DynamicDemultiplexConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_dynamic_demultiplex",
            "base_path": "/ovms/bazel-bin/src/lib_node_dynamic_demultiplex.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "my_pipeline",
            "inputs": ["pipeline_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_dynamic_demultiplex",
                    "type": "custom",
                    "demultiply_count": 0,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "pipeline_input"}}
                    ],
                    "outputs": [
                        {"data_item": "dynamic_demultiplex_results",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "custom_node2",
                    "library_name": "lib_dynamic_demultiplex",
                    "type": "custom",
                    "demultiply_count": 0,
                    "inputs": [
                        {"input_numbers": {"node_name": "custom_node",
                                           "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "dynamic_demultiplex_results",
                         "alias": "custom_node_output"}
                    ]
                }
            ],
            "outputs": [
                {"pipeline_output": {"node_name": "custom_node2",
                                     "data_item": "custom_node_output"}
                }
            ]
        }
    ]
})";

TEST_F(EnsembleFlowCustomNodeAndDynamicDemultiplexerLoadConfigThenExecuteTest, 2DynamicDemultiplexersNotAllowed) {
    std::unique_ptr<Pipeline> pipeline;
    this->loadConfiguration(pipelineCustomNode2DynamicDemultiplexConfig, StatusCode::NOT_IMPLEMENTED);
}

struct LibraryProduceImages5Dimensions {
    static int execute(const struct CustomNodeTensor* inputs, int, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam*, int) {
        const CustomNodeTensor& input = *inputs;
        std::vector<float> inputData((float*)input.data, ((float*)input.data) + (input.dataBytes / sizeof(float)));

        *outputsCount = 1;
        int elements = 3 * 1 * 1 * 2 * 3;
        *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
        float* result = (float*)malloc(elements * sizeof(float));
        std::vector<float> data;
        for (size_t i = 0; i < 3; i++) {
            for (float v : inputData) {
                data.push_back(v + float(i) + 1.0);
            }
        }
        std::memcpy(result, data.data(), elements * sizeof(float));

        CustomNodeTensor& resultTensor = (*outputs)[0];
        resultTensor.name = "custom_node_output";
        resultTensor.data = reinterpret_cast<uint8_t*>(result);
        resultTensor.dimsCount = 5;
        resultTensor.dims = (uint64_t*)malloc(resultTensor.dimsCount * sizeof(uint64_t));
        resultTensor.dims[0] = 3;
        resultTensor.dims[1] = 1;
        resultTensor.dims[2] = 1;
        resultTensor.dims[3] = 2;
        resultTensor.dims[4] = 3;
        resultTensor.dataBytes = elements * sizeof(float);
        resultTensor.precision = FP32;
        return 0;
    }
    static int getInputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int getOutputsInfo(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int) {
        return 0;
    }
    static int release(void* ptr) {
        free(ptr);
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, DemultiplexerConnectedToNhwcNode) {
    // Prepare request
    const std::vector<float> inputValues{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    PredictRequest request;
    PredictResponse response;
    tensorflow::TensorProto& proto = (*request.mutable_inputs())[pipelineInputName];
    proto.set_dtype(tensorflow::DataType::DT_FLOAT);
    proto.mutable_tensor_content()->assign((char*)inputValues.data(), inputValues.size() * sizeof(float));
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.mutable_tensor_shape()->add_dim()->set_size(3);
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.mutable_tensor_shape()->add_dim()->set_size(2);

    // Prepare model
    ConstructorEnabledModelManager manager;
    ModelConfig config = INCREMENT_1x3x4x5_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(1,3,1,2)"), ovms::StatusCode::OK);
    ASSERT_EQ(config.parseLayoutParameter("nhwc"), ovms::StatusCode::OK);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    // Prepare pipeline
    std::optional<uint32_t> demultiplyCount = 0;
    std::set<std::string> gather = {"image_demultiplexer_node"};
    std::unordered_map<std::string, std::string> aliases{{"custom_node_output", "custom_node_output"}};

    auto inputTensorInfo = std::make_shared<ovms::TensorInfo>(pipelineOutputName,
        InferenceEngine::Precision::FP32,
        shape_t{0, 3, 1, 2},
        InferenceEngine::Layout::ANY);
    const tensor_map_t inputsInfo{{pipelineInputName, inputTensorInfo}};
    auto input_node = std::make_unique<EntryNode>(&request, inputsInfo);
    auto tensorInfo = std::make_shared<ovms::TensorInfo>(pipelineOutputName,
        InferenceEngine::Precision::FP32,
        shape_t{0, 1, 3, 1, 2},
        InferenceEngine::Layout::ANY);
    const tensor_map_t outputsInfo{{pipelineOutputName, tensorInfo}};
    auto output_node = std::make_unique<ExitNode>(&response, outputsInfo, gather);
    auto custom_node = std::make_unique<CustomNode>(
        "image_demultiplexer_node",
        createLibraryMock<LibraryProduceImages5Dimensions>(),
        parameters_t{}, aliases, demultiplyCount);
    auto model_node = std::make_unique<DLNode>("increment_node", "increment_1x3x4x5", std::nullopt, manager);

    auto pipeline = std::make_unique<Pipeline>(*input_node, *output_node);
    pipeline->connect(*input_node, *custom_node, {{pipelineInputName, "any"}});
    pipeline->connect(*custom_node, *model_node, {{"custom_node_output", "input"}});
    pipeline->connect(*model_node, *output_node, {{"output", pipelineOutputName}});

    pipeline->push(std::move(input_node));
    pipeline->push(std::move(custom_node));
    pipeline->push(std::move(model_node));
    pipeline->push(std::move(output_node));

    // Execute
    ASSERT_EQ(pipeline->execute(), ovms::StatusCode::OK);
    checkIncrement4DimResponse(pipelineOutputName, {3.0, 6.0, 4.0, 7.0, 5.0, 8.0, 4.0, 7.0, 5.0, 8.0, 6.0, 9.0, 5.0, 8.0, 6.0, 9.0, 7.0, 10.0}, request, response, {3, 1, 3, 1, 2});
}
