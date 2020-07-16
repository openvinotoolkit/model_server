//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../pipeline.hpp"

#define DEBUG
#include "../timer.hpp"

using namespace ovms;
using namespace tensorflow::serving;

using ::testing::UnorderedElementsAre;

TEST(Ensemble, OneModel) {
    PredictRequest request;
    PredictResponse response;
    ModelInstance instance;

    // Most basic configuration, just process resnet

    // input   resnet   output
    //  O------->O------->O

    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>(&instance);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model);
    pipeline.connect(*model, *output);

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    // MessageQueue::push(Message(pipeline.getEntry()));
    // while (not done pipeline.getExit())
    //     run event loop
}

TEST(Ensemble, TwoModels) {
    PredictRequest request;
    PredictResponse response;
    ModelInstance instance1;
    ModelInstance instance2;

    // Two model configuration, process resnet 2x concurrently

    // input   resnet1  output
    //   /------->O------\
    //  O                 O
    //   \------->O------/
    //         resnet2

    auto input = std::make_unique<EntryNode>(&request);
    auto resnet1 = std::make_unique<DLNode>(&instance1);
    auto resnet2 = std::make_unique<DLNode>(&instance2);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *resnet1);
    pipeline.connect(*input, *resnet2);
    pipeline.connect(*resnet1, *output);
    pipeline.connect(*resnet2, *output);

    pipeline.push(std::move(input));
    pipeline.push(std::move(resnet1));
    pipeline.push(std::move(resnet2));
    pipeline.push(std::move(output));

    // MessageQueue::push(Message(pipeline.getEntry()));
    // while (not done pipeline.getExit())
    //     run event loop
}

TEST(Ensemble, MultipleModels) {
    PredictRequest request;
    PredictResponse response;
    ModelInstance instance;
    Timer timer;
    timer.start("A");

    const int N = 10000;

    auto input = std::make_unique<EntryNode>(&request);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    std::vector<std::unique_ptr<DLNode>> nodes;
    for (int i = 0; i < N; i++) {
        nodes.push_back(std::make_unique<DLNode>(&instance));
        pipeline.connect(*input, *nodes[i]);
        pipeline.connect(*nodes[i], *output);
    }

    pipeline.push(std::move(input));
    pipeline.push(std::move(output));

    for (int i = 0; i < N; i++) {
        pipeline.push(std::move(nodes[i]));
    }

    // MessageQueue::push(Message(pipeline.getEntry()));
    // while (not done pipeline.getExit())
    //     run event loop

    timer.stop("A");
    std::cout << timer.elapsed<std::chrono::microseconds>("A") / 1000 << "ms" << std::endl;
}
