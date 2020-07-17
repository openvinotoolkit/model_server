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
    std::string model_name = "resnet";
    std::optional<model_version_t> model_version = 1;

    tensorflow::TensorProto image;
    image.mutable_tensor_content()->assign("\x22\x5A\x91\x05\x12\x5F");  // int8_t tensor 2x3: [[0x22, 0x5a, 0x91], [0x05, 0x12, 0x5f]]
    image.mutable_tensor_shape()->add_dim()->set_size(2);
    image.mutable_tensor_shape()->add_dim()->set_size(3);

    (*request.mutable_inputs())["image"] = image;

    // Most basic configuration, just process resnet

    // input   resnet   output
    //  O------->O------->O

    auto input = std::make_unique<EntryNode>(&request);
    auto model = std::make_unique<DLNode>("resnet_node", model_name, model_version);
    auto output = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input, *output);

    pipeline.connect(*input, *model, {"image"});
    pipeline.connect(*model, *output, {"probability"});

    Node* input_ptr = input.get();
    Node* model_ptr = model.get();
    Node* output_ptr = output.get();

    pipeline.push(std::move(input));
    pipeline.push(std::move(model));
    pipeline.push(std::move(output));

    BlobMap map;
    input_ptr->execute();
    input_ptr->fetchResults(map);

    model_ptr->setInputs(*input_ptr, map);
    model_ptr->execute();
    map.clear();
    model_ptr->fetchResults(map);

    output_ptr->setInputs(*model_ptr, map);
    output_ptr->execute();
    map.clear();
    model_ptr->fetchResults(map);

    // ResponseProto ready
}
