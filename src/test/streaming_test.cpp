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

#include "../status.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"

using namespace ovms;
using namespace ::testing;

template <class W, class R>
class MockedServerReaderWriter final : public ::grpc::ServerReaderWriterInterface<W, R> {
 public:
  MOCK_METHOD(void, SendInitialMetadata, (), (override));
  MOCK_METHOD(bool, NextMessageSize, (uint32_t* sz), (override));
  MOCK_METHOD(bool, Read, (R* msg), (override));
  MOCK_METHOD(bool, Write, (const W& msg, ::grpc::WriteOptions options), (override));
};

bool my_func(::inference::ModelInferRequest* msg) {
    return false;
}

TEST(StreamingTest, Ok) {
    const std::string name{"my_graph"};
    const std::string version{"1"};
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "DummyCalculator"
  input_stream: "in"
  output_stream: "out"
}
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    MediapipeGraphExecutor executor{
        name, version, config,
        {{"in",mediapipe_packet_type_enum::UNKNOWN}},
        {{"out",mediapipe_packet_type_enum::UNKNOWN}},
        {"in"}, {"out"}, {}
    };

    ::inference::ModelInferRequest firstRequest;
    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;

    ON_CALL(stream, Read(_))
        .WillByDefault([](::inference::ModelInferRequest* req) {
            return true;
        });
    EXPECT_CALL(stream, Write(_, _)).Times(3);

    ASSERT_EQ(executor.inferStream(&firstRequest, &stream), StatusCode::OK);
}
