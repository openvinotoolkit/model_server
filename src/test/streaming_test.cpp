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

#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace ::testing;

template <class W, class R>
class MockedServerReaderWriter final : public ::grpc::ServerReaderWriterInterface<W, R> {
public:
    MOCK_METHOD(void, SendInitialMetadata, (), (override));
    MOCK_METHOD(bool, NextMessageSize, (uint32_t * sz), (override));
    MOCK_METHOD(bool, Read, (R * msg), (override));
    MOCK_METHOD(bool, Write, (const W& msg, ::grpc::WriteOptions options), (override));
};

class StreamingTest : public Test {
protected:
    // Defaults for executor
    const std::string name{"my_graph"};
    const std::string version{"1"};

    ::inference::ModelInferRequest firstRequest;
    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;
};

static void prepareRequest(::inference::ModelInferRequest& request, const std::vector<std::tuple<std::string, float>>& content) {
    request.Clear();
    for (auto const& it : content) {
        prepareKFSInferInputTensor(request, std::get<0>(it) /* name */, std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1}, Precision::FP32}, {std::get<1>(it) /* data */}, false);
    }
}

static void assertResponse(const ::inference::ModelStreamInferResponse& resp, const std::vector<std::tuple<std::string, float>>& expectedContent) {
    ASSERT_EQ(resp.infer_response().outputs_size(), expectedContent.size());
    ASSERT_EQ(resp.infer_response().raw_output_contents_size(), expectedContent.size());
    for (const auto& [name, value] : expectedContent) {
        auto it = std::find_if(resp.infer_response().outputs().begin(), resp.infer_response().outputs().end(), [name](const auto& input) {
            return name == input.name();
        });
        ASSERT_NE(it, resp.infer_response().outputs().end());
        auto index = it - resp.infer_response().outputs().begin();
        const auto& content = resp.infer_response().raw_output_contents(index);
        ASSERT_EQ(content.size(), sizeof(float));
        ASSERT_EQ(*((float*)content.data()), value);
    }
}

static auto Disconnect() {
    return [](::inference::ModelInferRequest* req) {
        return false;
    };
}

static auto Receive(std::vector<std::tuple<std::string, float>> content) {
    return [content](::inference::ModelInferRequest* req) {
        prepareRequest(*req, content);
        return true;
    };
}

static auto Send(std::vector<std::tuple<std::string, float>> content) {
    return [content](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
        assertResponse(msg, content);
        return true;
    };
}

// Regular case
TEST_F(StreamingTest, RequestX_ReceiveX) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "StreamingTestCalculator"
  input_stream: "in"
  output_stream: "out"
}
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    MediapipeGraphExecutor executor{
        this->name, this->version, config,
        {{"in", mediapipe_packet_type_enum::OVTENSOR}},
        {{"out", mediapipe_packet_type_enum::OVTENSOR}},
        {"in"}, {"out"}, {}};

    // Mock receiving 3 requests and disconnection
    prepareRequest(this->firstRequest, {{"in", 3.5f}});
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(Receive({{"in", 7.2f}}))
        .WillOnce(Receive({{"in", 102.4f}}))
        .WillOnce(Disconnect());

    // Expect 3 responses
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(Send({{"out", 4.5f}}))
        .WillOnce(Send({{"out", 8.2f}}))
        .WillOnce(Send({{"out", 103.4f}}));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

// Generative AI case
TEST_F(StreamingTest, Request1_ReceiveX) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "StreamingTestCalculator"
  input_stream: "in"
  input_stream: "signal"
  input_stream_info: {
    tag_index: ':1',
    back_edge: true
  }
  input_stream_handler {
    input_stream_handler: 'ImmediateInputStreamHandler'
  }
  output_stream: "out"
  output_stream: "signal"
  node_options: {
    [type.googleapis.com / mediapipe.StreamingTestCalculatorOptions]: {
      kind: "cycle"
    }
  }
}
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    MediapipeGraphExecutor executor{
        this->name, this->version, config,
        {{"in", mediapipe_packet_type_enum::OVTENSOR}},
        {{"out", mediapipe_packet_type_enum::OVTENSOR}},
        {"in"}, {"out"}, {}};

    // Mock only 1 request and disconnection
    prepareRequest(this->firstRequest, {{"in", 3.5f}});
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(Disconnect());

    // Expect 3 responses (cycle)
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(Send({{"out", 4.5f}}))
        .WillOnce(Send({{"out", 5.5f}}))
        .WillOnce(Send({{"out", 6.5f}}));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

// TODO
// Positive:
// [x] Send X requests receive X responses (regular)
// [x] Send 1 request receive X responses (cycle)
// Send X requests with same timestamp, receive Y responses (partial, sync MP side)

// Automatic timestamping
// Manual timestamping

// Negative:
// Error during graph initialization (bad pbtxt)
// Error installing observer (wrong outputName)
// Error during serialization - how to mock such packet?
// Error during startrun - how?
// Error during graph execution - Process() returning non Ok?
// Error during first deserialization
// Error during subsequent deserializations
// Error when closing all packet sources
// Error waiting until done
// Error when writing to disconnected client
