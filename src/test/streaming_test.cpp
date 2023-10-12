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
#include <chrono>
#include <mutex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace ::testing;
using namespace std::chrono_literals;

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

static void prepareRequest(::inference::ModelInferRequest& request, const std::vector<std::tuple<std::string, float>>& content, std::optional<int64_t> timestamp = std::nullopt) {
    request.Clear();
    for (auto const& it : content) {
        prepareKFSInferInputTensor(request, std::get<0>(it) /* name */, std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1}, Precision::FP32}, {std::get<1>(it) /* data */}, false);
    }
    if (timestamp.has_value()) {
        request.set_id(std::to_string(timestamp.value()));
    }
}

static void assertResponse(const ::inference::ModelStreamInferResponse& resp, const std::vector<std::tuple<std::string, float>>& expectedContent, std::optional<int64_t> expectedTimestamp = std::nullopt) {
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
    if (expectedTimestamp.has_value()) {
        ASSERT_EQ(std::to_string(expectedTimestamp.value()), resp.infer_response().id());
    }
}

static auto Disconnect() {
    return [](::inference::ModelInferRequest* req) {
        SPDLOG_DEBUG("Disconnecting on read");
        return false;
    };
}

static auto DisconnectWhenNotified(std::mutex& mtx) {
    return [&mtx](::inference::ModelInferRequest* req) {
        SPDLOG_DEBUG("Waiting to disconnect on read...");
        std::lock_guard<std::mutex> lock(mtx);  // waits for lock to be released
        SPDLOG_DEBUG("Disconnecting on read after waiting");
        return false;
    };
}

static auto Receive(std::vector<std::tuple<std::string, float>> content) {
    return [content](::inference::ModelInferRequest* req) {
        SPDLOG_DEBUG("Preparing the request with no timestamp");
        prepareRequest(*req, content);
        return true;
    };
}

static auto ReceiveWithTimestamp(std::vector<std::tuple<std::string, float>> content, int64_t timestamp) {
    return [content, timestamp](::inference::ModelInferRequest* req) {
        SPDLOG_DEBUG("Preparing the request with timestamp: {}", timestamp);
        prepareRequest(*req, content);
        req->set_id(std::to_string(timestamp));
        return true;
    };
}

static auto ReceiveWithTimestampWhenNotified(std::vector<std::tuple<std::string, float>> content, int64_t timestamp, std::mutex& mtx) {
    return [content, timestamp, &mtx](::inference::ModelInferRequest* req) {
        SPDLOG_DEBUG("Waiting to preparing the request with timestamp: {}", timestamp);
        std::lock_guard<std::mutex> lock(mtx);
        SPDLOG_DEBUG("Preparing the request with timestamp: {} after waiting", timestamp);
        prepareRequest(*req, content);
        req->set_id(std::to_string(timestamp));
        return true;
    };
}

// static auto Send(std::vector<std::tuple<std::string, float>> content) {
//     return [content](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
//         assertResponse(msg, content);
//         return true;
//     };
// }

static auto DisconnectOnWrite() {
    return [](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
        SPDLOG_DEBUG("Disconnecting on write");
        return false;
    };
}

static auto SendWithTimestamp(std::vector<std::tuple<std::string, float>> content, int64_t timestamp) {
    return [content, timestamp](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
        SPDLOG_DEBUG("Assuring response is sent with timestamp: {}", timestamp);
        assertResponse(msg, content, timestamp);
        return true;
    };
}

static auto SendWithTimestampAndNotifyEnd(std::vector<std::tuple<std::string, float>> content, int64_t timestamp, std::mutex& mtx) {
    SPDLOG_DEBUG("Temporarily disabling Read operation");
    mtx.lock();
    return [content, timestamp, &mtx](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
        SPDLOG_DEBUG("Assuring response is sent with timestamp: {} and enabling Read operation", timestamp);
        assertResponse(msg, content, timestamp);
        mtx.unlock();
        return true;
    };
}

// Regular case + automatic timestamping server-side
TEST_F(StreamingTest, SingleStreamSend3Receive3AutomaticTimestamp) {
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
    prepareRequest(this->firstRequest, {{"in", 3.5f}});  // no timestamp specified, server will assign one
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(Receive({{"in", 7.2f}}))    // no timestamp specified, server will assign one
        .WillOnce(Receive({{"in", 102.4f}}))  // no timestamp specified, server will assign one
        .WillOnce(Disconnect());

    // Expect 3 responses
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestamp({{"out", 4.5f}}, 0))
        .WillOnce(SendWithTimestamp({{"out", 8.2f}}, 1))
        .WillOnce(SendWithTimestamp({{"out", 103.4f}}, 2));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

// Regular case + manual timestamping client-side
TEST_F(StreamingTest, SingleStreamSend3Receive3ManualTimestamp) {
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

    // Mock receiving 3 requests with manually (client) assigned descending order of timestamp and disconnection
    prepareRequest(this->firstRequest, {{"in", 3.5f}}, 3);  // first request with timestamp 3
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(ReceiveWithTimestamp({{"in", 7.2f}}, 12))   // this is correct because 12 > 3
        .WillOnce(ReceiveWithTimestamp({{"in", 99.9f}}, 99))  // this is also correct because 99 > 12
        .WillOnce(Disconnect());

    // Expect 1 response because first request had not malformed the timestamp
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestamp({{"out", 4.5f}}, 3))
        .WillOnce(SendWithTimestamp({{"out", 8.2f}}, 12))
        .WillOnce(SendWithTimestamp({{"out", 100.9f}}, 99));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

// Generative AI case + automatic timestamping server-side
TEST_F(StreamingTest, SingleStreamSend1Receive3) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "StreamingCycleTestCalculator"
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
    // The StreamingCycleTestCalculator produces increasing timestamps
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestamp({{"out", 4.5f}}, 1))
        .WillOnce(SendWithTimestamp({{"out", 5.5f}}, 2))
        .WillOnce(SendWithTimestamp({{"out", 6.5f}}, 3));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

// Sending inputs separately for synchronized graph
TEST_F(StreamingTest, MultipleStreamsDeliveredViaMultipleRequests) {
    const std::string pbTxt{R"(
input_stream: "in1"
input_stream: "in2"
input_stream: "in3"
output_stream: "out1"
output_stream: "out2"
output_stream: "out3"
node {
  calculator: "StreamingMultiInputsOutputsTestCalculator"
  input_stream: "in1"
  input_stream: "in2"
  input_stream: "in3"
  output_stream: "out1"
  output_stream: "out2"
  output_stream: "out3"
}
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    MediapipeGraphExecutor executor{
        this->name, this->version, config,
        {{"in1", mediapipe_packet_type_enum::OVTENSOR}, {"in2", mediapipe_packet_type_enum::OVTENSOR}, {"in3", mediapipe_packet_type_enum::OVTENSOR}},
        {{"out1", mediapipe_packet_type_enum::OVTENSOR}, {"out2", mediapipe_packet_type_enum::OVTENSOR}, {"out3", mediapipe_packet_type_enum::OVTENSOR}},
        {"in1", "in2", "in3"}, {"out1", "out2", "out3"}, {}};

    std::mutex mtx;
    const int64_t timestamp = 64;

    prepareRequest(this->firstRequest, {{"in1", 3.5f}}, timestamp);
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(ReceiveWithTimestamp({{"in2", 7.2f}}, timestamp))
        .WillOnce(ReceiveWithTimestamp({{"in3", 102.4f}}, timestamp))
        .WillOnce(DisconnectWhenNotified(mtx));

    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestamp({{"out1", 4.5f}}, timestamp))
        .WillOnce(SendWithTimestamp({{"out2", 8.2f}}, timestamp))
        .WillOnce(SendWithTimestampAndNotifyEnd({{"out3", 103.4f}}, timestamp, mtx));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

// Sending inputs together for synchronized graph
TEST_F(StreamingTest, MultipleStreamsDeliveredViaSingleRequest) {
    const std::string pbTxt{R"(
input_stream: "in1"
input_stream: "in2"
input_stream: "in3"
output_stream: "out1"
output_stream: "out2"
output_stream: "out3"
node {
  calculator: "StreamingMultiInputsOutputsTestCalculator"
  input_stream: "in1"
  input_stream: "in2"
  input_stream: "in3"
  output_stream: "out1"
  output_stream: "out2"
  output_stream: "out3"
}
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    MediapipeGraphExecutor executor{
        this->name, this->version, config,
        {{"in1", mediapipe_packet_type_enum::OVTENSOR}, {"in2", mediapipe_packet_type_enum::OVTENSOR}, {"in3", mediapipe_packet_type_enum::OVTENSOR}},
        {{"out1", mediapipe_packet_type_enum::OVTENSOR}, {"out2", mediapipe_packet_type_enum::OVTENSOR}, {"out3", mediapipe_packet_type_enum::OVTENSOR}},
        {"in1", "in2", "in3"}, {"out1", "out2", "out3"}, {}};

    std::mutex mtx;
    const int64_t timestamp = 64;

    prepareRequest(this->firstRequest, {{"in1", 3.5f}, {"in2", 7.2f}, {"in3", 102.4f}}, timestamp);
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(DisconnectWhenNotified(mtx));

    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestamp({{"out1", 4.5f}}, timestamp))
        .WillOnce(SendWithTimestamp({{"out2", 8.2f}}, timestamp))
        .WillOnce(SendWithTimestampAndNotifyEnd({{"out3", 103.4f}}, timestamp, mtx));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

TEST_F(StreamingTest, WrongOrderOfManualTimestamps) {
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

    std::mutex mtx;

    // Mock receiving 3 requests with manually (client) assigned descending order of timestamp and disconnection
    prepareRequest(this->firstRequest, {{"in", 3.5f}}, 3);  // first request with timestamp 3
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(ReceiveWithTimestampWhenNotified({{"in", 7.2f}}, 2, mtx));  // This should break the execution loop because 2<3

    // Expect 1 response because first request had not malformed the timestamp
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestampAndNotifyEnd({{"out", 4.5f}}, 3, mtx));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::UNKNOWN_ERROR);  // TODO: Proper error
}

TEST_F(StreamingTest, ErrorInstallingObserver) {
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
        {"in"}, {"wrong_name"}, {}};  // cannot install observer due to wrong name

    EXPECT_CALL(this->stream, Read(_)).Times(0);
    EXPECT_CALL(this->stream, Write(_, _)).Times(0);

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::UNKNOWN_ERROR);  // TODO: Proper error
}

TEST_F(StreamingTest, WritingToDisconnectedClient) {
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

    prepareRequest(this->firstRequest, {{"in", 3.5f}});
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(Disconnect());

    // Expect 3 responses
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(DisconnectOnWrite());

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

// TODO
// Positive:
// [x] Send X requests receive X responses (regular)
// [x] Send 1 request receive X responses (cycle)
// [x] Send X requests with same timestamp, receive Y responses (partial, sync MP side)
// [x] Send 1 request, receive Y responses (sync client side)

// [x] Automatic timestamping
// [x] Manual timestamping

// Negative:
// Error during graph initialization (bad pbtxt)
// [x] Error installing observer (wrong outputName)
// Error during serialization - how to mock such packet?
// Error during startrun - how?
// Error during graph execution - Process() returning non Ok?
// Error during first deserialization
// Error during subsequent deserializations
// Error when closing all packet sources (cannot, this function never fails)
// Error waiting until done (this will return any an error during execution - has list of errors)
// [x] Error when writing to disconnected client
// [x] Wrong timestamping (non monotonous) on client side
