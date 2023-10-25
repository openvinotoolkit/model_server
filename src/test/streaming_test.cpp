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
    for (auto const& [name, val] : content) {
        prepareKFSInferInputTensor(request, name, std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1}, Precision::FP32}, {val}, false);
    }
    if (timestamp.has_value()) {
        request.set_id(std::to_string(timestamp.value()));
    }
}

static void prepareInvalidRequest(::inference::ModelInferRequest& request, const std::vector<std::string>& inputs, std::optional<int64_t> timestamp = std::nullopt) {
    request.Clear();
    int i = 0;
    for (auto const& name : inputs) {
        prepareKFSInferInputTensor(request, name, std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1}, Precision::FP32}, {1.0f /*data*/}, false);
        request.mutable_raw_input_contents()->Mutable(i++)->clear();
    }
    if (timestamp.has_value()) {
        request.set_id(std::to_string(timestamp.value()));
    }
}

static void assertResponse(const ::inference::ModelStreamInferResponse& resp, const std::vector<std::tuple<std::string, float>>& expectedContent, std::optional<int64_t> expectedTimestamp = std::nullopt) {
    ASSERT_EQ(resp.error_message().size(), 0);
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

static void assertResponseError(const ::inference::ModelStreamInferResponse& resp, const std::string& expectedErrorMesssage) {
    ASSERT_EQ(resp.error_message(), expectedErrorMesssage);
    ASSERT_EQ(resp.infer_response().outputs_size(), 0);
    ASSERT_EQ(resp.infer_response().raw_output_contents_size(), 0);
    // TODO: response id?
}

static auto Disconnect() {
    return [](::inference::ModelInferRequest* req) {
        return false;
    };
}

static auto DisconnectWhenNotified(std::mutex& mtx) {
    return [&mtx](::inference::ModelInferRequest* req) {
        std::lock_guard<std::mutex> lock(mtx);  // waits for lock to be released
        return false;
    };
}

static auto Receive(std::vector<std::tuple<std::string, float>> content) {
    return [content](::inference::ModelInferRequest* req) {
        prepareRequest(*req, content);
        return true;
    };
}

static auto ReceiveWithTimestamp(std::vector<std::tuple<std::string, float>> content, int64_t timestamp) {
    return [content, timestamp](::inference::ModelInferRequest* req) {
        prepareRequest(*req, content);
        req->set_id(std::to_string(timestamp));
        return true;
    };
}

static auto ReceiveWhenNotified(std::vector<std::tuple<std::string, float>> content, std::mutex& mtx) {
    return [content, &mtx](::inference::ModelInferRequest* req) {
        std::lock_guard<std::mutex> lock(mtx);
        prepareRequest(*req, content);
        return true;
    };
}

static auto ReceiveWithTimestampWhenNotified(std::vector<std::tuple<std::string, float>> content, int64_t timestamp, std::mutex& mtx) {
    return [content, timestamp, &mtx](::inference::ModelInferRequest* req) {
        std::lock_guard<std::mutex> lock(mtx);
        prepareRequest(*req, content);
        req->set_id(std::to_string(timestamp));
        return true;
    };
}

static auto ReceiveInvalidWithTimestampWhenNotified(std::vector<std::string> inputs, int64_t timestamp, std::mutex& mtx) {
    return [inputs, timestamp, &mtx](::inference::ModelInferRequest* req) {
        std::lock_guard<std::mutex> lock(mtx);
        prepareInvalidRequest(*req, inputs);
        req->set_id(std::to_string(timestamp));
        return true;
    };
}

static auto DisconnectOnWriteAndNotifyEnd(std::mutex& mtx) {
    mtx.lock();
    return [&mtx](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
        mtx.unlock();
        return false;
    };
}

static auto SendWithTimestamp(std::vector<std::tuple<std::string, float>> content, int64_t timestamp) {
    return [content, timestamp](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
        assertResponse(msg, content, timestamp);
        return true;
    };
}

static auto SendWithTimestampAndNotifyEnd(std::vector<std::tuple<std::string, float>> content, int64_t timestamp, std::mutex& mtx) {
    mtx.lock();
    return [content, timestamp, &mtx](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
        assertResponse(msg, content, timestamp);
        mtx.unlock();
        return true;
    };
}

static auto SendErrorAndNotifyEnd(const std::string& expectedMessage, std::mutex& mtx) {
    mtx.lock();
    return [expectedMessage, &mtx](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
        assertResponseError(msg, expectedMessage);
        mtx.unlock();
        return true;
    };
}

// Positive:
// Send X requests receive X responses (regular)
// Send 1 request receive X responses (cycle)
// Send X requests with same timestamp, receive Y responses (partial, sync MP side)
// Send 1 request, receive Y responses (sync client side)
// Automatic timestamping
// Manual timestamping

// Negative:
// Error during graph initialization (bad pbtxt)
// Error installing observer (wrong outputName)
// Error during graph execution - Process() returning non Ok?
// Error during first deserialization
// Error during subsequent deserializations
// Error waiting until done (this will return any an error during execution - has list of errors)
// Error when writing to disconnected client
// Wrong timestamping (non monotonous) on client side
// Error when using reserved timestamps (Unset, Unstarted, PreStream, PostStream, OneOverPostStream, Done)
// Error when timestamp not an int64

// Regular case + automatic timestamping server-side
TEST_F(StreamingTest, SingleStreamSend3Receive3AutomaticTimestamp) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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
  calculator: "AddOneSingleStreamTestCalculator"
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

    // Mock receiving 3 requests with manually (client) assigned ascending order of timestamp and disconnection
    prepareRequest(this->firstRequest, {{"in", 3.5f}}, 3);  // first request with timestamp 3
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(ReceiveWithTimestamp({{"in", 7.2f}}, 12))   // this is correct because 12 > 3
        .WillOnce(ReceiveWithTimestamp({{"in", 99.9f}}, 99))  // this is also correct because 99 > 12
        .WillOnce(Disconnect());

    // Expect 3 responses with correct timestamps
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
  calculator: "AddOne3CycleIterationsTestCalculator"
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

    // Mock only 1 request and disconnect immediately
    prepareRequest(this->firstRequest, {{"in", 3.5f}});
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(Disconnect());

    // Expect 3 responses (cycle)
    // The AddOne3CycleIterationsTestCalculator produces increasing timestamps
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
  calculator: "AddNumbersMultiInputsOutputsTestCalculator"
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
        {{"in1", mediapipe_packet_type_enum::OVTENSOR},
            {"in2", mediapipe_packet_type_enum::OVTENSOR},
            {"in3", mediapipe_packet_type_enum::OVTENSOR}},
        {{"out1", mediapipe_packet_type_enum::OVTENSOR},
            {"out2", mediapipe_packet_type_enum::OVTENSOR},
            {"out3", mediapipe_packet_type_enum::OVTENSOR}},
        {"in1", "in2", "in3"},
        {"out1", "out2", "out3"},
        {}};

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
  calculator: "AddNumbersMultiInputsOutputsTestCalculator"
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
        {{"in1", mediapipe_packet_type_enum::OVTENSOR},
            {"in2", mediapipe_packet_type_enum::OVTENSOR},
            {"in3", mediapipe_packet_type_enum::OVTENSOR}},
        {{"out1", mediapipe_packet_type_enum::OVTENSOR},
            {"out2", mediapipe_packet_type_enum::OVTENSOR},
            {"out3", mediapipe_packet_type_enum::OVTENSOR}},
        {"in1", "in2", "in3"},
        {"out1", "out2", "out3"},
        {}};

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
  calculator: "AddOneSingleStreamTestCalculator"
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

    // Expect 1 correct response (second request malformed the timestamp)
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestampAndNotifyEnd({{"out", 4.5f}}, 3, mtx));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(StreamingTest, ErrorInstallingObserver) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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
        {"in"}, {"wrong_name"}, {}};  // cannot install observer due to wrong output name (should never happen due to validation)

    EXPECT_CALL(this->stream, Read(_)).Times(0);
    EXPECT_CALL(this->stream, Write(_, _)).Times(0);

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::INTERNAL_ERROR);
}

TEST_F(StreamingTest, ExitOnDisconnectionDuringRead) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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

    prepareRequest(this->firstRequest, {});
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(Disconnect());

    EXPECT_CALL(this->stream, Write(_, _)).Times(0);

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

TEST_F(StreamingTest, ErrorOnDisconnectionDuringWrite) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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

    prepareRequest(this->firstRequest, {{"in", 3.5f}});
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(DisconnectWhenNotified(mtx));

    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(DisconnectOnWriteAndNotifyEnd(mtx));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(StreamingTest, InvalidGraph) {
    // Non existing stream handler
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
  input_stream: "in"
  output_stream: "out"
  input_stream_handler {
    input_stream_handler: 'NonExistingStreamHandler'
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

    prepareRequest(this->firstRequest, {{"in", 3.5f}});
    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(StreamingTest, ErrorDuringFirstRequestDeserialization) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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

    // Invalid request - missing data in buffer
    prepareInvalidRequest(this->firstRequest, {"in"});  // no timestamp specified, server will assign one

    std::mutex mtx;

    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(DisconnectWhenNotified(mtx));
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendErrorAndNotifyEnd(
            Status(StatusCode::INVALID_CONTENT_SIZE).string() + std::string{" - Expected: 4 bytes; Actual: 0 bytes; input name: in; partial deserialization of first request"},
            mtx));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

TEST_F(StreamingTest, ErrorDuringSubsequentRequestDeserializations) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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

    std::mutex mtx[2];

    // Mock receiving 3 requests, the last one malicious
    prepareRequest(this->firstRequest, {{"in", 3.5f}}, 0);  // correct request
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(ReceiveWithTimestamp({{"in", 7.2f}}, 1))                     // correct request
        .WillOnce(ReceiveInvalidWithTimestampWhenNotified({"in"}, 2, mtx[0]))  // invalid request - missing data in buffer
        .WillOnce(DisconnectWhenNotified(mtx[1]));

    // Expect 2 responses, no more due to error
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestamp({{"out", 4.5f}}, 0))
        .WillOnce(SendWithTimestampAndNotifyEnd({{"out", 8.2f}}, 1, mtx[0]))
        .WillOnce(SendErrorAndNotifyEnd(
            Status(StatusCode::INVALID_CONTENT_SIZE).string() + std::string{" - Expected: 4 bytes; Actual: 0 bytes; input name: in; partial deserialization of subsequent requests"},
            mtx[1]));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

TEST_F(StreamingTest, ErrorInProcessStopsStream) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "ErrorInProcessTestCalculator"
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

    prepareRequest(this->firstRequest, {{"in", 3.5f}}, 0);
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(Disconnect());

    EXPECT_CALL(this->stream, Write(_, _)).Times(0);

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(StreamingTest, ManualTimestampWrongType) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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
    this->firstRequest.set_id("not an int");

    std::mutex mtx;

    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(DisconnectWhenNotified(mtx));
    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendErrorAndNotifyEnd(
            Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, "Invalid timestamp format in request id field").string() + std::string{"; partial deserialization of first request"},
            mtx));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}

TEST_F(StreamingTest, ManualTimestampNotInRange) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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

    // Timestamps not allowed in stream
    // Expect continuity of operation and response with error message
    for (auto timestamp : std::vector<int64_t>{
             ::mediapipe::kint64min,      // ::mediapipe::Timestamp::Unset()
             ::mediapipe::kint64min + 1,  // ::mediapipe::Timestamp::Unstarted()
             ::mediapipe::kint64min + 2,  // ::mediapipe::Timestamp::PreStream()
             ::mediapipe::kint64max - 2,  // ::mediapipe::Timestamp::PostStream()
             ::mediapipe::kint64max - 1,  // ::mediapipe::Timestamp::OneOverPostStream()
             ::mediapipe::kint64max,      // ::mediapipe::Timestamp::Done()
         }) {
        std::mutex mtx;
        prepareRequest(this->firstRequest, {{"in", 3.5f}}, timestamp);
        EXPECT_CALL(this->stream, Read(_))
            .WillOnce(DisconnectWhenNotified(mtx));
        EXPECT_CALL(this->stream, Write(_, _))
            .WillOnce(SendErrorAndNotifyEnd(
                Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP).string() + std::string{" - "} + ::mediapipe::Timestamp::CreateNoErrorChecking(timestamp).DebugString() + std::string{"; partial deserialization of first request"},
                mtx));
        ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
    }
}

TEST_F(StreamingTest, ManualTimestampInRange) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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

    // Allowed in stream
    for (auto timestamp : std::vector<::mediapipe::Timestamp>{
             ::mediapipe::Timestamp::Min(),
             ::mediapipe::Timestamp::Max(),
         }) {
        std::mutex mtx;
        prepareRequest(this->firstRequest, {{"in", 3.5f}}, timestamp.Value());
        EXPECT_CALL(this->stream, Read(_))
            .WillOnce(DisconnectWhenNotified(mtx));  // To ensure the read loop is stopped
        EXPECT_CALL(this->stream, Write(_, _))
            .WillOnce(SendWithTimestampAndNotifyEnd({{"out", 4.5f}}, timestamp.Value(), mtx));
        ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
    }
}

TEST_F(StreamingTest, AutomaticTimestampingExceedsMax) {
    const std::string pbTxt{R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "AddOneSingleStreamTestCalculator"
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

    std::mutex mtx[2];

    prepareRequest(this->firstRequest, {{"in", 3.5f}}, ::mediapipe::Timestamp::Max().Value());  // valid
    EXPECT_CALL(this->stream, Read(_))
        .WillOnce(ReceiveWhenNotified({{"in", 10.f}}, mtx[0]))  // automatic timestamping overflow
        .WillOnce(DisconnectWhenNotified(mtx[1]));              // automatic timestamping overflow

    EXPECT_CALL(this->stream, Write(_, _))
        .WillOnce(SendWithTimestampAndNotifyEnd({{"out", 4.5f}}, ::mediapipe::Timestamp::Max().Value(), mtx[0]))
        .WillOnce(SendErrorAndNotifyEnd(
            Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP).string() + std::string{" - Timestamp::OneOverPostStream(); partial deserialization of subsequent requests"},
            mtx[1]));

    ASSERT_EQ(executor.inferStream(this->firstRequest, this->stream), StatusCode::OK);
}
