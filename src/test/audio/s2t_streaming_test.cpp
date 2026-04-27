//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <future>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../../audio/speech_to_text/streaming_text_queue.hpp"
#include "../../audio/speech_to_text/s2t_streaming_handler.hpp"
#include "../../sse_utils.hpp"

using ovms::StreamingTextQueue;

// ====================== StreamingTextQueue Tests ======================

TEST(StreamingTextQueueTest, PushAndPop) {
    StreamingTextQueue queue;
    queue.push("hello");
    std::string out;
    EXPECT_TRUE(queue.waitAndPop(out));
    EXPECT_EQ(out, "hello");
}

TEST(StreamingTextQueueTest, FIFOOrder) {
    StreamingTextQueue queue;
    queue.push("first");
    queue.push("second");
    queue.push("third");
    std::string out;
    EXPECT_TRUE(queue.waitAndPop(out));
    EXPECT_EQ(out, "first");
    EXPECT_TRUE(queue.waitAndPop(out));
    EXPECT_EQ(out, "second");
    EXPECT_TRUE(queue.waitAndPop(out));
    EXPECT_EQ(out, "third");
}

TEST(StreamingTextQueueTest, DoneWithEmptyQueue) {
    StreamingTextQueue queue;
    queue.setDone();
    std::string out;
    EXPECT_FALSE(queue.waitAndPop(out));
}

TEST(StreamingTextQueueTest, DoneAfterAllPopped) {
    StreamingTextQueue queue;
    queue.push("data");
    queue.setDone();
    std::string out;
    EXPECT_TRUE(queue.waitAndPop(out));
    EXPECT_EQ(out, "data");
    EXPECT_FALSE(queue.waitAndPop(out));
}

TEST(StreamingTextQueueTest, WaitAndPopBlocksUntilPush) {
    StreamingTextQueue queue;
    std::string result;
    auto future = std::async(std::launch::async, [&queue, &result]() {
        return queue.waitAndPop(result);
    });
    // Give the consumer thread time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue.push("delayed");
    EXPECT_TRUE(future.get());
    EXPECT_EQ(result, "delayed");
}

TEST(StreamingTextQueueTest, WaitAndPopUnblocksOnDone) {
    StreamingTextQueue queue;
    std::string result;
    auto future = std::async(std::launch::async, [&queue, &result]() {
        return queue.waitAndPop(result);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue.setDone();
    EXPECT_FALSE(future.get());
}

TEST(StreamingTextQueueTest, ConcurrentProducerConsumer) {
    StreamingTextQueue queue;
    const int numItems = 100;
    std::vector<std::string> received;

    auto producer = std::async(std::launch::async, [&queue, numItems]() {
        for (int i = 0; i < numItems; ++i) {
            queue.push(std::to_string(i));
        }
        queue.setDone();
    });

    auto consumer = std::async(std::launch::async, [&queue, &received]() {
        std::string out;
        while (queue.waitAndPop(out)) {
            received.push_back(out);
        }
    });

    producer.get();
    consumer.get();

    ASSERT_EQ(static_cast<int>(received.size()), numItems);
    for (int i = 0; i < numItems; ++i) {
        EXPECT_EQ(received[i], std::to_string(i));
    }
}

TEST(StreamingTextQueueTest, EmptyStringPush) {
    StreamingTextQueue queue;
    queue.push("");
    queue.setDone();
    std::string out;
    EXPECT_TRUE(queue.waitAndPop(out));
    EXPECT_EQ(out, "");
    EXPECT_FALSE(queue.waitAndPop(out));
}

// ====================== SSE Utils Tests ======================

TEST(SseUtilsTest, WrapSimpleMessage) {
    std::string result = ovms::wrapTextInServerSideEventMessage("hello");
    EXPECT_EQ(result, "data: hello\n\n");
}

TEST(SseUtilsTest, WrapJsonMessage) {
    std::string result = ovms::wrapTextInServerSideEventMessage("{\"text\":\"hi\"}");
    EXPECT_EQ(result, "data: {\"text\":\"hi\"}\n\n");
}

TEST(SseUtilsTest, WrapDoneMarker) {
    std::string result = ovms::wrapTextInServerSideEventMessage("[DONE]");
    EXPECT_EQ(result, "data: [DONE]\n\n");
}

TEST(SseUtilsTest, WrapEmptyMessage) {
    std::string result = ovms::wrapTextInServerSideEventMessage("");
    EXPECT_EQ(result, "data: \n\n");
}

// ====================== S2tStreamingHandler event serialization Tests ======================

TEST(S2tStreamingHandlerTest, SerializeDeltaEventSimple) {
    std::string result = mediapipe::S2tStreamingHandler::serializeDeltaEvent("hello world");
    EXPECT_EQ(result, "{\"type\":\"transcript.text.delta\",\"delta\":\"hello world\",\"logprobs\":[]}");
}

TEST(S2tStreamingHandlerTest, SerializeDeltaEventEmpty) {
    std::string result = mediapipe::S2tStreamingHandler::serializeDeltaEvent("");
    EXPECT_EQ(result, "{\"type\":\"transcript.text.delta\",\"delta\":\"\",\"logprobs\":[]}");
}

TEST(S2tStreamingHandlerTest, SerializeDeltaEventSpecialCharacters) {
    std::string result = mediapipe::S2tStreamingHandler::serializeDeltaEvent("say \"hello\" & <goodbye>");
    // rapidjson escapes quotes
    EXPECT_NE(result.find("\"delta\""), std::string::npos);
    EXPECT_NE(result.find("say \\\"hello\\\""), std::string::npos);
}

TEST(S2tStreamingHandlerTest, SerializeDeltaEventUnicode) {
    std::string result = mediapipe::S2tStreamingHandler::serializeDeltaEvent("日本語テスト");
    EXPECT_NE(result.find("\"delta\""), std::string::npos);
}

TEST(S2tStreamingHandlerTest, SerializeDoneEventSimple) {
    std::string result = mediapipe::S2tStreamingHandler::serializeDoneEvent("hello world");
    EXPECT_EQ(result, "{\"type\":\"transcript.text.done\",\"text\":\"hello world\",\"logprobs\":[]}");
}

TEST(S2tStreamingHandlerTest, SerializeDoneEventUnicode) {
    std::string result = mediapipe::S2tStreamingHandler::serializeDoneEvent("日本語テスト");
    EXPECT_NE(result.find("\"text\""), std::string::npos);
}

// ====================== Full SSE streaming chunk formatting ======================

TEST(S2tStreamingHandlerTest, FullStreamingChunkFormat) {
    std::string json = mediapipe::S2tStreamingHandler::serializeDeltaEvent("token");
    std::string sse = ovms::wrapTextInServerSideEventMessage(json);
    EXPECT_EQ(sse, "data: {\"type\":\"transcript.text.delta\",\"delta\":\"token\",\"logprobs\":[]}\n\n");
}

TEST(S2tStreamingHandlerTest, FullDoneEventFormat) {
    std::string json = mediapipe::S2tStreamingHandler::serializeDoneEvent("all text");
    std::string sse = ovms::wrapTextInServerSideEventMessage(json);
    EXPECT_EQ(sse, "data: {\"type\":\"transcript.text.done\",\"text\":\"all text\",\"logprobs\":[]}\n\n");
}
