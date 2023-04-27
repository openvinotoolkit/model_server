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
#include <filesystem>
#include <random>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../ovinferrequestsqueue.hpp"
#include "../timer.hpp"

using namespace testing;

const std::string DUMMY_MODEL_PATH = std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml";

TEST(OVInferRequestQueue, ShortQueue) {
    ov::Core ieCore;
    auto model = ieCore.read_model(DUMMY_MODEL_PATH);
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ovms::OVInferRequestsQueue inferRequestsQueue(compiledModel, 3);
    int reqid;
    reqid = inferRequestsQueue.getIdleStream().get();
    EXPECT_EQ(reqid, 0);
    reqid = inferRequestsQueue.getIdleStream().get();
    EXPECT_EQ(reqid, 1);
    reqid = inferRequestsQueue.getIdleStream().get();
    EXPECT_EQ(reqid, 2);
    inferRequestsQueue.returnStream(0);
    reqid = inferRequestsQueue.getIdleStream().get();
    EXPECT_EQ(reqid, 0);
}

static void releaseStream(ovms::OVInferRequestsQueue& requestsQueue) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    requestsQueue.returnStream(3);
}

enum : unsigned int {
    QUEUE,
    TIMER_END
};

TEST(OVInferRequestQueue, FullQueue) {
    ovms::Timer<TIMER_END> timer;
    ov::Core ieCore;
    auto model = ieCore.read_model(DUMMY_MODEL_PATH);
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ovms::OVInferRequestsQueue inferRequestsQueue(compiledModel, 50);
    int reqid;
    for (int i = 0; i < 50; i++) {
        reqid = inferRequestsQueue.getIdleStream().get();
    }
    timer.start(QUEUE);
    std::thread th(&releaseStream, std::ref(inferRequestsQueue));
    th.detach();
    reqid = inferRequestsQueue.getIdleStream().get();  // it should wait 1s for released request
    timer.stop(QUEUE);

    EXPECT_GT(timer.elapsed<std::chrono::microseconds>(QUEUE), 1'000'000);
    EXPECT_EQ(reqid, 3);
}

static void inferenceSimulate(ovms::OVInferRequestsQueue& ms, std::vector<int>& tv) {
    for (int i = 1; i <= 10; i++) {
        int st = ms.getIdleStream().get();
        int rd = std::rand();
        tv[st] = rd;
        std::mt19937_64 eng{std::random_device{}()};
        std::uniform_int_distribution<> dist{10, 50};  // mocked inference delay range in ms
        std::this_thread::sleep_for(std::chrono::milliseconds{dist(eng)});
        std::mutex mut;
        // test if no other thread updated the content of vector element of reserved id
        EXPECT_EQ(rd, tv[st]);
        ms.returnStream(st);
    }
}

TEST(OVInferRequestQueue, MultiThread) {
    int nireq = 10;            // represnet queue size
    int number_clients = 100;  // represent number of serving clients
    ov::Core ieCore;
    auto model = ieCore.read_model(DUMMY_MODEL_PATH);
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");

    ovms::OVInferRequestsQueue inferRequestsQueue(compiledModel, nireq);

    std::vector<int> test_vector(nireq);  // vector to test if only one thread can manage each element
    std::vector<std::thread> clients;
    for (int i = 0; i < number_clients; ++i) {
        clients.emplace_back(inferenceSimulate, std::ref(inferRequestsQueue), std::ref(test_vector));
    }
    for (auto& t : clients) {
        t.join();
    }
    // wait for all thread to complete successfully
}

TEST(OVInferRequestQueue, AsyncGetInferRequest) {
    ov::Core ieCore;
    auto model = ieCore.read_model(DUMMY_MODEL_PATH);
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    const int nireq = 1;
    ovms::OVInferRequestsQueue inferRequestsQueue(compiledModel, nireq);

    std::future<int> firstStreamRequest = inferRequestsQueue.getIdleStream();
    std::future<int> secondStreamRequest = inferRequestsQueue.getIdleStream();

    EXPECT_EQ(std::future_status::ready, firstStreamRequest.wait_for(std::chrono::microseconds(1)));
    EXPECT_EQ(std::future_status::timeout, secondStreamRequest.wait_for(std::chrono::milliseconds(1)));

    const int firstStreamId = firstStreamRequest.get();
    inferRequestsQueue.returnStream(firstStreamId);
    EXPECT_EQ(std::future_status::ready, secondStreamRequest.wait_for(std::chrono::microseconds(1)));
    const int secondStreamId = secondStreamRequest.get();
    EXPECT_EQ(firstStreamId, secondStreamId);
}
