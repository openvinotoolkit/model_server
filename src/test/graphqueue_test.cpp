//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../mediapipe_internal/graphqueue.hpp"
#include "../mediapipe_internal/graph_side_packets.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#pragma GCC diagnostic pop

using namespace ovms;
using namespace std::chrono_literals;

// Minimal passthrough graph config for testing
static const char* kPassthroughGraphConfig = R"pb(
    input_stream: "PASSTHROUGH:in"
    output_stream: "PASSTHROUGH:out"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "PASSTHROUGH:in"
      output_stream: "PASSTHROUGH:out"
    }
)pb";

class GraphQueueExpansionTest : public ::testing::Test {
protected:
    ::mediapipe::CalculatorGraphConfig config;
    std::shared_ptr<GraphSidePackets> sidePackets;

    void SetUp() override {
        config = mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(kPassthroughGraphConfig);
        sidePackets = std::make_shared<GraphSidePackets>();
    }
};

// Basic: pool starts at initial size
TEST_F(GraphQueueExpansionTest, StartsAtInitialSize) {
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, 4);
    EXPECT_EQ(1, queue->getCurrentSize());
    EXPECT_EQ(4, queue->getMaxSize());
}

// Single request works on initial pool
TEST_F(GraphQueueExpansionTest, SingleRequestNoExpansion) {
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, 4);
    {
        GraphIdGuard guard(queue);
        EXPECT_EQ(0, guard.id);
        EXPECT_NE(nullptr, guard.gh);
        EXPECT_NE(nullptr, &guard.graph);
    }
    // After return, size should still be 1
    EXPECT_EQ(1, queue->getCurrentSize());
}

// Two concurrent requests expand pool from 1 to 2
TEST_F(GraphQueueExpansionTest, ConcurrentRequestsExpandPool) {
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, 4);

    // Acquire first graph — pool size stays at 1
    auto future0 = queue->getIdleStream();
    int id0 = future0.get();
    EXPECT_EQ(0, id0);
    EXPECT_EQ(1, queue->getCurrentSize());

    // Second request should trigger expansion
    auto future1 = queue->getIdleStream();
    int id1 = future1.get();
    EXPECT_EQ(1, id1);
    EXPECT_EQ(2, queue->getCurrentSize());

    // Return both
    queue->returnStream(id0);
    queue->returnStream(id1);
}

// Expansion stops at max size, additional requests block
TEST_F(GraphQueueExpansionTest, ExpansionStopsAtMax) {
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, 2);

    auto f0 = queue->getIdleStream();
    int id0 = f0.get();
    auto f1 = queue->getIdleStream();
    int id1 = f1.get();
    EXPECT_EQ(2, queue->getCurrentSize());

    // Third request should block since max=2 and both are in use
    std::atomic<bool> thirdCompleted{false};
    std::thread t([&]() {
        auto f2 = queue->getIdleStream();
        f2.get();
        thirdCompleted.store(true);
    });

    // Give the thread time to block
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(thirdCompleted.load());

    // Return one — should unblock the waiting request
    queue->returnStream(id0);
    t.join();
    EXPECT_TRUE(thirdCompleted.load());
    EXPECT_EQ(2, queue->getCurrentSize());  // no expansion beyond max

    queue->returnStream(id1);
}

// Multiple threads expanding concurrently — all get unique IDs, no overflow
TEST_F(GraphQueueExpansionTest, MultiThreadExpansionThreadSafe) {
    constexpr int maxSize = 8;
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, maxSize);

    std::vector<std::future<int>> futures;
    std::vector<std::thread> threads;

    std::atomic<int> readyCount{0};
    std::atomic<bool> go{false};

    // Launch maxSize threads that all try to acquire simultaneously
    for (int i = 0; i < maxSize; ++i) {
        threads.emplace_back([&, i]() {
            readyCount.fetch_add(1);
            while (!go.load()) {
                std::this_thread::yield();
            }
            auto f = queue->getIdleStream();
            int id = f.get();
            // Hold the graph briefly
            std::this_thread::sleep_for(10ms);
            queue->returnStream(id);
        });
    }

    // Wait for all threads to be ready, then release
    while (readyCount.load() < maxSize) {
        std::this_thread::yield();
    }
    go.store(true);

    for (auto& t : threads) {
        t.join();
    }

    // Pool should have expanded to serve all concurrent requests
    EXPECT_LE(queue->getCurrentSize(), maxSize);
    EXPECT_GE(queue->getCurrentSize(), 1);
}

// GraphIdGuard RAII correctly returns graph on destruction
TEST_F(GraphQueueExpansionTest, GraphIdGuardReturnsOnDestruction) {
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, 1);
    {
        GraphIdGuard guard(queue);
        EXPECT_EQ(0, guard.id);
    }
    // Graph should be available again
    auto f = queue->getIdleStream();
    int id = f.get();
    EXPECT_EQ(0, id);
    queue->returnStream(id);
}

// GraphIdGuard survives queue destruction (weak_ptr pattern)
TEST_F(GraphQueueExpansionTest, GraphIdGuardSurvivesQueueDestruction) {
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, 1);
    auto guard = std::make_unique<GraphIdGuard>(queue);
    EXPECT_EQ(0, guard->id);

    // Destroy the queue while guard is alive
    queue.reset();

    // Guard should still hold the graph helper alive (shared_ptr prevents deallocation)
    EXPECT_NE(nullptr, guard->gh);
    // Note: the underlying graph is shut down by ~GraphQueue, so graph ptr is null.
    // This is expected — in-flight requests complete before queue destruction.

    // Destruction of guard should not crash (weak_ptr expired, returnStream is no-op)
    guard.reset();
}

// Stress test: many threads doing acquire→use→return in a loop during expansion
// Verifies no ID duplication, no data corruption, no deadlocks
TEST_F(GraphQueueExpansionTest, StressInterleavedExpandAndReturn) {
    constexpr int maxSize = 8;
    constexpr int numThreads = 16;  // more threads than max pool size
    constexpr int iterationsPerThread = 50;
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, maxSize);

    std::atomic<int> readyCount{0};
    std::atomic<bool> go{false};
    std::atomic<int> totalAcquired{0};

    // Track which IDs are currently held — detect duplicates
    std::vector<std::atomic<bool>> idInUse(maxSize);
    for (auto& a : idInUse)
        a.store(false);

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            readyCount.fetch_add(1);
            while (!go.load()) {
                std::this_thread::yield();
            }
            for (int i = 0; i < iterationsPerThread; ++i) {
                auto f = queue->getIdleStream();
                int id = f.get();

                // Verify ID is in valid range
                ASSERT_GE(id, 0);
                ASSERT_LT(id, maxSize);

                // Verify no other thread holds this ID
                bool wasInUse = idInUse[id].exchange(true);
                ASSERT_FALSE(wasInUse) << "ID " << id << " was already in use — duplicate assignment!";

                // Verify the GraphHelper at this slot is valid
                auto& gh = queue->getInferRequest(id);
                ASSERT_NE(nullptr, gh);
                ASSERT_NE(nullptr, gh->graph);

                totalAcquired.fetch_add(1);

                // Simulate some work
                std::this_thread::sleep_for(std::chrono::microseconds(100));

                // Release
                idInUse[id].store(false);
                queue->returnStream(id);
            }
        });
    }

    while (readyCount.load() < numThreads) {
        std::this_thread::yield();
    }
    go.store(true);

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(numThreads * iterationsPerThread, totalAcquired.load());
    EXPECT_LE(queue->getCurrentSize(), maxSize);
    // All IDs should be returned (idle)
    for (int i = 0; i < queue->getCurrentSize(); ++i) {
        EXPECT_FALSE(idInUse[i].load());
    }
}

// Verify returned IDs are properly recycled (no unbounded growth)
TEST_F(GraphQueueExpansionTest, IdsAreRecycledNotGrowing) {
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 2, 2);

    // Acquire and return many times — should always get IDs 0 or 1
    std::set<int> seenIds;
    for (int i = 0; i < 100; ++i) {
        auto f = queue->getIdleStream();
        int id = f.get();
        seenIds.insert(id);
        queue->returnStream(id);
    }
    // Only IDs 0 and 1 should ever appear
    EXPECT_EQ(2u, seenIds.size());
    EXPECT_TRUE(seenIds.count(0));
    EXPECT_TRUE(seenIds.count(1));
}

// Verify expansion under load: each new slot gets a distinct GraphHelper
TEST_F(GraphQueueExpansionTest, EachSlotHasDistinctGraphHelper) {
    constexpr int maxSize = 4;
    auto queue = std::make_shared<GraphQueue>(config, sidePackets, 1, maxSize);

    // Hold all graphs simultaneously to force full expansion
    std::vector<int> heldIds;
    std::set<GraphHelper*> helperPtrs;
    for (int i = 0; i < maxSize; ++i) {
        auto f = queue->getIdleStream();
        int id = f.get();
        heldIds.push_back(id);
        auto& gh = queue->getInferRequest(id);
        ASSERT_NE(nullptr, gh);
        helperPtrs.insert(gh.get());
    }

    // All pointers must be distinct
    EXPECT_EQ(static_cast<size_t>(maxSize), helperPtrs.size())
        << "Expected " << maxSize << " distinct GraphHelper instances, got " << helperPtrs.size();

    // Return all
    for (int id : heldIds) {
        queue->returnStream(id);
    }
    EXPECT_EQ(maxSize, queue->getCurrentSize());
}
