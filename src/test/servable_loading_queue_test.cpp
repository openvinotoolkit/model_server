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
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "src/servable_management/servable_loading_queue.hpp"
#include "src/servable_management/servable_loading_task.hpp"

using namespace ovms;

// Gate that the test thread can use to block/unblock the processor
struct TaskGate {
    std::mutex mtx;
    std::condition_variable cv;
    bool open = false;

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return open; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mtx);
        open = true;
        cv.notify_all();
    }
};

class ServableLoadingQueueTest : public ::testing::Test {
protected:
    std::mutex orderMtx;
    std::vector<std::string> executionOrder;

    void recordExecution(const std::string& name) {
        std::lock_guard<std::mutex> lock(orderMtx);
        executionOrder.push_back(name);
    }

    Status defaultProcessor(ServableLoadingTask& task) {
        recordExecution(task.name);
        if (task.name.find("bad") != std::string::npos) {
            return StatusCode::MODEL_NAME_MISSING;
        }
        return StatusCode::OK;
    }
};

TEST_F(ServableLoadingQueueTest, NonPriorityTaskCompletes) {
    ServableLoadingQueue queue;
    queue.start([this](ServableLoadingTask& task) { return defaultProcessor(task); });

    ServableLoadingTask task{ServableLoadingTaskType::LoadModel, "model_a"};
    auto future = queue.scheduleTask(std::move(task));
    auto status = future.get();

    EXPECT_EQ(status, StatusCode::OK);
    ASSERT_EQ(executionOrder.size(), 1);
    EXPECT_EQ(executionOrder[0], "model_a");
}

TEST_F(ServableLoadingQueueTest, PriorityTaskCompletes) {
    ServableLoadingQueue queue;
    queue.start([this](ServableLoadingTask& task) { return defaultProcessor(task); });

    bool isPriorityRequest{true};
    ServableLoadingTask task{ServableLoadingTaskType::LoadModel, "urgent_model", isPriorityRequest};
    auto future = queue.scheduleTask(std::move(task));
    auto status = future.get();

    EXPECT_EQ(status, StatusCode::OK);
    ASSERT_EQ(executionOrder.size(), 1);
    EXPECT_EQ(executionOrder[0], "urgent_model");
}

TEST_F(ServableLoadingQueueTest, ProcessorStatusPropagated) {
    ServableLoadingQueue queue;
    queue.start([this](ServableLoadingTask& task) { return defaultProcessor(task); });

    ServableLoadingTask task{ServableLoadingTaskType::LoadModel, "bad_model"};
    auto future = queue.scheduleTask(std::move(task));

    EXPECT_EQ(future.get(), StatusCode::MODEL_NAME_MISSING);
    ASSERT_EQ(executionOrder.size(), 1);
    EXPECT_EQ(executionOrder[0], "bad_model");
}

TEST_F(ServableLoadingQueueTest, PriorityTaskRunsBeforeQueuedNonPriority) {
    // Block the processor on the first task so we can queue up tasks behind it
    TaskGate gate;
    TaskGate processingStarted;
    ServableLoadingQueue queue;
    queue.start([this, &gate, &processingStarted](ServableLoadingTask& task) -> Status {
        if (task.name == "blocker") {
            processingStarted.release();
            gate.wait();
        }
        return defaultProcessor(task);
    });

    // Task 1: non-priority, will block in processor
    ServableLoadingTask blocker{ServableLoadingTaskType::LoadModel, "blocker"};
    auto blockerFuture = queue.scheduleTask(std::move(blocker));

    // Wait until the worker is actually processing the blocker
    processingStarted.wait();

    // Task 2: non-priority, queued behind blocker
    ServableLoadingTask normal{ServableLoadingTaskType::LoadModel, "normal"};
    auto normalFuture = queue.scheduleTask(std::move(normal));

    // Task 3: priority, should jump ahead of "normal"
    bool isPriorityRequest{true};
    ServableLoadingTask urgent{ServableLoadingTaskType::LoadModel, "urgent", isPriorityRequest};
    auto urgentFuture = queue.scheduleTask(std::move(urgent));

    // Release the blocker — worker processes remaining tasks in queue order
    gate.release();

    EXPECT_EQ(blockerFuture.get(), StatusCode::OK);
    EXPECT_EQ(urgentFuture.get(), StatusCode::OK);
    EXPECT_EQ(normalFuture.get(), StatusCode::OK);

    ASSERT_EQ(executionOrder.size(), 3);
    EXPECT_EQ(executionOrder[0], "blocker");
    EXPECT_EQ(executionOrder[1], "urgent");
    EXPECT_EQ(executionOrder[2], "normal");
}

TEST_F(ServableLoadingQueueTest, StopFinishesCurrentTaskNotPendingOnes) {
    TaskGate gate;
    TaskGate processingStarted;
    ServableLoadingQueue queue;
    queue.start([this, &gate, &processingStarted](ServableLoadingTask& task) -> Status {
        if (task.name == "blocker") {
            processingStarted.release();
            gate.wait();
        }
        return defaultProcessor(task);
    });

    ServableLoadingTask blocker{ServableLoadingTaskType::LoadModel, "blocker"};
    auto f1 = queue.scheduleTask(std::move(blocker));

    processingStarted.wait();

    ServableLoadingTask pending{ServableLoadingTaskType::LoadModel, "pending"};
    auto f2 = queue.scheduleTask(std::move(pending));

    // Signal stop while worker is still blocked — ensures !running before it loops
    queue.requestStop();
    gate.release();
    queue.stop();

    EXPECT_EQ(f1.get(), StatusCode::OK);
    EXPECT_EQ(f2.get(), StatusCode::SERVER_SHUTTING_DOWN);
    ASSERT_EQ(executionOrder.size(), 1);
    EXPECT_EQ(executionOrder[0], "blocker");
}

TEST_F(ServableLoadingQueueTest, MultipleTasksProcessedSerially) {
    std::atomic<int> concurrency{0};
    std::atomic<int> maxConcurrency{0};
    ServableLoadingQueue queue;
    queue.start([&](ServableLoadingTask&) -> Status {
        int cur = ++concurrency;
        int prev = maxConcurrency.load();
        while (cur > prev && !maxConcurrency.compare_exchange_weak(prev, cur)) {
        }
        std::this_thread::yield();
        --concurrency;
        return StatusCode::OK;
    });

    std::vector<std::future<Status>> futures;
    for (int i = 0; i < 10; ++i) {
        ServableLoadingTask task{ServableLoadingTaskType::LoadModel, "m" + std::to_string(i)};
        futures.push_back(queue.scheduleTask(std::move(task)));
    }
    for (auto& f : futures) {
        EXPECT_EQ(f.get(), StatusCode::OK);
    }
    EXPECT_EQ(maxConcurrency.load(), 1);
}
