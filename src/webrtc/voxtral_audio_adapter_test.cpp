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
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "voxtral_audio_adapter.hpp"

namespace ovms {
namespace {

struct FakeInferenceState {
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<float> receivedAudio;
    bool audioEnded = false;
    bool failGeneration = false;
    bool blockAudioPush = false;
    bool audioPushStarted = false;
    bool releaseAudioPush = false;
};

class FakeInferenceStream : public VoxtralAudioAdapter::InferenceStream {
public:
    explicit FakeInferenceStream(std::shared_ptr<FakeInferenceState> state) :
        state_(std::move(state)) {}

    void pushAudio(const std::vector<float>& audio) override {
        std::unique_lock<std::mutex> lock(state_->mutex);
        if (state_->blockAudioPush) {
            state_->audioPushStarted = true;
            state_->condition.notify_all();
            state_->condition.wait(lock, [this] { return state_->releaseAudioPush; });
        }
        state_->receivedAudio.insert(state_->receivedAudio.end(), audio.begin(), audio.end());
    }

    void endAudio() override {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->audioEnded = true;
        state_->condition.notify_all();
    }

    std::string generate(VoxtralAudioAdapter::TextCallback deltaCallback) override {
        if (state_->failGeneration)
            throw std::runtime_error("fake inference failure");
        if (deltaCallback)
            deltaCallback("partial");
        return "complete";
    }

private:
    std::shared_ptr<FakeInferenceState> state_;
};

VoxtralAudioAdapter createAdapter(const std::shared_ptr<FakeInferenceState>& state) {
    return VoxtralAudioAdapter([state] {
        return std::make_unique<FakeInferenceStream>(state);
    });
}

AudioChunk validFrame() {
    return AudioChunk{{0.25f, 0.5f}, 16000, 0, 1};
}

TEST(VoxtralAudioAdapterTest, RejectsInvalidFrames) {
    auto state = std::make_shared<FakeInferenceState>();
    auto adapter = createAdapter(state);
    auto stream = adapter.createStream({}, {}, {});

    EXPECT_THROW(stream->push(AudioChunk{{}, 16000, 0, 1}), std::invalid_argument);
    EXPECT_THROW(stream->push(AudioChunk{{0.25f}, 0, 0, 1}), std::invalid_argument);
    EXPECT_THROW(stream->push(AudioChunk{{0.25f}, 16000, 0, 0}), std::invalid_argument);
    EXPECT_THROW(stream->push(AudioChunk{{0.25f}, 16000, 0, 2}), std::invalid_argument);
    EXPECT_FALSE(stream->hasAudio());

    stream->close();
    EXPECT_TRUE(stream->isCompleted());
    EXPECT_TRUE(state->audioEnded);
    EXPECT_TRUE(state->receivedAudio.empty());
}

TEST(VoxtralAudioAdapterTest, DrainsFramesAndCompletesCallbacksOnce) {
    auto state = std::make_shared<FakeInferenceState>();
    auto adapter = createAdapter(state);
    std::vector<std::string> deltas;
    std::vector<std::string> finals;
    std::vector<std::exception_ptr> errors;
    auto stream = adapter.createStream(
        [&deltas](const std::string& text) { deltas.push_back(text); },
        [&finals](const std::string& text) { finals.push_back(text); },
        [&errors](std::exception_ptr error) { errors.push_back(std::move(error)); });

    stream->push(validFrame());
    EXPECT_TRUE(stream->hasAudio());
    stream->finish();
    stream->finish();
    stream->close();
    stream->close();

    EXPECT_TRUE(stream->isCompleted());
    EXPECT_TRUE(state->audioEnded);
    EXPECT_EQ(state->receivedAudio, (std::vector<float>{0.25f, 0.5f}));
    EXPECT_EQ(deltas, (std::vector<std::string>{"partial"}));
    EXPECT_EQ(finals, (std::vector<std::string>{"complete"}));
    EXPECT_TRUE(errors.empty());
}

TEST(VoxtralAudioAdapterTest, IgnoresFramesPushedAfterFinish) {
    auto state = std::make_shared<FakeInferenceState>();
    auto adapter = createAdapter(state);
    auto stream = adapter.createStream({}, {}, {});

    stream->finish();
    EXPECT_NO_THROW(stream->push(validFrame()));
    EXPECT_FALSE(stream->hasAudio());
    stream->close();

    EXPECT_TRUE(stream->isCompleted());
    EXPECT_TRUE(state->receivedAudio.empty());
}

TEST(VoxtralAudioAdapterTest, ReportsGenerationFailureWithoutFinalCallback) {
    auto state = std::make_shared<FakeInferenceState>();
    state->failGeneration = true;
    auto adapter = createAdapter(state);
    size_t finalCallbackCount = 0;
    std::exception_ptr error;
    auto stream = adapter.createStream({}, [&finalCallbackCount](const std::string&) {
        ++finalCallbackCount;
    }, [&error](std::exception_ptr exception) {
        error = std::move(exception);
    });

    stream->close();

    ASSERT_TRUE(error);
    EXPECT_THROW(std::rethrow_exception(error), std::runtime_error);
    EXPECT_EQ(finalCallbackCount, 0u);
    EXPECT_TRUE(stream->isCompleted());
}

TEST(VoxtralAudioAdapterTest, RejectsFramesWhenQueueIsFull) {
    constexpr size_t maxQueuedFrames = 1500;
    auto state = std::make_shared<FakeInferenceState>();
    state->blockAudioPush = true;
    auto adapter = createAdapter(state);
    auto stream = adapter.createStream({}, {}, {});

    stream->push(validFrame());
    bool pushStarted = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        pushStarted = state->condition.wait_for(lock, std::chrono::seconds(5), [&] {
            return state->audioPushStarted;
        });
    }
    if (!pushStarted) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->releaseAudioPush = true;
        }
        state->condition.notify_all();
        stream->close();
        FAIL() << "Fake inference stream did not begin consuming audio";
    }
    for (size_t frame = 0; frame < maxQueuedFrames; ++frame)
        stream->push(validFrame());
    EXPECT_THROW(stream->push(validFrame()), std::runtime_error);

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->releaseAudioPush = true;
    }
    state->condition.notify_all();
    stream->close();
    EXPECT_TRUE(stream->isCompleted());
}

TEST(VoxtralAudioAdapterTest, RequiresAStreamFactory) {
    EXPECT_THROW(VoxtralAudioAdapter(VoxtralAudioAdapter::StreamFactory{}), std::invalid_argument);
}

TEST(VoxtralAudioAdapterTest, CreatesAnInferenceStreamPerAudioStream) {
    std::vector<std::shared_ptr<FakeInferenceState>> states{
        std::make_shared<FakeInferenceState>(),
        std::make_shared<FakeInferenceState>(),
    };
    std::atomic<size_t> nextState{0};
    VoxtralAudioAdapter adapter([&states, &nextState] {
        const size_t index = nextState.fetch_add(1);
        return std::make_unique<FakeInferenceStream>(states.at(index));
    });
    auto first = adapter.createStream({}, {}, {});
    auto second = adapter.createStream({}, {}, {});

    first->push(AudioChunk{{0.1f}, 16000, 0, 1});
    second->push(AudioChunk{{0.9f}, 16000, 0, 1});
    first->close();
    second->close();

    EXPECT_TRUE((states[0]->receivedAudio == std::vector<float>{0.1f} &&
                    states[1]->receivedAudio == std::vector<float>{0.9f}) ||
                (states[0]->receivedAudio == std::vector<float>{0.9f} &&
                    states[1]->receivedAudio == std::vector<float>{0.1f}));
    EXPECT_EQ(nextState.load(), 2u);
}

}  // namespace
}  // namespace ovms
