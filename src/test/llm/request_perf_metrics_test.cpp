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

#include <cmath>

#include <gtest/gtest.h>

#include "../../llm/servable.hpp"

namespace ovms {
namespace {

ov::genai::PerfMetrics makePerfMetrics(size_t inputTokenCount, size_t outputTokenCount, double ttftMs) {
    ov::genai::PerfMetrics perfMetrics;
    perfMetrics.num_input_tokens = inputTokenCount;
    perfMetrics.num_generated_tokens = outputTokenCount;
    perfMetrics.ttft = {static_cast<float>(ttftMs), 0.0f};
    perfMetrics.m_evaluated = true;
    return perfMetrics;
}

TEST(RequestPerfMetricsTest, CalculatesLLMMetrics) {
    auto perfMetrics = makePerfMetrics(20, 5, 10.0);

    const auto metrics = getRequestPerfMetrics(perfMetrics);

    EXPECT_EQ(metrics.inputTokenCount, 20);
    EXPECT_EQ(metrics.outputTokenCount, 5);
    EXPECT_EQ(metrics.totalTokenCount, 25);
    EXPECT_DOUBLE_EQ(metrics.llmTtftMs, 10.0);
    EXPECT_DOUBLE_EQ(metrics.ttftMs, 10.0);
    EXPECT_DOUBLE_EQ(metrics.prefillSpeedTps, 2000.0);
}

TEST(RequestPerfMetricsTest, SeparatesLegacyVLMEmbeddingTimeFromTTFT) {
    auto perfMetrics = makePerfMetrics(20, 5, 10.0);

    const auto metrics = getRequestPerfMetrics(perfMetrics, 3.0, true);

    EXPECT_DOUBLE_EQ(metrics.llmTtftMs, 7.0);
    EXPECT_DOUBLE_EQ(metrics.ttftMs, 10.0);
    EXPECT_DOUBLE_EQ(metrics.prefillSpeedTps, 20000.0 / 7.0);
}

TEST(RequestPerfMetricsTest, AddsContinuousBatchingVLMEmbeddingTimeToTTFT) {
    auto perfMetrics = makePerfMetrics(20, 5, 7.0);

    const auto metrics = getRequestPerfMetrics(perfMetrics, 3.0, false);

    EXPECT_DOUBLE_EQ(metrics.llmTtftMs, 7.0);
    EXPECT_DOUBLE_EQ(metrics.ttftMs, 10.0);
    EXPECT_DOUBLE_EQ(metrics.prefillSpeedTps, 20000.0 / 7.0);
}

TEST(RequestPerfMetricsTest, KeepsPrefillSpeedFiniteForZeroTTFT) {
    auto perfMetrics = makePerfMetrics(20, 5, 0.0);

    const auto metrics = getRequestPerfMetrics(perfMetrics);

    EXPECT_TRUE(std::isfinite(metrics.prefillSpeedTps));
    EXPECT_GT(metrics.prefillSpeedTps, 0.0);
}

}  // namespace
}  // namespace ovms
