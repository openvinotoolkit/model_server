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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/metrics/llm_metric_reporter.hpp"
#include "src/metrics/metric.hpp"
#include "src/metrics/metric_config.hpp"
#include "src/metrics/metric_family.hpp"
#include "src/metrics/metric_registry.hpp"
#include "src/status.hpp"

using namespace ovms;

using testing::HasSubstr;
using testing::Not;

class LLMMetricsTest : public ::testing::Test {
protected:
    MetricRegistry registry;
    MetricConfig config;

    void SetUp() override {
        ASSERT_EQ(config.loadFromCLIString(true,
                      METRIC_NAME_KV_CACHE_UTILIZATION + "," + METRIC_NAME_KV_CACHE_SIZE + "," + METRIC_NAME_LLM_PROMPT_TOKENS + "," +
                          METRIC_NAME_LLM_GENERATED_TOKENS + "," + METRIC_NAME_LLM_SCHEDULED_REQUESTS + "," + METRIC_NAME_LLM_TTFT + "," +
                          METRIC_NAME_LLM_TPOT),
            StatusCode::OK);
    }
};

TEST_F(LLMMetricsTest, OptionalFamiliesAreNotEnabledByDefault) {
    MetricConfig defaults(true);
    EXPECT_FALSE(defaults.isFamilyEnabled(METRIC_NAME_KV_CACHE_UTILIZATION));
    EXPECT_FALSE(defaults.isFamilyEnabled(METRIC_NAME_KV_CACHE_SIZE));
    EXPECT_FALSE(defaults.isFamilyEnabled(METRIC_NAME_LLM_PROMPT_TOKENS));
    EXPECT_FALSE(defaults.isFamilyEnabled(METRIC_NAME_LLM_GENERATED_TOKENS));
    EXPECT_FALSE(defaults.isFamilyEnabled(METRIC_NAME_LLM_SCHEDULED_REQUESTS));
    EXPECT_FALSE(defaults.isFamilyEnabled(METRIC_NAME_LLM_TTFT));
    EXPECT_FALSE(defaults.isFamilyEnabled(METRIC_NAME_LLM_TPOT));
    LLMMetricReporter reporter(&defaults, &registry, "model", "llm");
    reporter.setCacheUtilization(42);
    reporter.observeCompletedRequest({100, 7, 20, 50});
    reporter.setScheduledRequests(3);
    EXPECT_FALSE(reporter.isCacheUtilizationEnabled());
    EXPECT_FALSE(reporter.hasCompletedRequestMetricsEnabled());
    EXPECT_FALSE(reporter.hasAnyCompletionMetricsEnabled());
    EXPECT_FALSE(reporter.isScheduledRequestsEnabled());
    EXPECT_TRUE(registry.collect().empty());
}

TEST_F(LLMMetricsTest, DisabledMetricsDoNotRegisterSeries) {
    config.metricsEnabled = false;
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    reporter.setCacheUtilization(42);
    reporter.observeCompletedRequest({100, 7, 20, 50});
    EXPECT_FALSE(reporter.isCacheUtilizationEnabled());
    EXPECT_TRUE(registry.collect().empty());
}

TEST_F(LLMMetricsTest, MissingRegistryOrConfigurationDoesNotRegisterSeries) {
    LLMMetricReporter noRegistry(&config, nullptr, "model", "llm");
    LLMMetricReporter noConfig(nullptr, &registry, "model", "llm");
    EXPECT_FALSE(noRegistry.isCacheUtilizationEnabled());
    EXPECT_FALSE(noConfig.isCacheUtilizationEnabled());
    EXPECT_TRUE(registry.collect().empty());
}

TEST_F(LLMMetricsTest, FamiliesCanBeSelectedIndependently) {
    MetricConfig cacheConfig;
    ASSERT_EQ(cacheConfig.loadFromCLIString(true, METRIC_NAME_KV_CACHE_UTILIZATION), StatusCode::OK);
    LLMMetricReporter cacheReporter(&cacheConfig, &registry, "cache_model", "llm");
    EXPECT_TRUE(cacheReporter.isCacheUtilizationEnabled());
    cacheReporter.observeCompletedRequest({100, 7, 20, 50});
    EXPECT_THAT(registry.collect(), Not(HasSubstr(METRIC_NAME_LLM_PROMPT_TOKENS)));

    MetricConfig cacheSizeConfig;
    ASSERT_EQ(cacheSizeConfig.loadFromCLIString(true, METRIC_NAME_KV_CACHE_SIZE), StatusCode::OK);
    LLMMetricReporter cacheSizeReporter(&cacheSizeConfig, &registry, "cache_size_model", "llm");
    EXPECT_TRUE(cacheSizeReporter.isCacheSizeEnabled());
    EXPECT_FALSE(cacheSizeReporter.isCacheUtilizationEnabled());
    cacheSizeReporter.setCacheSizeBytes(4096);
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_kv_cache_size_bytes{name=\"cache_size_model\",node=\"llm\"} 4096\n"));

    MetricConfig promptTokensConfig;
    ASSERT_EQ(promptTokensConfig.loadFromCLIString(true, METRIC_NAME_LLM_PROMPT_TOKENS), StatusCode::OK);
    LLMMetricReporter promptTokensReporter(&promptTokensConfig, &registry, "prompt_model", "llm");
    EXPECT_FALSE(promptTokensReporter.isCacheUtilizationEnabled());
    EXPECT_TRUE(promptTokensReporter.hasCompletedRequestMetricsEnabled());
    promptTokensReporter.observeCompletedRequest({100, 7, 20, 50});
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_prompt_tokens_total{name=\"prompt_model\",node=\"llm\"} 100\n"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr(METRIC_NAME_LLM_GENERATED_TOKENS)));
    promptTokensReporter.deactivate();
    EXPECT_FALSE(promptTokensReporter.hasAnyCompletionMetricsEnabled());
}

TEST_F(LLMMetricsTest, GaugeRetainsLatestValueAcrossScrapes) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    ASSERT_TRUE(reporter.isCacheUtilizationEnabled());
    reporter.setCacheUtilization(25);
    reporter.setCacheUtilization(42.5);
    const std::string firstScrape = registry.collect();
    EXPECT_THAT(firstScrape, HasSubstr("ovms_kv_cache_utilization_percent{name=\"model\",node=\"llm\"} 42.5\n"));
    EXPECT_EQ(registry.collect(), firstScrape);
    reporter.setCacheUtilization(0);
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_kv_cache_utilization_percent{name=\"model\",node=\"llm\"} 0\n"));
    reporter.setCacheUtilization(100);
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_kv_cache_utilization_percent{name=\"model\",node=\"llm\"} 100\n"));
}

TEST_F(LLMMetricsTest, CacheSizeGaugeRetainsLatestValueAcrossScrapes) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    ASSERT_TRUE(reporter.isCacheSizeEnabled());
    reporter.setCacheSizeBytes(1024);
    reporter.setCacheSizeBytes(2048);
    const std::string firstScrape = registry.collect();
    EXPECT_THAT(firstScrape, HasSubstr("ovms_kv_cache_size_bytes{name=\"model\",node=\"llm\"} 2048\n"));
    EXPECT_EQ(registry.collect(), firstScrape);
}

TEST_F(LLMMetricsTest, NewFamiliesCanBeSelectedIndependently) {
    for (const auto& family : {METRIC_NAME_LLM_PROMPT_TOKENS, METRIC_NAME_LLM_GENERATED_TOKENS,
             METRIC_NAME_LLM_SCHEDULED_REQUESTS, METRIC_NAME_LLM_TTFT, METRIC_NAME_LLM_TPOT}) {
        MetricConfig selected;
        ASSERT_EQ(selected.loadFromCLIString(true, family), StatusCode::OK);
        LLMMetricReporter reporter(&selected, &registry, "model", "llm");
        EXPECT_EQ(reporter.hasCompletedRequestMetricsEnabled(),
            family != METRIC_NAME_LLM_SCHEDULED_REQUESTS);
        reporter.observeCompletedRequest({100, 7, 20, 50});
        reporter.setScheduledRequests(3);
        const std::string scrape = registry.collect();
        EXPECT_THAT(scrape, HasSubstr(family));
        for (const auto& other : {METRIC_NAME_LLM_PROMPT_TOKENS, METRIC_NAME_LLM_GENERATED_TOKENS,
                 METRIC_NAME_LLM_SCHEDULED_REQUESTS, METRIC_NAME_LLM_TTFT, METRIC_NAME_LLM_TPOT}) {
            if (other != family) {
                EXPECT_THAT(scrape, Not(HasSubstr(other)));
            }
        }
    }
}

TEST_F(LLMMetricsTest, CompletedRequestMetricsAccumulateAcrossScrapes) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    reporter.observeCompletedRequest({100, 7, 20, 30});
    reporter.observeCompletedRequest({50, 5, 200, 80});
    reporter.observeCompletedRequest({25, 0, 100, 120});
    const std::string scrape = registry.collect();
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_prompt_tokens_total{name=\"model\",node=\"llm\"} 175\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_generated_tokens_total{name=\"model\",node=\"llm\"} 12\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_ttft_milliseconds_count{name=\"model\",node=\"llm\"} 3\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_ttft_milliseconds_sum{name=\"model\",node=\"llm\"} 320\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_ttft_milliseconds_bucket{name=\"model\",node=\"llm\",le=\"20\"} 1\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_ttft_milliseconds_bucket{name=\"model\",node=\"llm\",le=\"250\"} 3\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_tpot_milliseconds_count{name=\"model\",node=\"llm\"} 3\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_tpot_milliseconds_sum{name=\"model\",node=\"llm\"} 230\n"));
    EXPECT_EQ(registry.collect(), scrape);
}

TEST_F(LLMMetricsTest, TTFTHistogramUsesVllmSecondBucketsConvertedToMilliseconds) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    const std::string scrape = registry.collect();
    const std::vector<int> expectedBoundaries{1, 5, 10, 20, 40, 60, 80, 100, 250, 500, 750, 1000,
        2500, 5000, 7500, 10000, 20000, 40000, 80000, 160000, 640000};

    for (const int boundary : expectedBoundaries) {
        EXPECT_THAT(scrape, HasSubstr("ovms_llm_ttft_milliseconds_bucket{name=\"model\",node=\"llm\",le=\"" +
                                      std::to_string(boundary) + "\"} 0\n"));
    }
}

TEST_F(LLMMetricsTest, TPOTHistogramUsesVllmRangeInMilliseconds) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    const std::string scrape = registry.collect();
    const std::vector<int> expectedBoundaries{10, 20, 40, 60, 80, 100, 250, 500, 750, 1000,
        2500, 5000, 7500, 10000, 20000, 40000, 80000};

    for (const int boundary : expectedBoundaries) {
        EXPECT_THAT(scrape, HasSubstr("ovms_llm_tpot_milliseconds_bucket{name=\"model\",node=\"llm\",le=\"" +
                                      std::to_string(boundary) + "\"} 0\n"));
    }
}

TEST_F(LLMMetricsTest, ScheduledRequestsRetainLatestValueAndCanReturnToIdle) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    ASSERT_TRUE(reporter.isScheduledRequestsEnabled());
    reporter.setScheduledRequests(5);
    reporter.setScheduledRequests(2);
    const std::string scrape = registry.collect();
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_scheduled_requests{name=\"model\",node=\"llm\"} 2\n"));
    EXPECT_EQ(registry.collect(), scrape);
    reporter.setScheduledRequests(0);
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_scheduled_requests{name=\"model\",node=\"llm\"} 0\n"));
}

TEST_F(LLMMetricsTest, InvalidTTFTDoesNotDiscardValidTokenCount) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    for (double value : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        reporter.observeCompletedRequest({3, 2, value, 20});
    }
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_prompt_tokens_total{name=\"model\",node=\"llm\"} 12\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_generated_tokens_total{name=\"model\",node=\"llm\"} 8\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_ttft_milliseconds_count{name=\"model\",node=\"llm\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_tpot_milliseconds_count{name=\"model\",node=\"llm\"} 4\n"));
}

TEST_F(LLMMetricsTest, HistogramRetainsDistributionAcrossScrapes) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    reporter.observeCompletedRequest({100, 7, 20, 30});
    reporter.observeCompletedRequest({50, 5, 200, 80});
    reporter.setScheduledRequests(3);
    const std::string scrape = registry.collect();
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_ttft_milliseconds_count{name=\"model\",node=\"llm\"} 2\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_tpot_milliseconds_count{name=\"model\",node=\"llm\"} 2\n"));
    EXPECT_THAT(scrape, HasSubstr("ovms_llm_tpot_milliseconds_sum{name=\"model\",node=\"llm\"} 110\n"));
    EXPECT_EQ(registry.collect(), scrape);
}

TEST_F(LLMMetricsTest, InvalidSamplesLeaveStoredValuesUnchanged) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    reporter.setCacheUtilization(42);
    const std::string scrape = registry.collect();
    for (double value : {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        reporter.setCacheUtilization(value);
    }
    reporter.setCacheUtilization(100.01);
    EXPECT_EQ(registry.collect(), scrape);
}

TEST_F(LLMMetricsTest, PipelineLabelsAreIndependentAndRemovedOnDestruction) {
    LLMMetricReporter first(&config, &registry, "model", "first");
    first.setCacheUtilization(20);
    {
        LLMMetricReporter second(&config, &registry, "model", "second");
        LLMMetricReporter otherGraph(&config, &registry, "other", "first");
        second.setCacheUtilization(40);
        otherGraph.setCacheUtilization(60);
        EXPECT_THAT(registry.collect(), HasSubstr("ovms_kv_cache_utilization_percent{name=\"model\",node=\"first\"} 20\n"));
        EXPECT_THAT(registry.collect(), HasSubstr("ovms_kv_cache_utilization_percent{name=\"model\",node=\"second\"} 40\n"));
        EXPECT_THAT(registry.collect(), HasSubstr("ovms_kv_cache_utilization_percent{name=\"other\",node=\"first\"} 60\n"));
    }
    EXPECT_THAT(registry.collect(), Not(HasSubstr("node=\"second\"")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name=\"other\"")));
    first.setCacheUtilization(30);
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_kv_cache_utilization_percent{name=\"model\",node=\"first\"} 30\n"));
}

TEST_F(LLMMetricsTest, ReloadStartsFreshSeries) {
    {
        LLMMetricReporter reporter(&config, &registry, "model", "llm");
        reporter.setCacheUtilization(42);
        reporter.observeCompletedRequest({100, 7, 20, 30});
        reporter.setScheduledRequests(3);
    }
    EXPECT_TRUE(registry.collect().empty());
    LLMMetricReporter reloaded(&config, &registry, "model", "llm");
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_kv_cache_utilization_percent{name=\"model\",node=\"llm\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_prompt_tokens_total{name=\"model\",node=\"llm\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_generated_tokens_total{name=\"model\",node=\"llm\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_scheduled_requests{name=\"model\",node=\"llm\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_ttft_milliseconds_count{name=\"model\",node=\"llm\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_tpot_milliseconds_count{name=\"model\",node=\"llm\"} 0\n"));
}

TEST_F(LLMMetricsTest, RetainedOldPipelineCannotModifyOrRemoveReplacementMetrics) {
    auto oldReporter = std::make_unique<LLMMetricReporter>(&config, &registry, "model", "llm");
    oldReporter->setCacheUtilization(42);
    oldReporter->observeCompletedRequest({100, 7, 20, 30});
    oldReporter->setScheduledRequests(3);
    oldReporter->deactivate();
    EXPECT_FALSE(oldReporter->isCacheUtilizationEnabled());
    EXPECT_TRUE(registry.collect().empty());

    LLMMetricReporter replacement(&config, &registry, "model", "llm");
    replacement.setCacheUtilization(25);
    replacement.observeCompletedRequest({50, 5, 10, 15});
    replacement.setScheduledRequests(1);
    const std::string scrape = registry.collect();
    oldReporter->setCacheUtilization(90);
    oldReporter->observeCompletedRequest({1000, 100, 1000, 2000});
    oldReporter->setScheduledRequests(10);
    EXPECT_EQ(registry.collect(), scrape);
    oldReporter.reset();
    EXPECT_EQ(registry.collect(), scrape);
}

TEST_F(LLMMetricsTest, RegistrationFailureRollsBackPartialSeries) {
    auto incompatibleFamily = registry.createFamily<MetricGauge>(METRIC_NAME_LLM_PROMPT_TOKENS, "incompatible");
    ASSERT_NE(incompatibleFamily, nullptr);
    EXPECT_THROW(LLMMetricReporter(&config, &registry, "model", "llm"), std::logic_error);
    EXPECT_TRUE(registry.collect().empty());
}

TEST_F(LLMMetricsTest, NewFamilyRegistrationFailureRollsBackPartialSeries) {
    auto incompatibleFamily = registry.createFamily<MetricGauge>(METRIC_NAME_LLM_TTFT, "incompatible");
    ASSERT_NE(incompatibleFamily, nullptr);
    EXPECT_THROW(LLMMetricReporter(&config, &registry, "model", "llm"), std::logic_error);
    EXPECT_TRUE(registry.collect().empty());
}

TEST_F(LLMMetricsTest, ConcurrentUpdatesAndScraping) {
    LLMMetricReporter reporter(&config, &registry, "model", "llm");
    auto writer = std::async(std::launch::async, [&reporter]() {
        for (int i = 0; i < 1000; ++i) {
            reporter.setCacheUtilization(i % 101);
            reporter.observeCompletedRequest({3, 2, 20, 5});
            reporter.setScheduledRequests(i % 5);
        }
    });
    for (int i = 0; i < 100; ++i) {
        EXPECT_THAT(registry.collect(), HasSubstr("# TYPE ovms_kv_cache_utilization_percent gauge"));
    }
    writer.get();
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_prompt_tokens_total{name=\"model\",node=\"llm\"} 3000\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_generated_tokens_total{name=\"model\",node=\"llm\"} 2000\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_ttft_milliseconds_count{name=\"model\",node=\"llm\"} 1000\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_ttft_milliseconds_sum{name=\"model\",node=\"llm\"} 20000\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_tpot_milliseconds_count{name=\"model\",node=\"llm\"} 1000\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("ovms_llm_tpot_milliseconds_sum{name=\"model\",node=\"llm\"} 5000\n"));
}
