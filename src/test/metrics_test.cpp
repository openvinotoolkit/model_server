//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../metric.hpp"
#include "../metric_family.hpp"
#include "../metric_registry.hpp"

using namespace ovms;

using testing::ContainsRegex;
using testing::HasSubstr;

TEST(MetricsCommon, FamilyName) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    EXPECT_EQ(family->getName(), "name");
}

TEST(MetricsCommon, FamilyDesc) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    EXPECT_EQ(family->getDesc(), "desc");
}

TEST(MetricsCounter, IncrementDefault) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricCounter>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 1\n"));
    metric->increment();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 2\n"));
}

TEST(MetricsCounter, Increment) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricCounter>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment(24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 24.43\n"));
    metric->increment(13.57);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 38\n"));
}

TEST(MetricsCounter, IncrementNegative) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricCounter>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment(-24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
}

TEST(MetricsGauge, IncrementDefault) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricGauge>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 1\n"));
    metric->increment();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 2\n"));
}

TEST(MetricsGauge, Increment) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricGauge>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment(24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 24.43\n"));
    metric->increment(13.57);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 38\n"));
}

TEST(MetricsGauge, IncrementNegative) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricGauge>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment(-24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} -24.43\n"));
}

TEST(MetricsGauge, DecrementDefault) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricGauge>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->decrement();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} -1\n"));
    metric->decrement();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} -2\n"));
}

TEST(MetricsGauge, Decrement) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricGauge>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->decrement(24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} -24.43\n"));
    metric->decrement(13.57);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} -38\n"));
}

TEST(MetricsGauge, DecrementNegative) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricGauge>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->decrement(-24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 24.43\n"));
}

TEST(MetricsHistogram, Observe) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricHistogram>("name", "desc")->addMetric({{"label", "value"}}, {1.0, 10.0});
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"1\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"10\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_count{label=\"value\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_sum{label=\"value\"} 0\n"));
    metric->observe(0.01);
    metric->observe(5);
    metric->observe(12);
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"1\"} 1\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"10\"} 2\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"} 3\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_count{label=\"value\"} 3\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_sum{label=\"value\"} 17.01\n"));
}

TEST(MetricsFlow, Counter) {
    MetricRegistry registry;
    auto pass_family = registry.createFamily<MetricCounter>("infer_pass", "number of passed inferences");
    auto fail_family = registry.createFamily<MetricCounter>("infer_fail", "number of failed inferences");

    auto metric = pass_family->addMetric({
        {"protocol", "grpc"},
        {"api", "kfs"},
    });
    for (int i = 0; i < 30; i++)
        metric->increment();

    metric = pass_family->addMetric({
        {"protocol", "grpc"},
        {"api", "tfs"},
    });
    for (int i = 0; i < 15; i++)
        metric->increment();

    metric = fail_family->addMetric({
        {"protocol", "grpc"},
        {"api", "kfs"},
    });
    for (int i = 0; i < 12; i++)
        metric->increment();

    metric = fail_family->addMetric({
        {"protocol", "grpc"},
        {"api", "tfs"},
    });
    for (int i = 0; i < 8; i++)
        metric->increment();

    EXPECT_THAT(registry.collect(), HasSubstr("# HELP infer_pass number of passed inferences\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE infer_pass counter\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP infer_fail number of failed inferences\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE infer_fail counter\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("infer_pass{api=\"kfs\",protocol=\"grpc\"} 30\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("infer_pass{api=\"tfs\",protocol=\"grpc\"} 15\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("infer_fail{api=\"kfs\",protocol=\"grpc\"} 12\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("infer_fail{api=\"tfs\",protocol=\"grpc\"} 8\n"));
}

TEST(MetricsFlow, Gauge) {
    MetricRegistry registry;
    auto nireq_family = registry.createFamily<MetricGauge>("nireq_in_use", "number of inference requests in use");
    auto pipe_family = registry.createFamily<MetricGauge>("pipelines_running", "number of pipelines currently being executed");

    auto metric = nireq_family->addMetric({
        {"model_name", "resnet"},
        {"model_version", "1"},
    });
    for (int i = 0; i < 30; i++) {
        metric->increment();
        metric->increment();
        metric->decrement();
    }

    metric = nireq_family->addMetric({
        {"model_name", "dummy"},
        {"model_version", "2"},
    });
    for (int i = 0; i < 15; i++) {
        metric->increment();
        metric->decrement();
        metric->increment();
    }

    metric = pipe_family->addMetric({
        {"pipeline_name", "ocr"},
    });
    for (int i = 0; i < 12; i++) {
        metric->increment();
        metric->increment();
        metric->decrement();
        metric->decrement();
        metric->increment();
    }

    metric = pipe_family->addMetric({
        {"pipeline_name", "face_blur"},
    });
    for (int i = 0; i < 8; i++) {
        metric->increment();
        metric->increment();
        metric->decrement();
        metric->increment();
    }

    EXPECT_THAT(registry.collect(), HasSubstr("# HELP nireq_in_use number of inference requests in use\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE nireq_in_use gauge\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP pipelines_running number of pipelines currently being executed\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE pipelines_running gauge\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("nireq_in_use{model_name=\"resnet\",model_version=\"1\"} 30\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("nireq_in_use{model_name=\"dummy\",model_version=\"2\"} 15\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("pipelines_running{pipeline_name=\"ocr\"} 12\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("pipelines_running{pipeline_name=\"face_blur\"} 16\n"));
}

TEST(MetricsFlow, Histogram) {
    MetricRegistry registry;
    auto deserialization_family = registry.createFamily<MetricHistogram>("deserialization", "time spent in deserialization");

    auto metric = deserialization_family->addMetric({
                                                        {"model_name", "resnet"},
                                                        {"model_version", "1"},
                                                    },
        {0.1, 1.0, 10.0, 100.0});

    for (int i = 0; i < 30; i++) {
        metric->observe(0.2);
        metric->observe(105);
        metric->observe(0.01);
    }

    // Metadata
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP deserialization time spent in deserialization\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE deserialization histogram\n"));

    // Buckets
    EXPECT_THAT(registry.collect(), HasSubstr("deserialization_bucket{model_name=\"resnet\",model_version=\"1\",le=\"0.1\"} 30\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("deserialization_bucket{model_name=\"resnet\",model_version=\"1\",le=\"1\"} 60\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("deserialization_bucket{model_name=\"resnet\",model_version=\"1\",le=\"10\"} 60\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("deserialization_bucket{model_name=\"resnet\",model_version=\"1\",le=\"100\"} 60\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("deserialization_bucket{model_name=\"resnet\",model_version=\"1\",le=\"+Inf\"} 90\n"));

    // Count
    EXPECT_THAT(registry.collect(), HasSubstr("deserialization_count{model_name=\"resnet\",model_version=\"1\"} 90\n"));

    // Sum
    EXPECT_THAT(registry.collect(), ContainsRegex("deserialization_sum\\{model_name=\"resnet\",model_version=\"1\"\\} 3156.3.*\\n"));
}

// // TODO: Removal of reported metric
// // TODO: Corner cases
// // TODO: Multithreading, test for possible data race
// // TODO: License
