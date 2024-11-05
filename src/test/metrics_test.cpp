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
#include <future>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../metric.hpp"
#include "../metric_family.hpp"
#include "../metric_registry.hpp"

using namespace ovms;

using testing::ContainsRegex;
using testing::HasSubstr;
using testing::Not;

TEST(Metrics, RegistryEmpty) {
    MetricRegistry registry;
    EXPECT_EQ(registry.collect().size(), 0);
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

TEST(MetricsCounter, IncrementRemovedMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    family->remove(metric);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->increment(24.43);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
}

TEST(MetricsCounter, IncrementRemovedFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    registry.remove(family);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->increment(24.43);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
}

TEST(MetricsCounter, IncrementNegativeAmount) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricCounter>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment(-24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
}

TEST(MetricsCounter, RemoveMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    auto metric1 = family->addMetric({{"label", "value"}});
    auto metric2 = family->addMetric({{"other", "data"}});
    family->remove(metric1);
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP name"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE name"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
    EXPECT_THAT(registry.collect(), HasSubstr("name{other=\"data\"} 0\n"));
}

TEST(MetricsCounter, RemoveLastMetricRemovesFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    family->remove(metric);
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsCounter, RemoveRemovedMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    family->remove(metric);
    family->remove(metric);
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsCounter, RemoveFamilyFromWrongRegistry) {
    MetricRegistry registry1;
    MetricRegistry registry2;
    auto family1 = registry1.createFamily<MetricCounter>("name", "desc");
    EXPECT_FALSE(registry2.remove(family1));
}

TEST(MetricsCounter, RemoveMetricFromWrongFamily) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricCounter>("name1", "desc");
    auto family2 = registry.createFamily<MetricCounter>("name2", "desc");
    auto metric = family1->addMetric();
    family2->remove(metric);
}

TEST(MetricsCounter, RemoveEntireFamilyOfMetrics) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricCounter>("name", "desc");
    auto family2 = registry.createFamily<MetricCounter>("fam", "desc");
    auto metric1 = family1->addMetric({{"label", "value"}});
    auto metric2 = family1->addMetric({{"other", "data"}});
    auto metric3 = family2->addMetric({{"other", "data"}});
    EXPECT_TRUE(registry.remove(family1));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("# HELP name")));
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP fam"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("# TYPE name")));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE fam"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{other=\"data\"}")));
    EXPECT_THAT(registry.collect(), HasSubstr("fam{other=\"data\"} 0\n"));
}

TEST(MetricsCounter, RemoveRemovedFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    family->addMetric();
    EXPECT_TRUE(registry.remove(family));
    EXPECT_FALSE(registry.remove(family));
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsCounter, RevertingMetricResetsValue) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 1\n"));
    family->remove(metric);
    metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
}

TEST(MetricsCounter, RevertingFamilyResetsValue) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 1\n"));
    registry.remove(family);
    family = registry.createFamily<MetricCounter>("name", "desc");
    metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
}

TEST(MetricsCounter, CorrectOrderOfTextRepresentation) {
    MetricRegistry registry;
    registry.createFamily<MetricCounter>("family1", "desc")->addMetric({{"label", "value"}})->increment();
    registry.createFamily<MetricCounter>("family2", "desc")->addMetric({{"label", "value"}})->increment();
    std::string expected = R"(# HELP family1 desc
# TYPE family1 counter
family1{label="value"} 1
# HELP family2 desc
# TYPE family2 counter
family2{label="value"} 1
)";
    EXPECT_EQ(registry.collect(), expected);
}

TEST(MetricsCounter, CreateFamilyWithSameNameSameMetricType) {
    MetricRegistry registry;
    EXPECT_NE(registry.createFamily<MetricCounter>("family", "desc"), nullptr);
    EXPECT_NE(registry.createFamily<MetricCounter>("family", "desc"), nullptr);
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsCounter, MultipleFamiliesWithSameNameReferToSameMetric) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricCounter>("family", "desc");
    auto family2 = registry.createFamily<MetricCounter>("family", "desc");
    family1->addMetric()->increment();
    family2->addMetric()->increment();
    std::string expected = R"(# HELP family desc
# TYPE family counter
family 2
)";
    EXPECT_EQ(registry.collect(), expected);
}

TEST(MetricsCounter, CreateFamilyWithSameNameDifferentMetricTypeReturnsNull) {
    MetricRegistry registry;
    EXPECT_NE(registry.createFamily<MetricGauge>("family", "desc"), nullptr);
    EXPECT_EQ(registry.createFamily<MetricCounter>("family", "desc"), nullptr);
}

TEST(MetricsCounter, MultipleMetricWithSameLabelsReferToSameCounter) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricCounter>("family", "desc");
    auto metric1 = family->addMetric({{"name", "resnet"}});
    auto metric2 = family->addMetric({{"name", "resnet"}});
    metric1->increment();
    metric2->increment();
    std::string expected = R"(# HELP family desc
# TYPE family counter
family{name="resnet"} 2
)";
    EXPECT_EQ(registry.collect(), expected);
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

TEST(MetricsGauge, IncrementRemovedMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    family->remove(metric);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->increment(24.43);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
}

TEST(MetricsGauge, IncrementRemovedFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    registry.remove(family);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->increment(24.43);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
}

TEST(MetricsGauge, IncrementNegativeAmount) {
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

TEST(MetricsGauge, DecrementRemovedMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    family->remove(metric);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->decrement(24.43);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
}

TEST(MetricsGauge, DecrementRemovedFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    registry.remove(family);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->decrement(24.43);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
}

TEST(MetricsGauge, DecrementNegativeAmount) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricGauge>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->decrement(-24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 24.43\n"));
}

TEST(MetricsGauge, Set) {
    MetricRegistry registry;
    auto metric = registry.createFamily<MetricGauge>("name", "desc")->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->set(24.43);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 24.43\n"));
    metric->set(-13.57);
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} -13.57\n"));
}

TEST(MetricsGauge, SetRemovedMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    family->remove(metric);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->set(24.43);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
}

TEST(MetricsGauge, SetRemovedFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    registry.remove(family);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->set(24.43);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
}

TEST(MetricsGauge, RemoveMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric1 = family->addMetric({{"label", "value"}});
    auto metric2 = family->addMetric({{"other", "data"}});
    family->remove(metric1);
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP name"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE name"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
    EXPECT_THAT(registry.collect(), HasSubstr("name{other=\"data\"} 0\n"));
}

TEST(MetricsGauge, RemoveLastMetricRemovesFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    family->remove(metric);
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsGauge, RemoveRemovedMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    family->remove(metric);
    family->remove(metric);
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsGauge, RemoveFamilyFromWrongRegistry) {
    MetricRegistry registry1;
    MetricRegistry registry2;
    auto family1 = registry1.createFamily<MetricGauge>("name", "desc");
    EXPECT_FALSE(registry2.remove(family1));
}

TEST(MetricsGauge, RemoveMetricFromWrongFamily) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricGauge>("name1", "desc");
    auto family2 = registry.createFamily<MetricGauge>("name2", "desc");
    auto metric = family1->addMetric();
    family2->remove(metric);
}

TEST(MetricsGauge, RemoveEntireFamilyOfMetrics) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricGauge>("name", "desc");
    auto family2 = registry.createFamily<MetricGauge>("fam", "desc");
    auto metric1 = family1->addMetric({{"label", "value"}});
    auto metric2 = family1->addMetric({{"other", "data"}});
    auto metric3 = family2->addMetric({{"other", "data"}});
    EXPECT_TRUE(registry.remove(family1));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("# HELP name")));
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP fam"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("# TYPE name")));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE fam"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{label=\"value\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name{other=\"data\"}")));
    EXPECT_THAT(registry.collect(), HasSubstr("fam{other=\"data\"} 0\n"));
}

TEST(MetricsGauge, RemoveRemovedFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    family->addMetric();
    EXPECT_TRUE(registry.remove(family));
    EXPECT_FALSE(registry.remove(family));
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsGauge, RevertingMetricResetsValue) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment();
    metric->decrement();
    metric->decrement();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} -1\n"));
    family->remove(metric);
    metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
}

TEST(MetricsGauge, RevertingFamilyResetsValue) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
    metric->increment();
    metric->decrement();
    metric->decrement();
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} -1\n"));
    registry.remove(family);
    family = registry.createFamily<MetricGauge>("name", "desc");
    metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name{label=\"value\"} 0\n"));
}

TEST(MetricsGauge, CorrectOrderOfTextRepresentation) {
    MetricRegistry registry;
    registry.createFamily<MetricGauge>("family1", "desc")->addMetric({{"label", "value"}})->increment();
    registry.createFamily<MetricGauge>("family2", "desc")->addMetric({{"label", "value"}})->decrement();
    std::string expected = R"(# HELP family1 desc
# TYPE family1 gauge
family1{label="value"} 1
# HELP family2 desc
# TYPE family2 gauge
family2{label="value"} -1
)";
    EXPECT_EQ(registry.collect(), expected);
}

TEST(MetricsGauge, CreateFamilyWithSameNameSameMetricType) {
    MetricRegistry registry;
    EXPECT_NE(registry.createFamily<MetricGauge>("family", "desc"), nullptr);
    EXPECT_NE(registry.createFamily<MetricGauge>("family", "desc"), nullptr);
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsGauge, MultipleFamiliesWithSameNameReferToSameMetric) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricGauge>("family", "desc");
    auto family2 = registry.createFamily<MetricGauge>("family", "desc");
    family1->addMetric()->decrement();
    family2->addMetric()->decrement();
    std::string expected = R"(# HELP family desc
# TYPE family gauge
family -2
)";
    EXPECT_EQ(registry.collect(), expected);
}

TEST(MetricsGauge, CreateFamilyWithSameNameDifferentMetricTypeReturnsNull) {
    MetricRegistry registry;
    EXPECT_NE(registry.createFamily<MetricCounter>("family", "desc"), nullptr);
    EXPECT_EQ(registry.createFamily<MetricGauge>("family", "desc"), nullptr);
}

TEST(MetricsGauge, MultipleMetricWithSameLabelsReferToSameCounter) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricGauge>("family", "desc");
    auto metric1 = family->addMetric({{"name", "resnet"}});
    auto metric2 = family->addMetric({{"name", "resnet"}});
    metric1->decrement();
    metric2->decrement();
    std::string expected = R"(# HELP family desc
# TYPE family gauge
family{name="resnet"} -2
)";
    EXPECT_EQ(registry.collect(), expected);
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

TEST(MetricsHistogram, ObserveRemovedMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}}, {1.0, 10.0});
    EXPECT_THAT(registry.collect(), HasSubstr("name_sum{label=\"value\"} 0\n"));
    family->remove(metric);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->observe(0.01);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_sum{label=\"value\"}")));
}

TEST(MetricsHistogram, ObserveRemovedFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}}, {1.0, 10.0});
    EXPECT_THAT(registry.collect(), HasSubstr("name_sum{label=\"value\"} 0\n"));
    registry.remove(family);
    GTEST_SKIP_("Skipped: Using metric after removal is undefined behavior");
    metric->observe(0.01);
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_sum{label=\"value\"}")));
}

TEST(MetricsHistogram, RemoveMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("name", "desc");
    auto metric1 = family->addMetric({{"label", "value"}}, {10});
    auto metric2 = family->addMetric({{"other", "data"}}, {10});
    family->remove(metric1);
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP name"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE name"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_bucket{label=\"value\",le=\"10\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_sum{label=\"value\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_count{label=\"value\"}")));
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{other=\"data\",le=\"10\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{other=\"data\",le=\"+Inf\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_sum{other=\"data\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("name_count{other=\"data\"} 0\n"));
}

TEST(MetricsHistogram, RemoveLastMetricRemovesFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}}, {10});
    family->remove(metric);
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsHistogram, RemoveRemovedMetric) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}});
    family->remove(metric);
    family->remove(metric);
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsHistogram, RemoveEntireFamilyOfMetrics) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricHistogram>("name", "desc");
    auto family2 = registry.createFamily<MetricHistogram>("fam", "desc");
    auto metric1 = family1->addMetric({{"label", "value"}}, {10});
    auto metric2 = family1->addMetric({{"other", "data"}}, {10});
    auto metric3 = family2->addMetric({{"other", "data"}}, {10});
    EXPECT_TRUE(registry.remove(family1));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("# HELP name")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("# TYPE name")));
    EXPECT_THAT(registry.collect(), HasSubstr("# HELP fam"));
    EXPECT_THAT(registry.collect(), HasSubstr("# TYPE fam"));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_bucket{label=\"value\",le=\"10\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_sum{label=\"value\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_count{label=\"value\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_bucket{other=\"data\",le=\"10\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_bucket{other=\"data\",le=\"+Inf\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_sum{other=\"data\"}")));
    EXPECT_THAT(registry.collect(), Not(HasSubstr("name_count{other=\"data\"}")));
    EXPECT_THAT(registry.collect(), HasSubstr("fam_bucket{other=\"data\",le=\"10\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("fam_bucket{other=\"data\",le=\"+Inf\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("fam_sum{other=\"data\"} 0\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("fam_count{other=\"data\"} 0\n"));
}

TEST(MetricsHistogram, RemoveRemovedFamily) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("name", "desc");
    family->addMetric();
    EXPECT_TRUE(registry.remove(family));
    EXPECT_FALSE(registry.remove(family));
    EXPECT_EQ(registry.collect().size(), 0);
}

TEST(MetricsHistogram, RemoveFamilyFromWrongRegistry) {
    MetricRegistry registry1;
    MetricRegistry registry2;
    auto family1 = registry1.createFamily<MetricHistogram>("name", "desc");
    EXPECT_FALSE(registry2.remove(family1));
}

TEST(MetricsHistogram, RemoveMetricFromWrongFamily) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricHistogram>("name1", "desc");
    auto family2 = registry.createFamily<MetricHistogram>("name2", "desc");
    auto metric = family1->addMetric();
    family2->remove(metric);
}

TEST(MetricsHistogram, RevertingMetricResetsValue) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}}, {2.2});
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"} 0\n"));
    metric->observe(2.0);
    metric->observe(2.5);
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"} 2\n"));
    family->remove(metric);
    family = registry.createFamily<MetricHistogram>("name", "desc");
    metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"} 0\n"));
}

TEST(MetricsHistogram, RevertingFamilyResetsValue) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("name", "desc");
    auto metric = family->addMetric({{"label", "value"}}, {2.2});
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"} 0\n"));
    metric->observe(2.0);
    metric->observe(2.5);
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"} 2\n"));
    registry.remove(family);
    family = registry.createFamily<MetricHistogram>("name", "desc");
    metric = family->addMetric({{"label", "value"}});
    EXPECT_THAT(registry.collect(), HasSubstr("name_bucket{label=\"value\",le=\"+Inf\"} 0\n"));
}

TEST(MetricsHistogram, CorrectOrderOfTextRepresentation) {
    MetricRegistry registry;
    registry.createFamily<MetricHistogram>("family1", "desc")->addMetric({{"label", "value"}}, {1.0})->observe(5.2);
    registry.createFamily<MetricHistogram>("family2", "desc")->addMetric({{"label", "value"}}, {2.0})->observe(0.2);
    std::string expected = R"(# HELP family1 desc
# TYPE family1 histogram
family1_count{label="value"} 1
family1_sum{label="value"} 5.2
family1_bucket{label="value",le="1"} 0
family1_bucket{label="value",le="+Inf"} 1
# HELP family2 desc
# TYPE family2 histogram
family2_count{label="value"} 1
family2_sum{label="value"} 0.2
family2_bucket{label="value",le="2"} 1
family2_bucket{label="value",le="+Inf"} 1
)";
    EXPECT_EQ(registry.collect(), expected);
}

TEST(MetricsHistogram, CreateFamilyWithSameNameSameMetricType) {
    MetricRegistry registry;
    EXPECT_NE(registry.createFamily<MetricHistogram>("family", "desc"), nullptr);
    EXPECT_NE(registry.createFamily<MetricHistogram>("family", "desc"), nullptr);
}

TEST(MetricsHistogram, MultipleFamiliesWithSameNameReferToSameMetric) {
    MetricRegistry registry;
    auto family1 = registry.createFamily<MetricHistogram>("family", "desc");
    auto family2 = registry.createFamily<MetricHistogram>("family", "desc");
    family1->addMetric({}, {})->observe(2.5);
    family2->addMetric({}, {})->observe(3.5);
    std::string expected = R"(# HELP family desc
# TYPE family histogram
family_count 2
family_sum 6
family_bucket{le="+Inf"} 2
)";
    EXPECT_EQ(registry.collect(), expected);
}

TEST(MetricsHistogram, CreateFamilyWithSameNameDifferentMetricTypeReturnsNull) {
    MetricRegistry registry;
    EXPECT_NE(registry.createFamily<MetricCounter>("family", "desc"), nullptr);
    EXPECT_EQ(registry.createFamily<MetricHistogram>("family", "desc"), nullptr);
}

// Removing metric from wrong family.
// Removing family from wrong registry.
TEST(MetricsHistogram, MultipleMetricWithSameLabelsAndBucketsReferToSameValue) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("family", "desc");
    auto metric1 = family->addMetric({{"name", "resnet"}}, {});
    auto metric2 = family->addMetric({{"name", "resnet"}}, {});
    metric1->observe(1.2);
    metric2->observe(2.4);
    std::string expected = R"(# HELP family desc
# TYPE family histogram
family_count{name="resnet"} 2
family_sum{name="resnet"} 3.6
family_bucket{name="resnet",le="+Inf"} 2
)";
    EXPECT_EQ(registry.collect(), expected);
}

TEST(MetricsHistogram, MultipleMetricWithSameLabelsButDifferentBucketsReferToSameValue) {
    MetricRegistry registry;
    auto family = registry.createFamily<MetricHistogram>("family", "desc");
    auto metric2 = family->addMetric({{"name", "resnet"}}, {0.1});
    auto metric1 = family->addMetric({{"name", "resnet"}}, {0.2});  // Different bucket
    metric1->observe(1.2);
    metric2->observe(2.4);
    std::string expected = R"(# HELP family desc
# TYPE family histogram
family_count{name="resnet"} 2
family_sum{name="resnet"} 3.6
family_bucket{name="resnet",le="0.1"} 0
family_bucket{name="resnet",le="+Inf"} 2
)";
    EXPECT_EQ(registry.collect(), expected);
}

TEST(MetricsManyOps, Counter) {
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

TEST(MetricsManyOps, Gauge) {
    MetricRegistry registry;
    auto nireq_family = registry.createFamily<MetricGauge>("nireq_in_use", "number of inference requests in use");
    auto pipe_family = registry.createFamily<MetricGauge>("pipelines_running", "number of pipelines currently being executed");

    auto metric = nireq_family->addMetric({
        {"model_name", "resnet"},
        {"model_version", "1"},
    });
    metric->set(2);
    for (int i = 0; i < 30; i++) {
        metric->increment();
        metric->increment();
        metric->decrement();
    }

    metric = nireq_family->addMetric({
        {"model_name", "dummy"},
        {"model_version", "2"},
    });
    metric->set(3);
    for (int i = 0; i < 15; i++) {
        metric->increment();
        metric->decrement();
        metric->increment();
    }

    metric = pipe_family->addMetric({
        {"pipeline_name", "ocr"},
    });
    metric->set(4);
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
    metric->set(5);
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
    EXPECT_THAT(registry.collect(), HasSubstr("nireq_in_use{model_name=\"resnet\",model_version=\"1\"} 32\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("nireq_in_use{model_name=\"dummy\",model_version=\"2\"} 18\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("pipelines_running{pipeline_name=\"ocr\"} 16\n"));
    EXPECT_THAT(registry.collect(), HasSubstr("pipelines_running{pipeline_name=\"face_blur\"} 21\n"));
}

TEST(MetricsManyOps, Histogram) {
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

TEST(MetricsManyOps, MultipleThreads) {
    // Preparation
    const int numberOfWorkers = 30;
    const int numberOfFamilies = 20;
    const int numberOfMetricsPerFamily = 5;

    std::vector<std::unique_ptr<std::thread>> workers;
    std::vector<std::promise<void>> signals(numberOfWorkers);

    std::vector<std::shared_ptr<MetricCounter>> counterMetrics;
    std::vector<std::shared_ptr<MetricGauge>> gaugeMetrics;
    std::vector<std::shared_ptr<MetricHistogram>> histogramMetrics;

    MetricRegistry registry;
    for (int i = 0; i < numberOfFamilies; i++) {
        auto familyC = registry.createFamily<MetricCounter>(std::string{"family_name_c_"} + std::to_string(i), "desc");
        for (int j = 0; j < numberOfMetricsPerFamily; j++)
            counterMetrics.emplace_back(familyC->addMetric({{std::string{"metric_label_name"}, std::string{"metric_value_"} + std::to_string(j)}}));
        auto familyG = registry.createFamily<MetricGauge>(std::string{"family_name_g_"} + std::to_string(i), "desc");
        for (int j = 0; j < numberOfMetricsPerFamily; j++)
            gaugeMetrics.emplace_back(familyG->addMetric({{std::string{"metric_label_name"}, std::string{"metric_value_"} + std::to_string(j)}}));
        auto familyH = registry.createFamily<MetricHistogram>(std::string{"family_name_h_"} + std::to_string(i), "desc");
        for (int j = 0; j < numberOfMetricsPerFamily; j++)
            histogramMetrics.emplace_back(familyH->addMetric({{std::string{"metric_label_name"}, std::string{"metric_value_"} + std::to_string(j)}}, {0.1, 1.0, 10.0}));
    }

    // Parallel execution
    const int numberOfOperations = 1000;
    for (int i = 0; i < numberOfWorkers; i++)
        workers.emplace_back(std::make_unique<std::thread>([this, i, &numberOfOperations, &signals, &counterMetrics, &gaugeMetrics, &histogramMetrics]() {
            signals[i].get_future().get();
            for (int j = 0; j < numberOfOperations; j++) {
                for (auto& metric : counterMetrics)
                    metric->increment(1.5);
                for (auto& metric : gaugeMetrics) {
                    metric->increment(3.25);
                    metric->decrement(2.25);
                }
                for (auto& metric : histogramMetrics) {
                    metric->observe(0.05);
                    metric->observe(0.5);
                    metric->observe(5.0);
                    metric->observe(50.0);
                }
            }
        }));

    std::for_each(signals.begin(), signals.end(), [](auto& sig) { sig.set_value(); });
    std::for_each(workers.begin(), workers.end(), [](auto& thread) { thread->join(); });

    // Expect
    std::string content = registry.collect();
    for (int i = 0; i < numberOfFamilies; i++) {
        for (int j = 0; j < numberOfMetricsPerFamily; j++) {
            // Counters
            // numberOfWorkers * numberOfOperations * 1.5 = 45000
            EXPECT_THAT(content, HasSubstr(std::string{"family_name_c_"} + std::to_string(i) + std::string{"{metric_label_name=\"metric_value_"} + std::to_string(j) + std::string{"\"} 45000\n"}));

            // Gauges
            // numberOfWorkers * numberOfOperations * (3.25 - 2.25) = 30000
            EXPECT_THAT(content, HasSubstr(std::string{"family_name_g_"} + std::to_string(i) + std::string{"{metric_label_name=\"metric_value_"} + std::to_string(j) + std::string{"\"} 30000\n"}));

            // Histograms
            auto prefix = std::string{"family_name_h_"} + std::to_string(i) + std::string{"_bucket{metric_label_name=\"metric_value_"} + std::to_string(j);
            EXPECT_THAT(content, HasSubstr(prefix + std::string{"\",le=\"0.1\"} 30000\n"}));    // numberOfWorkers * numberOfOperations * 1 (observation)
            EXPECT_THAT(content, HasSubstr(prefix + std::string{"\",le=\"1\"} 60000\n"}));      // numberOfWorkers * numberOfOperations * 2 (observations)
            EXPECT_THAT(content, HasSubstr(prefix + std::string{"\",le=\"10\"} 90000\n"}));     // numberOfWorkers * numberOfOperations * 3 (observations)
            EXPECT_THAT(content, HasSubstr(prefix + std::string{"\",le=\"+Inf\"} 120000\n"}));  // numberOfWorkers * numberOfOperations * 4 (observations)

            // numberOfWorkers * numberOfOperations * 4 (observations)
            EXPECT_THAT(content, HasSubstr(std::string{"family_name_h_"} + std::to_string(i) + std::string{"_count{metric_label_name=\"metric_value_"} + std::to_string(j) + std::string{"\"} 120000\n"}));
            // numberOfWorkers * numberOfOperations * (0.05 + 0.5 + 5.0 + 50.0) = 1666500.0
            EXPECT_THAT(content, ContainsRegex(std::string{"family_name_h_"} + std::to_string(i) + std::string{"_sum\\{metric_label_name=\"metric_value_"} + std::to_string(j) + std::string{"\"\\} 1666500.*\\n"}));
        }
    }
}
