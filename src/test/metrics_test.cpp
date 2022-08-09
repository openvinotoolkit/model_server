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

#include "../metric_registry.hpp"

using namespace ovms;

TEST(Metrics, Ok) {
    MetricRegistry registry;
    auto family = registry.createFamily("name", "desc");
    EXPECT_EQ(family->getDesc(), "desc");

    auto metric = family->addMetric({{"protocol", "grpc"}, {"api", "kfs"}});
    EXPECT_TRUE(metric->hasLabel("protocol"));
    EXPECT_FALSE(metric->hasLabel("wrong"));

    // EXPECT_EQ(registry.collect(), "dwa");
}
