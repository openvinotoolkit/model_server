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
#include "metric_family.hpp"

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include "metric.hpp"

namespace ovms {

MetricFamily::MetricFamily(MetricKind kind, const std::string& name, const std::string& description, prometheus::Registry& registryImplRef) :
    kind(kind),
    name(name),
    description(description),
    registryImplRef(registryImplRef) {}

MetricKind MetricFamily::getKind() const {
    return this->kind;
}

const std::string& MetricFamily::getName() const {
    return this->name;
}

const std::string& MetricFamily::getDesc() const {
    return this->description;
}

std::shared_ptr<Metric> MetricFamily::addMetric(const Metric::Labels& labels, const Metric::BucketBoundaries& bucketBoundaries) {
    switch (this->getKind()) {
    case MetricKind::COUNTER: {
        prometheus::Counter& counterImpl = prometheus::BuildCounter()
                                               .Name(this->getName())
                                               .Help(this->getDesc())
                                               .Register(this->registryImplRef)
                                               .Add(labels);
        return this->metrics.emplace_back(std::make_shared<MetricCounter>(labels, counterImpl));
    }
    case MetricKind::GAUGE: {
        prometheus::Gauge& gaugeImpl = prometheus::BuildGauge()
                                           .Name(this->getName())
                                           .Help(this->getDesc())
                                           .Register(this->registryImplRef)
                                           .Add(labels);
        return this->metrics.emplace_back(std::make_shared<MetricGauge>(labels, gaugeImpl));
    }
    case MetricKind::HISTOGRAM: {
        prometheus::Histogram& histogramImpl = prometheus::BuildHistogram()
                                                    .Name(this->getName())
                                                    .Help(this->getDesc())
                                                    .Register(this->registryImplRef)
                                                    .Add(labels, bucketBoundaries);
        return this->metrics.emplace_back(std::make_shared<MetricHistogram>(labels, bucketBoundaries, histogramImpl));
    }
    default:
        throw std::runtime_error("not implemented");
    }
}

}  // namespace ovms
