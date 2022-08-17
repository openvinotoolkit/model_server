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

template <>
MetricFamily<MetricCounter>::MetricFamily(const std::string& name, const std::string& description, prometheus::Registry& registryImplRef) :
    registryImplRef(registryImplRef),
    familyImplRef(&prometheus::BuildCounter()
                       .Name(name)
                       .Help(description)
                       .Register(this->registryImplRef)) {
}

template <>
MetricFamily<MetricGauge>::MetricFamily(const std::string& name, const std::string& description, prometheus::Registry& registryImplRef) :
    registryImplRef(registryImplRef),
    familyImplRef(&prometheus::BuildGauge()
                       .Name(name)
                       .Help(description)
                       .Register(this->registryImplRef)) {
}

template <>
MetricFamily<MetricHistogram>::MetricFamily(const std::string& name, const std::string& description, prometheus::Registry& registryImplRef) :
    registryImplRef(registryImplRef),
    familyImplRef(&prometheus::BuildHistogram()
                       .Name(name)
                       .Help(description)
                       .Register(this->registryImplRef)) {
}

template <>
std::shared_ptr<MetricCounter> MetricFamily<MetricCounter>::addMetric(const Metric::Labels& labels, const Metric::BucketBoundaries& bucketBoundaries) {
    auto familyImpl = static_cast<prometheus::Family<prometheus::Counter>*>(this->familyImplRef);
    prometheus::Counter& counterImpl = familyImpl->Add(labels);
    return std::make_shared<MetricCounter>(counterImpl);
}

template <>
std::shared_ptr<MetricGauge> MetricFamily<MetricGauge>::addMetric(const Metric::Labels& labels, const Metric::BucketBoundaries& bucketBoundaries) {
    auto familyImpl = static_cast<prometheus::Family<prometheus::Gauge>*>(this->familyImplRef);
    prometheus::Gauge& gaugeImpl = familyImpl->Add(labels);
    return std::make_shared<MetricGauge>(gaugeImpl);
}

template <>
std::shared_ptr<MetricHistogram> MetricFamily<MetricHistogram>::addMetric(const Metric::Labels& labels, const Metric::BucketBoundaries& bucketBoundaries) {
    auto familyImpl = static_cast<prometheus::Family<prometheus::Histogram>*>(this->familyImplRef);
    prometheus::Histogram& histogramImpl = familyImpl->Add(labels, bucketBoundaries);
    return std::make_shared<MetricHistogram>(histogramImpl);
}

template <>
void MetricFamily<MetricCounter>::remove(std::shared_ptr<MetricCounter> metric) {
    auto family = static_cast<prometheus::Family<prometheus::Counter>*>(this->familyImplRef);
    family->Remove(&metric->counterImpl);
}

template <>
void MetricFamily<MetricGauge>::remove(std::shared_ptr<MetricGauge> metric) {
    auto family = static_cast<prometheus::Family<prometheus::Gauge>*>(this->familyImplRef);
    family->Remove(&metric->gaugeImpl);
}

template <>
void MetricFamily<MetricHistogram>::remove(std::shared_ptr<MetricHistogram> metric) {
    auto family = static_cast<prometheus::Family<prometheus::Histogram>*>(this->familyImplRef);
    family->Remove(&metric->histogramImpl);
}

}  // namespace ovms
