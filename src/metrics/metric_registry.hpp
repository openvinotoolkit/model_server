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
#pragma once

#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <stdexcept>
#include <typeindex>
#include <type_traits>
#include <utility>
#include <vector>
#include <prometheus/registry.h>

namespace ovms {

template <typename MetricType>
class MetricFamily;
class MetricCounter;
class MetricHistogram;

class MetricRegistry {
public:
    MetricRegistry();
    MetricRegistry(const MetricRegistry&) = delete;
    MetricRegistry(MetricRegistry&&) = delete;
    MetricRegistry& operator=(const MetricRegistry&) = delete;

    template <typename MetricType>
    std::shared_ptr<MetricFamily<MetricType>> createFamily(const std::string& name, const std::string& description) {
        try {
            return std::shared_ptr<MetricFamily<MetricType>>(
                new MetricFamily<MetricType>(name, description, this->registryImpl));
        } catch (std::invalid_argument&) {
            return nullptr;
        }
    }

    template <typename MetricType>
    bool remove(std::shared_ptr<MetricFamily<MetricType>> family);

    // Returns all collected metrics in "Prometheus Text Exposition Format".
    std::string collect() const;

    /// Register a custom metric family. Every registration must use the same type, help text, and histogram buckets.
    template <typename MetricType>
    std::shared_ptr<MetricFamily<MetricType>> registerCustomFamily(const std::string& name, const std::string& help,
        const std::vector<double>& bucketBoundaries = {}) {
        std::lock_guard<std::mutex> lock(customFamiliesMutex);
        const auto existing = customFamilies.find(name);
        if (existing != customFamilies.end()) {
            if (existing->second.help != help || existing->second.type != std::type_index(typeid(MetricType)) ||
                existing->second.bucketBoundaries != bucketBoundaries) {
                throw std::invalid_argument("Custom metric '" + name +
                                            "' was registered with different type, help text, or histogram buckets; "
                                            "all registrations for the same metric name must match");
            }
            return std::static_pointer_cast<MetricFamily<MetricType>>(existing->second.family);
        }
        constexpr bool isHistogram = std::is_same_v<MetricType, MetricHistogram>;
        validateNoHistogramNameCollision(name, isHistogram);
        auto family = createFamily<MetricType>(name, help);
        if (!family) {
            throw std::invalid_argument("Cannot register custom metric family '" + name + "'");
        }
        customFamilies.emplace(name, CustomFamilyRegistration{help, std::type_index(typeid(MetricType)), bucketBoundaries, isHistogram, family});
        return family;
    }

    std::shared_ptr<MetricFamily<MetricCounter>> registerCustomCounterFamily(const std::string& name, const std::string& help);

private:
    void validateNoHistogramNameCollision(const std::string& name, bool isHistogram) const {
        constexpr const char* histogramSuffixes[] = {"_count", "_sum", "_bucket"};
        for (const auto& registration : customFamilies) {
            for (const char* suffix : histogramSuffixes) {
                if ((registration.second.isHistogram && name == registration.first + suffix) ||
                    (isHistogram && registration.first == name + suffix)) {
                    throw std::invalid_argument("Custom metric '" + name +
                                                "' conflicts with a histogram-generated metric name");
                }
            }
        }
    }

    template <typename Family, typename RemoveFn>
    bool removeFamilyAndForgetCustomRegistration(const std::shared_ptr<Family>& family, RemoveFn&& removeFamily) {
        std::lock_guard<std::mutex> lock(customFamiliesMutex);
        if (!removeFamily()) {
            return false;
        }
        for (auto it = customFamilies.begin(); it != customFamilies.end(); ++it) {
            if (it->second.family == family) {
                customFamilies.erase(it);
                break;
            }
        }
        return true;
    }

    struct CustomFamilyRegistration {
        std::string help;
        std::type_index type;
        std::vector<double> bucketBoundaries;
        bool isHistogram;
        std::shared_ptr<void> family;
    };

    prometheus::Registry registryImpl;
    std::mutex customFamiliesMutex;
    std::map<std::string, CustomFamilyRegistration> customFamilies;
};

}  // namespace ovms
