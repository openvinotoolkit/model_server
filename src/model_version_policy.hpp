//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <vector>
#include <memory>
#include <algorithm>

#include "modelconfig.hpp"

namespace ovms {

/**
 * @brief Base class for model version policy types
 */
class ModelVersionPolicy {
protected:
    ModelVersionPolicy() {}

public:
    /**
     * @brief Strips out model versions list passed by parameter depending on internal state
     * 
     * @param versions model versions to filter
     * @return Filtered version list
     */
    virtual std::vector<model_version_t> filter(std::vector<model_version_t> versions) const = 0;

    /**
     * @brief Creates default model version policy, by default only one version (highest) should be served
     * 
     * @param highestVersion highest version
     * @return default version policy
     */
    static std::shared_ptr<ModelVersionPolicy> getDefaultVersionPolicy();
};

/**
 * @brief Model version policy that enables all available versions
 */
class AllModelVersionPolicy : public ModelVersionPolicy {
public:
    /**
     * @brief Default constructor, nothing needs to be specified since all versions will be served
     */
    AllModelVersionPolicy() {}

    /**
     * @brief Filters passed versions depending on internal state
     * 
     * @param versions model versions to filter
     * @return Filtered version list
     */
    std::vector<model_version_t> filter(std::vector<model_version_t> versions) const override {
        return versions;
    }
};

/**
 * @brief Model version policy for explicitely specifying which versions should be enabled
 */
class SpecificModelVersionPolicy : public ModelVersionPolicy {
    std::vector<model_version_t> specificVersions;

public:
    /**
     * @brief Default constructor
     * 
     * @param versions list of all model versions that should be served
     */
    SpecificModelVersionPolicy(const std::vector<model_version_t>& versions) :
        specificVersions(versions) {
        std::sort(specificVersions.begin(), specificVersions.end());
    }

    /**
     * @brief Filters passed versions depending on internal state
     * 
     * @param versions model versions to filter
     * @return Filtered version list
     */
    std::vector<model_version_t> filter(std::vector<model_version_t> versions) const override;
};

/**
 * @brief Model version policy for serving only X latest versions
 */
class LatestModelVersionPolicy : public ModelVersionPolicy {
    model_version_t numVersions;

public:
    /**
     * @brief Default constructor
     * 
     * @param numVersions number of latest versions to be served
     */
    LatestModelVersionPolicy(model_version_t numVersions = 1) :
        numVersions(numVersions) {}

    /**
     * @brief Filters passed versions depending on internal state
     * 
     * @param versions model versions to filter
     * @return Filtered version list
     */
    std::vector<model_version_t> filter(std::vector<model_version_t> versions) const override;
};

}  //  namespace ovms
